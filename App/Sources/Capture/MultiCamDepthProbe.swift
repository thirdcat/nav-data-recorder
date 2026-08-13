import AVFoundation
import CoreVideo
import Foundation

/// Answers what the LiDAR depth camera actually delivers outside ARKit, while
/// the ultra-wide runs beside it.
///
/// The recorder gets depth from `ARFrame.sceneDepth`, which arrives with a
/// per-pixel `confidenceMap` — and that map is what makes the depth safe to
/// train on, because it is how the unreliable returns get dropped. Leaving
/// ARKit for a multi-camera session means leaving that map behind, and the
/// question is what replaces it.
///
/// Apple's answer is in `AVCaptureDepthDataOutput`: with filtering disabled the
/// LiDAR camera **omits low-confidence points** rather than labelling them. So
/// the confidence signal survives, demoted from three levels to a hole. That is
/// no loss in practice — the depth-supervised trainers this feeds threshold the
/// confidence into a binary mask anyway.
///
/// The hazard is the default. **Filtering is on unless it is turned off**, and
/// filtering fills holes: it replaces "no measurement here" with a plausible
/// interpolated number that is indistinguishable from a real one downstream.
///
/// Three things this reports that a single frame could not:
///
/// - **Intrinsic stability.** A first run found `fx` differing by 5 % between
///   two configurations of the same locked-focus camera, with the unfiltered
///   pass reporting `cx` at exactly the image centre — a value a real principal
///   point never takes. Five percent of focal length is 10 cm of back-projection
///   error at the frame edge at 3 m, so whether that was noise, a placeholder,
///   or a genuine difference has to be settled before the intrinsics are used.
/// - **What the frame rate costs.** `systemPressureCost` above 1.0 means the
///   system will eventually throttle. The first run measured 1.844 at 60 fps —
///   a rate this project has no use for, since it stores images at 5 Hz. The
///   sweep says what the rates we would actually pick cost.
/// - **How hard the scene was.** A depth measurement is only as interesting as
///   what was in front of the lens; the first run had everything inside 1.2 m,
///   which is the easiest case there is.
///
/// Not on the recording path. It runs alone, writes nothing, and produces a
/// text report meant to be read once and pasted into a design discussion.
final class MultiCamDepthProbe: NSObject {

    enum ProbeError: LocalizedError {
        case unsupported(String)

        var errorDescription: String? {
            switch self {
            case .unsupported(let why): return why
            }
        }
    }

    /// Delivered on the main queue.
    var onStatus: ((String) -> Void)?
    var onFinished: ((Result<String, Error>) -> Void)?

    /// Rates to price. 60 is what the format offers and what the first run
    /// accidentally used; 5 is what the recorder actually stores.
    private static let ratesToPrice = [60, 30, 15, 5]
    /// Enough frames to see whether the calibration moves, few enough that the
    /// pass stays short on a device that is already warm.
    private static let framesPerPass = 12
    /// The rate the depth passes run at — what a real scan would choose.
    private static let passRate = 30

    private let session = AVCaptureMultiCamSession()
    private let depthOutput = AVCaptureDepthDataOutput()
    private let wideVideoOutput = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "nav.multicam.probe")

    private var lidar: AVCaptureDevice?
    private var ultraWide: AVCaptureDevice?
    private var notes: [String] = []
    private var costs: [(rate: Int, hardware: Float, pressure: Float)] = []

    private var pass = 0
    private var collecting = false
    private var passes: [Int: [DepthSample]] = [:]
    private var ultraWideFrames = 0
    private var finished = false

    private struct DepthSample {
        var width = 0
        var height = 0
        var type = ""
        var accuracy = ""
        var quality = ""
        var filtered = false
        var finiteFraction = 0.0
        var beyondTwoMetres = 0.0
        var nearMetres = 0.0
        var medianMetres = 0.0
        var farMetres = 0.0
        var fx: Float = 0
        var fy: Float = 0
        var cx: Float = 0
        var cy: Float = 0
        var hasCalibration = false
        var distortionEntries = 0
        var extrinsicTranslation: (Float, Float, Float) = (0, 0, 0)
    }

    // MARK: - Running

    func run() {
        queue.async { [weak self] in
            guard let self else { return }
            do {
                try self.configure()
                self.status("starting session…")
                self.session.startRunning()
                self.queue.asyncAfter(deadline: .now() + 1.0) { self.priceRate(0) }
            } catch {
                self.complete(.failure(error))
            }
        }
    }

    func cancel() {
        queue.async { [weak self] in
            guard let self, !self.finished else { return }
            self.finished = true
            if self.session.isRunning { self.session.stopRunning() }
        }
    }

    /// Set a frame rate, let the session settle, and record what it costs.
    ///
    /// The costs are read while running because `systemPressureCost` describes
    /// a session under load; read before `startRunning` it is describing an
    /// intention rather than a measurement.
    private func priceRate(_ index: Int) {
        guard !finished else { return }
        guard index < Self.ratesToPrice.count else {
            beginPass(0)
            return
        }
        let rate = Self.ratesToPrice[index]
        setRate(rate)
        status("pricing \(rate) fps…")
        queue.asyncAfter(deadline: .now() + 0.9) {
            guard !self.finished else { return }
            self.costs.append((rate, self.session.hardwareCost, self.session.systemPressureCost))
            self.priceRate(index + 1)
        }
    }

    private func setRate(_ fps: Int) {
        guard let device = lidar else { return }
        do {
            try device.lockForConfiguration()
            let duration = CMTime(value: 1, timescale: CMTimeScale(fps))
            device.activeVideoMinFrameDuration = duration
            device.activeVideoMaxFrameDuration = duration
            device.unlockForConfiguration()
        } catch {
            notes.append("! could not set \(fps) fps: \(error.localizedDescription)")
        }
    }

    private func beginPass(_ index: Int) {
        guard !finished else { return }
        pass = index
        setRate(Self.passRate)
        depthOutput.isFilteringEnabled = (index == 1)
        passes[index] = []
        collecting = true
        status(index == 0 ? "collecting unfiltered frames…" : "collecting filtered frames…")
        queue.asyncAfter(deadline: .now() + 2.0) { self.endPass(index) }
    }

    private func endPass(_ index: Int) {
        guard !finished else { return }
        collecting = false
        if index == 0 {
            guard let frames = passes[0], !frames.isEmpty else {
                complete(.failure(ProbeError.unsupported(
                    "No depth frame arrived in 2 s with filtering off.\n\n"
                    + notes.joined(separator: "\n"))))
                return
            }
            beginPass(1)
            return
        }
        session.stopRunning()
        complete(.success(report()))
    }

    // MARK: - Configuration

    private func configure() throws {
        guard AVCaptureMultiCamSession.isMultiCamSupported else {
            throw ProbeError.unsupported("This device does not support multi-camera capture.")
        }

        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInLiDARDepthCamera, .builtInUltraWideCamera],
            mediaType: .video,
            position: .back)

        guard let lidar = discovery.devices.first(where: { $0.deviceType == .builtInLiDARDepthCamera }) else {
            throw ProbeError.unsupported("No LiDAR depth camera on this device.")
        }
        let ultraWide = discovery.devices.first { $0.deviceType == .builtInUltraWideCamera }
        self.lidar = lidar
        self.ultraWide = ultraWide

        let pairSupported = discovery.supportedMultiCamDeviceSets.contains { set in
            let types = set.map { $0.deviceType }
            return types.contains(.builtInLiDARDepthCamera)
                && types.contains(.builtInUltraWideCamera)
        }
        notes.append("LiDAR + ultra-wide in supportedMultiCamDeviceSets: \(pairSupported)")

        session.beginConfiguration()
        defer { session.commitConfiguration() }

        // Multi-cam sessions carry no preset; every device states its own
        // format, and connections are made by hand so each output is fed by the
        // port that is meant to feed it.
        let lidarInput = try AVCaptureDeviceInput(device: lidar)
        guard session.canAddInput(lidarInput) else {
            throw ProbeError.unsupported("Cannot add the LiDAR camera as a multi-cam input.")
        }
        session.addInputWithNoConnections(lidarInput)

        try selectFormat(on: lidar)

        guard session.canAddOutput(depthOutput) else {
            throw ProbeError.unsupported("Cannot add a depth output.")
        }
        depthOutput.isFilteringEnabled = false
        depthOutput.alwaysDiscardsLateDepthData = true
        session.addOutputWithNoConnections(depthOutput)
        depthOutput.setDelegate(self, callbackQueue: queue)

        // The depth port only exists once a depth format is active, which is
        // why this is read after `selectFormat` and not with the other inputs.
        let depthPorts = lidarInput.ports(for: .depthData,
                                          sourceDeviceType: .builtInLiDARDepthCamera,
                                          sourceDevicePosition: lidar.position)
        guard !depthPorts.isEmpty else {
            throw ProbeError.unsupported("The LiDAR input exposes no depth port in this configuration.")
        }
        let depthConnection = AVCaptureConnection(inputPorts: depthPorts, output: depthOutput)
        guard session.canAddConnection(depthConnection) else {
            throw ProbeError.unsupported("Cannot connect the depth port to the depth output.")
        }
        session.addConnection(depthConnection)

        // The ultra-wide is here to make the cost honest. Depth on its own says
        // nothing about whether depth survives *while a second lens is running*,
        // and that pairing is the whole reason to leave ARKit.
        if let ultraWide, pairSupported {
            do {
                let uwInput = try AVCaptureDeviceInput(device: ultraWide)
                let ports = uwInput.ports(for: .video,
                                          sourceDeviceType: .builtInUltraWideCamera,
                                          sourceDevicePosition: ultraWide.position)
                if session.canAddInput(uwInput), session.canAddOutput(wideVideoOutput),
                   !ports.isEmpty {
                    session.addInputWithNoConnections(uwInput)
                    session.addOutputWithNoConnections(wideVideoOutput)
                    wideVideoOutput.setSampleBufferDelegate(self, queue: queue)
                    let connection = AVCaptureConnection(inputPorts: ports, output: wideVideoOutput)
                    if session.canAddConnection(connection) {
                        session.addConnection(connection)
                        notes.append("ultra-wide added alongside")
                    } else {
                        notes.append("! ultra-wide connection refused — depth measured alone")
                    }
                } else {
                    notes.append("! ultra-wide input or video port unavailable — depth measured alone")
                }
            } catch {
                notes.append("! ultra-wide input failed: \(error.localizedDescription)")
            }
        } else {
            notes.append("ultra-wide not paired with LiDAR on this device")
        }
    }

    /// Pick the multi-cam format whose depth is largest, preferring 4:3 colour.
    ///
    /// Largest depth first because depth is the scarce signal — ARKit's is
    /// 256x192 and anything above is a gain — and 4:3 second because on this
    /// device the 16:9 formats drop to 320x180 depth, trading a quarter of the
    /// depth rows for a colour aspect the capture does not need.
    private func selectFormat(on device: AVCaptureDevice) throws {
        let candidates = device.formats.filter {
            $0.isMultiCamSupported && !$0.supportedDepthDataFormats.isEmpty
        }
        guard !candidates.isEmpty else {
            throw ProbeError.unsupported("No multi-cam format on the LiDAR camera carries depth.")
        }

        func depthPixels(_ format: AVCaptureDevice.Format) -> Int {
            format.supportedDepthDataFormats.map {
                let d = CMVideoFormatDescriptionGetDimensions($0.formatDescription)
                return Int(d.width) * Int(d.height)
            }.max() ?? 0
        }
        func isFourThree(_ format: AVCaptureDevice.Format) -> Bool {
            let d = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            guard d.height > 0 else { return false }
            return abs(Double(d.width) / Double(d.height) - 4.0 / 3.0) < 0.02
        }
        func colourPixels(_ format: AVCaptureDevice.Format) -> Int {
            let d = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            return Int(d.width) * Int(d.height)
        }

        let chosen = candidates.max { a, b in
            let (da, db) = (depthPixels(a), depthPixels(b))
            if da != db { return da < db }
            let (fa, fb) = (isFourThree(a), isFourThree(b))
            if fa != fb { return !fa && fb }
            return colourPixels(a) < colourPixels(b)
        }
        guard let chosen else {
            throw ProbeError.unsupported("Could not choose a format.")
        }
        let depthFormat = chosen.supportedDepthDataFormats.max { a, b in
            score(a) < score(b)
        }

        try device.lockForConfiguration()
        device.activeFormat = chosen
        if let depthFormat { device.activeDepthDataFormat = depthFormat }
        // Locked focus keeps the intrinsics fixed across the session, which is
        // what lets a reconstruction share one camera model instead of one per
        // frame. Locked exposure keeps appearance comparable between sessions,
        // which is what a multi-session merge needs and the ARKit path does not
        // offer.
        if device.isFocusModeSupported(.locked) { device.focusMode = .locked }
        if device.isExposureModeSupported(.locked) { device.exposureMode = .locked }
        device.unlockForConfiguration()

        let c = CMVideoFormatDescriptionGetDimensions(chosen.formatDescription)
        notes.append("colour format \(c.width)x\(c.height) fov "
                     + String(format: "%.1f", Double(chosen.videoFieldOfView)) + "°")
        if let depthFormat {
            let d = CMVideoFormatDescriptionGetDimensions(depthFormat.formatDescription)
            notes.append("depth format  \(d.width)x\(d.height) \(fourCC(depthFormat))")
        }
        notes.append("focus lock \(device.focusMode == .locked), "
                     + "exposure lock \(device.exposureMode == .locked)")
    }

    private func score(_ format: AVCaptureDevice.Format) -> Int {
        let subtype = CMFormatDescriptionGetMediaSubType(format.formatDescription)
        let d = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
        let pixels = Int(d.width) * Int(d.height)
        let kind: Int
        switch subtype {
        case kCVPixelFormatType_DepthFloat32: kind = 3
        case kCVPixelFormatType_DepthFloat16: kind = 2
        case kCVPixelFormatType_DisparityFloat32: kind = 1
        default: kind = 0
        }
        return kind * 100_000_000 + pixels
    }

    // MARK: - Measurement

    private func measure(_ depthData: AVDepthData) -> DepthSample {
        var sample = DepthSample()
        sample.filtered = depthData.isDepthDataFiltered
        sample.type = fourCC(depthData.depthDataType)
        switch depthData.depthDataAccuracy {
        case .absolute: sample.accuracy = "absolute (metric)"
        case .relative: sample.accuracy = "relative (NO metric scale)"
        @unknown default: sample.accuracy = "unknown"
        }
        switch depthData.depthDataQuality {
        case .high: sample.quality = "high"
        case .low: sample.quality = "low"
        @unknown default: sample.quality = "unknown"
        }

        var converted = depthData
        if depthData.depthDataType != kCVPixelFormatType_DepthFloat32,
           depthData.availableDepthDataTypes.contains(kCVPixelFormatType_DepthFloat32) {
            converted = depthData.converting(toDepthDataType: kCVPixelFormatType_DepthFloat32)
        }
        let map = converted.depthDataMap

        CVPixelBufferLockBaseAddress(map, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(map, .readOnly) }
        sample.width = CVPixelBufferGetWidth(map)
        sample.height = CVPixelBufferGetHeight(map)

        if CVPixelBufferGetPixelFormatType(map) == kCVPixelFormatType_DepthFloat32,
           let base = CVPixelBufferGetBaseAddress(map) {
            let bytesPerRow = CVPixelBufferGetBytesPerRow(map)
            var finite: [Float] = []
            finite.reserveCapacity(sample.width * sample.height / 2)
            for y in 0..<sample.height {
                let row = base.advanced(by: y * bytesPerRow)
                    .assumingMemoryBound(to: Float32.self)
                for x in 0..<sample.width {
                    let value = row[x]
                    if value.isFinite && value > 0 { finite.append(value) }
                }
            }
            let total = sample.width * sample.height
            if total > 0 {
                sample.finiteFraction = Double(finite.count) / Double(total)
            }
            if !finite.isEmpty {
                // How much of the frame is far away, so a reader can tell an
                // easy scene from a hard one without having been in the room.
                sample.beyondTwoMetres = Double(finite.filter { $0 > 2.0 }.count)
                    / Double(finite.count)
                finite.sort()
                sample.nearMetres = Double(finite[finite.count / 100])
                sample.medianMetres = Double(finite[finite.count / 2])
                sample.farMetres = Double(finite[min(finite.count - 1, finite.count * 99 / 100)])
            }
        }

        if let calibration = converted.cameraCalibrationData {
            sample.hasCalibration = true
            let k = calibration.intrinsicMatrix
            sample.fx = k.columns.0.x
            sample.fy = k.columns.1.y
            sample.cx = k.columns.2.x
            sample.cy = k.columns.2.y
            sample.distortionEntries = (calibration.lensDistortionLookupTable?.count ?? 0) / 4
            let t = calibration.extrinsicMatrix.columns.3
            sample.extrinsicTranslation = (t.x, t.y, t.z)
        }
        return sample
    }

    // MARK: - Report

    private func report() -> String {
        var lines: [String] = ["# Multi-cam depth probe", ""]
        lines.append(contentsOf: notes)
        lines.append("ultra-wide frames seen  \(ultraWideFrames)")
        lines.append("")

        lines.append("## What each frame rate costs")
        lines.append("   hardware > 1.0 means the session cannot run at all;")
        lines.append("   pressure > 1.0 means it runs and the system throttles it later.")
        lines.append("   rate   hardware   pressure")
        for cost in costs {
            lines.append(String(format: "   %3d fps   %6.3f     %6.3f",
                                cost.rate, Double(cost.hardware), Double(cost.pressure)))
        }
        lines.append("")

        for (index, label) in [(0, "filtering OFF — the mode to use"),
                               (1, "filtering ON — the default, and wrong here")] {
            lines.append("## \(label)")
            guard let frames = passes[index], let first = frames.first else {
                lines.append("   (no frame arrived)")
                lines.append("")
                continue
            }
            lines.append("   frames        \(frames.count) at \(Self.passRate) fps")
            lines.append("   map           \(first.width)x\(first.height) = "
                         + "\(first.width * first.height) px "
                         + "(ARKit sceneDepth is 256x192 = 49152)")
            lines.append("   type          \(first.type)")
            lines.append("   accuracy      \(first.accuracy)")
            lines.append("   quality       \(first.quality)")
            lines.append("   isFiltered    \(first.filtered)")

            let valid = frames.map { $0.finiteFraction }
            lines.append(String(format: "   valid pixels  %.1f%% mean, %.1f–%.1f%% over the pass",
                                mean(valid) * 100, (valid.min() ?? 0) * 100, (valid.max() ?? 0) * 100))
            lines.append(String(format: "   depth p1/med/p99  %.2f / %.2f / %.2f m",
                                first.nearMetres, first.medianMetres, first.farMetres))
            lines.append(String(format: "   beyond 2 m    %.1f%% of valid pixels  "
                                + "(a low number means an easy scene)",
                                first.beyondTwoMetres * 100))
            lines.append(contentsOf: calibrationLines(frames))
            lines.append("")
        }

        if let off = passes[0]?.first, let on = passes[1]?.first,
           let offAll = passes[0], let onAll = passes[1] {
            let invented = (mean(onAll.map { $0.finiteFraction })
                            - mean(offAll.map { $0.finiteFraction })) * 100
            lines.append("## What filtering would have invented")
            lines.append(String(format: "   %.1f%% of the frame gains a depth value it did not measure.",
                                invented))
            lines.append(String(format: "   Measured on a scene with %.1f%% of its depth beyond 2 m.",
                                off.beyondTwoMetres * 100))
            lines.append("   Those pixels are indistinguishable from measured ones downstream,")
            lines.append("   which is why a depth loss must be fed the unfiltered map.")
            _ = on
        }
        return lines.joined(separator: "\n")
    }

    /// Intrinsics across the pass, because a single frame cannot say whether
    /// they hold still — and a focal length that moves is a scale that moves.
    private func calibrationLines(_ frames: [DepthSample]) -> [String] {
        guard let first = frames.first, first.hasCalibration else {
            return ["   calibration   ABSENT — no intrinsics with the depth frame"]
        }
        let fxs = frames.map { Double($0.fx) }
        let cxs = frames.map { Double($0.cx) }
        let cys = frames.map { Double($0.cy) }
        let exactlyCentred = frames.filter { $0.cx == 960.0 }.count
        var lines = [
            String(format: "   fx            %.2f  (spread %.2f over the pass)",
                   mean(fxs), (fxs.max() ?? 0) - (fxs.min() ?? 0)),
            String(format: "   cx / cy       %.2f / %.2f  (cx spread %.2f, cy spread %.2f)",
                   mean(cxs), mean(cys),
                   (cxs.max() ?? 0) - (cxs.min() ?? 0), (cys.max() ?? 0) - (cys.min() ?? 0)),
            "   cx exactly 960.00 in \(exactlyCentred) of \(frames.count) frames"
                + (exactlyCentred > 0 ? "  <- a real principal point never is" : ""),
            "   distortion    lookup table \(first.distortionEntries) entries",
            String(format: "   extrinsics    translation (mm) %.2f, %.2f, %.2f",
                   Double(first.extrinsicTranslation.0),
                   Double(first.extrinsicTranslation.1),
                   Double(first.extrinsicTranslation.2))
        ]
        if (fxs.max() ?? 0) - (fxs.min() ?? 0) < 0.01 {
            lines.append("   -> the calibration holds still within this pass")
        }
        return lines
    }

    private func mean(_ values: [Double]) -> Double {
        values.isEmpty ? 0 : values.reduce(0, +) / Double(values.count)
    }

    // MARK: - Plumbing

    private func status(_ text: String) {
        DispatchQueue.main.async { self.onStatus?(text) }
    }

    private func complete(_ result: Result<String, Error>) {
        guard !finished else { return }
        finished = true
        if session.isRunning { session.stopRunning() }
        DispatchQueue.main.async { self.onFinished?(result) }
    }

    private func fourCC(_ code: OSType) -> String {
        let bytes: [UInt8] = [UInt8((code >> 24) & 0xff), UInt8((code >> 16) & 0xff),
                              UInt8((code >> 8) & 0xff), UInt8(code & 0xff)]
        let text = String(bytes: bytes, encoding: .ascii) ?? String(format: "0x%08x", code)
        switch text {
        case "hdep": return "hdep (float16 depth)"
        case "fdep": return "fdep (float32 depth)"
        case "hdis": return "hdis (float16 disparity)"
        case "fdis": return "fdis (float32 disparity)"
        default: return text
        }
    }

    private func fourCC(_ format: AVCaptureDevice.Format) -> String {
        fourCC(CMFormatDescriptionGetMediaSubType(format.formatDescription))
    }
}

extension MultiCamDepthProbe: AVCaptureDepthDataOutputDelegate {
    func depthDataOutput(_ output: AVCaptureDepthDataOutput,
                         didOutput depthData: AVDepthData,
                         timestamp: CMTime,
                         connection: AVCaptureConnection) {
        guard collecting else { return }
        guard var frames = passes[pass], frames.count < Self.framesPerPass else { return }
        frames.append(measure(depthData))
        passes[pass] = frames
    }
}

extension MultiCamDepthProbe: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        ultraWideFrames += 1
    }
}
