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
/// interpolated number that is indistinguishable from a real one downstream. A
/// depth loss fed filtered depth learns invented geometry, confidently. So this
/// probe captures the same scene both ways and reports how much of the frame
/// changes — which is the size of the mistake, in the units it would be made.
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

    private let session = AVCaptureMultiCamSession()
    private let depthOutput = AVCaptureDepthDataOutput()
    private let wideVideoOutput = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "nav.multicam.probe")

    private var lidar: AVCaptureDevice?
    private var ultraWide: AVCaptureDevice?
    private var notes: [String] = []

    /// Frames are collected in two passes: unfiltered first, because that is
    /// the mode the project would actually use, then filtered for comparison.
    private var pass = 0
    private var samples: [Int: DepthSample] = [:]
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
        var nearMetres = 0.0
        var medianMetres = 0.0
        var farMetres = 0.0
        var calibration: [String] = []
    }

    // MARK: - Running

    func run() {
        queue.async { [weak self] in
            guard let self else { return }
            do {
                try self.configure()
                self.status("starting session…")
                self.session.startRunning()
                // Two seconds is generous for a 30 Hz stream and short enough
                // that a device which never delivers a frame says so quickly.
                self.queue.asyncAfter(deadline: .now() + 2.5) { self.advance() }
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

    /// Move from the unfiltered pass to the filtered one, then report.
    private func advance() {
        guard !finished else { return }
        if pass == 0 {
            guard samples[0] != nil else {
                complete(.failure(ProbeError.unsupported(
                    "No depth frame arrived in 2.5 s with filtering off.\n\n"
                    + notes.joined(separator: "\n"))))
                return
            }
            pass = 1
            depthOutput.isFilteringEnabled = true
            notes.append("second pass: isFilteringEnabled = true")
            status("second pass, filtering on…")
            queue.asyncAfter(deadline: .now() + 2.0) { self.advance() }
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
        // Off, and set before the first frame. This is the whole point of the
        // probe: filtering on is the default, and it invents depth.
        depthOutput.isFilteringEnabled = false
        depthOutput.alwaysDiscardsLateDepthData = true
        session.addOutputWithNoConnections(depthOutput)
        depthOutput.setDelegate(self, callbackQueue: queue)
        notes.append("first pass: isFilteringEnabled = false")

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

        // The ultra-wide is here to make the measurement honest. Depth on its
        // own tells us nothing about whether depth survives *while a second
        // lens is running*, and that pairing is the whole reason to leave ARKit.
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
    /// Largest depth first because depth is the scarce signal here — ARKit's is
    /// 256x192 and anything above that is a gain — and 4:3 second because a
    /// 16:9 format is the same sensor with the top and bottom thrown away,
    /// which costs vertical field of view the capture cannot get back.
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
        // Float32 depth over float16, and depth over disparity: the project
        // works in metres and converts nothing it does not have to.
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
                     + String(format: "%.1f", chosen.videoFieldOfView) + "°")
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
        // Depth beats disparity, float32 beats float16, then size.
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

        // Convert once to float32 depth so the pixel walk has one case rather
        // than four, and so disparity — if that is what arrived — becomes the
        // metres the rest of the project speaks in.
        var converted = depthData
        if depthData.depthDataType != kCVPixelFormatType_DepthFloat32,
           depthData.availableDepthDataTypes.contains(NSNumber(value: kCVPixelFormatType_DepthFloat32)) {
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
                finite.sort()
                sample.nearMetres = Double(finite[finite.count / 100])
                sample.medianMetres = Double(finite[finite.count / 2])
                sample.farMetres = Double(finite[min(finite.count - 1, finite.count * 99 / 100)])
            }
        }

        if let calibration = converted.cameraCalibrationData {
            let k = calibration.intrinsicMatrix
            let ref = calibration.intrinsicMatrixReferenceDimensions
            sample.calibration = [
                "   intrinsics    fx=\(fmt(k.columns.0.x)) fy=\(fmt(k.columns.1.y)) "
                    + "cx=\(fmt(k.columns.2.x)) cy=\(fmt(k.columns.2.y))",
                "   reference     \(Int(ref.width))x\(Int(ref.height))",
                "   pixel size    \(fmt(calibration.pixelSize)) mm",
                "   distortion    lookup table "
                    + (calibration.lensDistortionLookupTable != nil
                       ? "present (\((calibration.lensDistortionLookupTable?.count ?? 0) / 4) entries)"
                       : "ABSENT"),
                "   extrinsics    translation (mm) "
                    + "\(fmt(calibration.extrinsicMatrix.columns.3.x)), "
                    + "\(fmt(calibration.extrinsicMatrix.columns.3.y)), "
                    + "\(fmt(calibration.extrinsicMatrix.columns.3.z))"
            ]
        } else {
            sample.calibration = ["   calibration   ABSENT — no intrinsics with the depth frame"]
        }
        return sample
    }

    // MARK: - Report

    private func report() -> String {
        var lines: [String] = ["# Multi-cam depth probe", ""]
        lines.append(contentsOf: notes)
        lines.append("ultra-wide frames seen  \(ultraWideFrames)")
        // `hardwareCost` above 1.0 is the session refusing to run at these
        // formats — the number that says whether this pairing is affordable at
        // the resolution the plan wants, rather than merely listed as supported.
        lines.append(String(format: "hardware cost           %.3f  (must be <= 1.0 to run)",
                            Double(session.hardwareCost)))
        lines.append(String(format: "system pressure cost    %.3f",
                            Double(session.systemPressureCost)))
        lines.append("")

        for (index, label) in [(0, "filtering OFF — the mode to use"),
                               (1, "filtering ON — the default, and wrong here")] {
            lines.append("## \(label)")
            guard let s = samples[index] else {
                lines.append("   (no frame arrived)")
                lines.append("")
                continue
            }
            lines.append("   map           \(s.width)x\(s.height) = \(s.width * s.height) px "
                         + "(ARKit sceneDepth is 256x192 = 49152)")
            lines.append("   type          \(s.type)")
            lines.append("   accuracy      \(s.accuracy)")
            lines.append("   quality       \(s.quality)")
            lines.append("   isFiltered    \(s.filtered)")
            lines.append(String(format: "   valid pixels  %.1f%%", s.finiteFraction * 100))
            lines.append(String(format: "   depth p1/med/p99  %.2f / %.2f / %.2f m",
                                s.nearMetres, s.medianMetres, s.farMetres))
            lines.append(contentsOf: s.calibration)
            lines.append("")
        }

        if let off = samples[0], let on = samples[1] {
            let invented = (on.finiteFraction - off.finiteFraction) * 100
            lines.append("## What filtering would have invented")
            lines.append(String(format: "   %.1f%% of the frame gains a depth value it did not measure.",
                                invented))
            lines.append("   Those pixels are indistinguishable from measured ones downstream,")
            lines.append("   which is why a depth loss must be fed the unfiltered map.")
        }
        return lines.joined(separator: "\n")
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

    private func fmt(_ value: Float) -> String { String(format: "%.2f", Double(value)) }

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
        // One sample per pass. Later frames of the same pass would only measure
        // how the scene drifted while the probe was running.
        guard samples[pass] == nil else { return }
        samples[pass] = measure(depthData)
        status(pass == 0 ? "unfiltered frame captured" : "filtered frame captured")
    }
}

extension MultiCamDepthProbe: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        ultraWideFrames += 1
    }
}
