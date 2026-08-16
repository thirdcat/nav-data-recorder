import AVFoundation
import CoreVideo
import Foundation
import UIKit

/// Records the ultra-wide lens alongside LiDAR depth, with no ARKit and no poses.
///
/// **Why this exists.** `docs/POSE.md` measured the field-of-view gradient by
/// narrowing the wide camera: thirteen sessions of thirteen got worse, median
/// 2.05x, monotone in ten. Widening cannot be synthesised from a narrower
/// capture, so the ultra-wide side of that curve has only ever been an
/// extrapolation. `ARWorldTrackingConfiguration.supportedVideoFormats` offers
/// twenty-two formats on this device and every one is `WideAngleCamera`, so
/// reaching the ultra-wide means leaving ARKit — and leaving ARKit means
/// leaving its pose behind.
///
/// That is the trade this recorder makes deliberately. It writes **no
/// `pose.jsonl`**. Poses come from `eval/pi3_chain.py` offline, which is an RGB
/// feedforward model that needs no tracker, and metric scale comes from the
/// LiDAR depth recorded here beside the images.
///
/// **The confidence map does not survive, and that is handled.** ARKit's
/// `sceneDepth` carries a per-pixel `confidenceMap`; `AVCaptureDepthDataOutput`
/// does not. With filtering *disabled* the LiDAR camera omits low-confidence
/// points instead of labelling them, so the signal survives demoted to a hole —
/// which is what the depth-supervised trainers downstream do with it anyway.
/// Filtering is on by default and fills holes with plausible interpolation
/// indistinguishable from a real return, so it is turned off explicitly.
///
/// **The layout is the recorder's own**, so every tool in `tools/` reads these
/// sessions unchanged: `frames/` and `frames.jsonl` for the ultra-wide JPEGs,
/// `depth.bin` and `depth.jsonl` for the depth, `motion.jsonl` for the IMU. The
/// absence of `pose.jsonl` is the one difference, and `manifest.json` says so
/// rather than leaving a reader to infer it from a missing file.
final class MultiCamRecorder: NSObject {

    enum RecorderError: LocalizedError {
        case unsupported(String)

        var errorDescription: String? {
            switch self {
            case .unsupported(let why): return why
            }
        }
    }

    /// Delivered on the main queue.
    var onStatus: ((String) -> Void)?
    var onFinished: ((Result<URL, Error>) -> Void)?

    private let session = AVCaptureMultiCamSession()
    private let depthOutput = AVCaptureDepthDataOutput()
    private let ultraWideOutput = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "nav.multicam.recorder")
    private lazy var motion = MotionRecorder(queue: queue)

    private var directory: URL?
    private var frameIndex: JSONLWriter?
    private var depthIndex: JSONLWriter?
    private var depthData: BufferedFileWriter?
    private var motionWriter: JSONLWriter?
    private var eventWriter: JSONLWriter?

    private var ultraWide: AVCaptureDevice?
    private var lidar: AVCaptureDevice?
    private var running = false
    private var stopping = false
    private var imageCount = 0
    private var depthCount = 0
    private var droppedImages = 0
    private var notes: [String] = []

    /// Images are stored at this rate, matching the ARKit recorder's stills
    /// mode. The sensor runs faster; everything between is dropped rather than
    /// written, because coverage comes from walking further and not from
    /// storing the same view again.
    private let stillsHz: Double
    private var lastImageStamp: Double = -.infinity

    init(stillsHz: Double = 5.0) {
        self.stillsHz = stillsHz
        super.init()
    }

    // MARK: - Lifecycle

    func start() {
        queue.async { [weak self] in
            guard let self else { return }
            do {
                try self.prepareDirectory()
                try self.configure()
                self.session.startRunning()
                self.motion.onMotion = { [weak self] sample in
                    try? self?.motionWriter?.write(sample)
                }
                self.motion.start(hz: 100, magnetometerCorrected: false)
                self.running = true
                self.status("recording — ultra-wide + LiDAR depth, no poses")
            } catch {
                self.finish(.failure(error))
            }
        }
    }

    func stop() {
        queue.async { [weak self] in
            guard let self, self.running, !self.stopping else { return }
            self.stopping = true
            self.session.stopRunning()
            self.motion.stop()
            do {
                try self.closeFiles()
                guard let dir = self.directory else { throw CocoaError(.fileNoSuchFile) }
                self.finish(.success(dir))
            } catch {
                self.finish(.failure(error))
            }
        }
    }

    // MARK: - Setup

    private func prepareDirectory() throws {
        let id = SessionStore.makeSessionID()
        let dir = SessionStore.root.appendingPathComponent(id, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: dir.appendingPathComponent(SessionStore.Filename.framesDirectory,
                                           isDirectory: true),
            withIntermediateDirectories: true)
        directory = dir
        frameIndex = try JSONLWriter(
            url: dir.appendingPathComponent(SessionStore.Filename.frameIndex))
        depthIndex = try JSONLWriter(
            url: dir.appendingPathComponent(SessionStore.Filename.depthIndex))
        depthData = try BufferedFileWriter(
            url: dir.appendingPathComponent(SessionStore.Filename.depthData))
        motionWriter = try JSONLWriter(
            url: dir.appendingPathComponent(SessionStore.Filename.motion))
        eventWriter = try JSONLWriter(
            url: dir.appendingPathComponent(SessionStore.Filename.events))
    }

    private func configure() throws {
        guard AVCaptureMultiCamSession.isMultiCamSupported else {
            throw RecorderError.unsupported("This device does not support multi-camera capture.")
        }
        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInLiDARDepthCamera, .builtInUltraWideCamera],
            mediaType: .video, position: .back)
        guard let lidar = discovery.devices.first(where: {
            $0.deviceType == .builtInLiDARDepthCamera
        }) else {
            throw RecorderError.unsupported("No LiDAR depth camera on this device.")
        }
        guard let ultraWide = discovery.devices.first(where: {
            $0.deviceType == .builtInUltraWideCamera
        }) else {
            throw RecorderError.unsupported("No ultra-wide camera on this device.")
        }
        let pairSupported = discovery.supportedMultiCamDeviceSets.contains { set in
            let types = set.map { $0.deviceType }
            return types.contains(.builtInLiDARDepthCamera)
                && types.contains(.builtInUltraWideCamera)
        }
        guard pairSupported else {
            throw RecorderError.unsupported(
                "LiDAR and ultra-wide cannot run together on this device.")
        }
        self.lidar = lidar
        self.ultraWide = ultraWide

        session.beginConfiguration()
        defer { session.commitConfiguration() }

        // A multi-cam session carries no preset: every device states its own
        // format and every connection is made by hand, so each output is fed by
        // the port meant to feed it.
        let lidarInput = try AVCaptureDeviceInput(device: lidar)
        guard session.canAddInput(lidarInput) else {
            throw RecorderError.unsupported("Cannot add the LiDAR camera as a multi-cam input.")
        }
        session.addInputWithNoConnections(lidarInput)
        try selectDepthFormat(on: lidar)

        guard session.canAddOutput(depthOutput) else {
            throw RecorderError.unsupported("Cannot add a depth output.")
        }
        // Off, deliberately: filtering replaces a missing return with an
        // interpolated one that no downstream consumer can tell from a
        // measurement. A hole is information; a plausible number is not.
        depthOutput.isFilteringEnabled = false
        depthOutput.alwaysDiscardsLateDepthData = true
        session.addOutputWithNoConnections(depthOutput)
        depthOutput.setDelegate(self, callbackQueue: queue)

        // The depth port only exists once a depth format is active, which is
        // why this is read after the format is chosen.
        let depthPorts = lidarInput.ports(for: .depthData,
                                          sourceDeviceType: .builtInLiDARDepthCamera,
                                          sourceDevicePosition: lidar.position)
        guard !depthPorts.isEmpty else {
            throw RecorderError.unsupported(
                "The LiDAR input exposes no depth port in this configuration.")
        }
        let depthConnection = AVCaptureConnection(inputPorts: depthPorts, output: depthOutput)
        guard session.canAddConnection(depthConnection) else {
            throw RecorderError.unsupported("Cannot connect the depth port to the depth output.")
        }
        session.addConnection(depthConnection)

        let uwInput = try AVCaptureDeviceInput(device: ultraWide)
        let uwPorts = uwInput.ports(for: .video,
                                    sourceDeviceType: .builtInUltraWideCamera,
                                    sourceDevicePosition: ultraWide.position)
        guard session.canAddInput(uwInput), !uwPorts.isEmpty,
              session.canAddOutput(ultraWideOutput) else {
            throw RecorderError.unsupported("Cannot add the ultra-wide camera alongside depth.")
        }
        session.addInputWithNoConnections(uwInput)
        ultraWideOutput.videoSettings =
            [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        ultraWideOutput.alwaysDiscardsLateVideoFrames = true
        session.addOutputWithNoConnections(ultraWideOutput)
        ultraWideOutput.setSampleBufferDelegate(self, queue: queue)
        let uwConnection = AVCaptureConnection(inputPorts: uwPorts, output: ultraWideOutput)
        guard session.canAddConnection(uwConnection) else {
            throw RecorderError.unsupported("Cannot connect the ultra-wide port.")
        }
        session.addConnection(uwConnection)

        // Focus locked so the intrinsics in `calib/` stay the intrinsics of
        // these frames. An autofocus pull changes the focal length mid-session
        // and there is no per-frame calibration on a video output to record it.
        try? lockFocus(on: ultraWide)

        notes.append("hardware cost \(session.hardwareCost), "
                     + "system pressure cost \(session.systemPressureCost)")
        notes.append("ultra-wide fov \(ultraWide.activeFormat.videoFieldOfView)")
    }

    private func selectDepthFormat(on device: AVCaptureDevice) throws {
        // The largest depth map that can run in a multi-cam session, since the
        // depth is the scale reference and there is no reason to throw
        // resolution away.
        let candidates = device.formats.filter {
            $0.isMultiCamSupported && !$0.supportedDepthDataFormats.isEmpty
        }
        guard let format = candidates.max(by: { a, b in
            let da = a.supportedDepthDataFormats.map {
                CMVideoFormatDescriptionGetDimensions($0.formatDescription)
            }.map { Int($0.width) * Int($0.height) }.max() ?? 0
            let db = b.supportedDepthDataFormats.map {
                CMVideoFormatDescriptionGetDimensions($0.formatDescription)
            }.map { Int($0.width) * Int($0.height) }.max() ?? 0
            return da < db
        }) else {
            throw RecorderError.unsupported("No multi-cam format on the LiDAR camera carries depth.")
        }
        let depthFormats = format.supportedDepthDataFormats.filter {
            CMFormatDescriptionGetMediaSubType($0.formatDescription)
                == kCVPixelFormatType_DepthFloat32
        }
        guard let depthFormat = depthFormats.max(by: { a, b in
            let da = CMVideoFormatDescriptionGetDimensions(a.formatDescription)
            let db = CMVideoFormatDescriptionGetDimensions(b.formatDescription)
            return Int(da.width) * Int(da.height) < Int(db.width) * Int(db.height)
        }) else {
            throw RecorderError.unsupported("No float32 depth format available.")
        }
        try device.lockForConfiguration()
        device.activeFormat = format
        device.activeDepthDataFormat = depthFormat
        device.unlockForConfiguration()
        let d = CMVideoFormatDescriptionGetDimensions(depthFormat.formatDescription)
        notes.append("depth \(d.width)x\(d.height) float32, filtering off")
    }

    private func lockFocus(on device: AVCaptureDevice) throws {
        try device.lockForConfiguration()
        if device.isFocusModeSupported(.locked) { device.focusMode = .locked }
        if device.isExposureModeSupported(.continuousAutoExposure) {
            device.exposureMode = .continuousAutoExposure
        }
        device.unlockForConfiguration()
    }

    // MARK: - Finishing

    private func closeFiles() throws {
        frameIndex?.close()
        depthIndex?.close()
        depthData?.close()
        motionWriter?.close()
        guard let dir = directory else { return }
        for note in notes {
            try? eventWriter?.write(SessionEvent(t: Clock.now(), kind: "multicam", detail: note))
        }
        eventWriter?.close()

        // The manifest says what is *not* here. A reader that finds no
        // `pose.jsonl` should not have to guess whether the recorder crashed.
        let manifest: [String: Any] = [
            "id": dir.lastPathComponent,
            "kind": "multicam",
            "poses": "none — this session has no tracker; run eval/pi3_chain.py",
            "images": imageCount,
            "depth_frames": depthCount,
            "images_dropped_for_rate": droppedImages,
            "lens": "UltraWideCamera",
            "depth_from": "LiDARDepthCamera, filtering disabled, holes mean no return",
            "calibration": "calib/iphone17-1_ultrawide.json — scale intrinsics by "
                         + "the image size, and apply R x + t to reach the wide "
                         + "camera's frame, which is where the depth already is",
            "notes": notes,
        ]
        let data = try JSONSerialization.data(withJSONObject: manifest,
                                              options: [.prettyPrinted, .sortedKeys])
        try data.write(to: dir.appendingPathComponent(SessionStore.Filename.manifest),
                       options: .atomic)
        FileManager.default.createFile(
            atPath: dir.appendingPathComponent(SessionStore.Filename.complete).path,
            contents: nil)
    }

    private func finish(_ result: Result<URL, Error>) {
        running = false
        DispatchQueue.main.async { self.onFinished?(result) }
    }

    private func status(_ text: String) {
        DispatchQueue.main.async { self.onStatus?(text) }
    }
}

// MARK: - Depth

extension MultiCamRecorder: AVCaptureDepthDataOutputDelegate {

    func depthDataOutput(_ output: AVCaptureDepthDataOutput,
                         didOutput depthData: AVDepthData,
                         timestamp: CMTime,
                         connection: AVCaptureConnection) {
        guard running, !stopping, let writer = self.depthData else { return }
        let converted = depthData.depthDataType == kCVPixelFormatType_DepthFloat32
            ? depthData
            : depthData.converting(toDepthDataType: kCVPixelFormatType_DepthFloat32)
        let map = converted.depthDataMap
        CVPixelBufferLockBaseAddress(map, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(map, .readOnly) }
        let width = CVPixelBufferGetWidth(map)
        let height = CVPixelBufferGetHeight(map)
        guard let base = CVPixelBufferGetBaseAddress(map) else { return }
        let rowBytes = CVPixelBufferGetBytesPerRow(map)

        // Written as float16 to match `depth.bin` from the ARKit recorder, so
        // `tools/read_session.py` reads both without a branch.
        var payload = Data(capacity: width * height * 2)
        for row in 0..<height {
            let line = base.advanced(by: row * rowBytes)
                .assumingMemoryBound(to: Float32.self)
            for column in 0..<width {
                var half = Float16(line[column])
                withUnsafeBytes(of: &half) { payload.append(contentsOf: $0) }
            }
        }
        let offset = writer.nextOffset
        do {
            try writer.append(payload)
            try depthIndex?.write(DepthIndexEntry(
                t: timestamp.seconds, frame: depthCount, offset: offset,
                length: payload.count, width: width, height: height,
                format: "float16", confidenceOffset: nil, confidenceLength: nil,
                cond: nil, weakAxis: nil, conditioningSamples: 0))
            depthCount += 1
        } catch {
            status("depth write failed: \(error.localizedDescription)")
        }
    }
}

// MARK: - Ultra-wide images

extension MultiCamRecorder: AVCaptureVideoDataOutputSampleBufferDelegate {

    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard running, !stopping, let dir = directory else { return }
        let stamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds
        guard stamp - lastImageStamp >= 1.0 / stillsHz else {
            droppedImages += 1
            return
        }
        guard let buffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let image = CIImage(cvPixelBuffer: buffer)
        let context = CIContext()
        guard let jpeg = context.jpegRepresentation(
            of: image, colorSpace: CGColorSpaceCreateDeviceRGB(),
            options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 0.85])
        else { return }

        let name = String(format: "%06d.jpg", imageCount)
        let relative = "\(SessionStore.Filename.framesDirectory)/\(name)"
        let url = dir.appendingPathComponent(relative)
        do {
            try jpeg.write(to: url, options: .atomic)
            try frameIndex?.write(FrameIndexEntry(
                t: stamp, frame: imageCount, file: relative,
                width: CVPixelBufferGetWidth(buffer),
                height: CVPixelBufferGetHeight(buffer),
                bytes: jpeg.count))
            imageCount += 1
            lastImageStamp = stamp
        } catch {
            status("image write failed: \(error.localizedDescription)")
        }
    }
}
