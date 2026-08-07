import ARKit
import CoreVideo
import Foundation
import simd

/// Camera, LiDAR depth and 6-DoF pose capture, all sourced from ARKit.
///
/// ARKit is the *only* camera client here, deliberately. An `AVCaptureSession`
/// and an `ARSession` cannot both own the camera, and going through ARKit means
/// the RGB frame, its depth map, the camera pose and the intrinsics all come
/// out of a single `ARFrame` carrying a single timestamp — frame-perfect
/// synchronisation for free, instead of correlating two independent streams
/// after the fact.
///
/// Everything runs on `arQueue`, a dedicated serial queue that ARKit delivers
/// frames on. `ARFrame` objects are never retained beyond the callback: ARKit
/// draws them from a small pool and holding one starves capture.
final class ARRecorder: NSObject, ARSessionDelegate {

    private let session = ARSession()
    private let arQueue = DispatchQueue(label: "nav.ar", qos: .userInitiated)
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    private var videoWriter: VideoWriter?
    private var stillsWriter: StillsWriter?
    private var lastStillTime: Double = -.infinity
    private var config = CaptureConfig.default
    private var running = false
    private var frameIndex = 0
    private var lastDepthTime: Double = -.infinity
    private var lastPreviewTime: Double = -.infinity

    /// Called on `arQueue`.
    var onPose: ((PoseSample) -> Void)?
    /// `(depthData, confidenceData?, t, frameIndex, width, height)`.
    var onDepth: ((Data, Data?, Double, Int, Int, Int) -> Void)?
    var onPlane: ((PlaneSample) -> Void)?
    var onStill: ((StillsWriter.Entry) -> Void)?
    var onEvent: ((String, String) -> Void)?

    /// Last time each plane's `updated` row was written. ARKit re-reports a
    /// growing plane on almost every frame, which would bury the useful
    /// add/remove events under thousands of near-identical rows.
    private var lastPlaneUpdate: [UUID: Double] = [:]
    private static let planeUpdateInterval: Double = 1.0
    /// Low-rate preview image for the UI. Delivered on `arQueue`; the receiver
    /// hops to main.
    var onPreview: ((CGImage) -> Void)?

    /// Counters read from the main thread while the AR queue is writing them,
    /// so they live behind a lock rather than being plain `var`s.
    struct Snapshot {
        var poses = 0
        var encodedFrames = 0
        var droppedFrames = 0
    }

    private let stateLock = NSLock()
    private var _snapshot = Snapshot()
    private var _isThrottled = false

    func snapshot() -> Snapshot {
        stateLock.lock()
        defer { stateLock.unlock() }
        return _snapshot
    }

    /// Set by the coordinator when the device gets hot. Poses keep flowing —
    /// they are nearly free — but encoding and depth extraction stop.
    var isThrottled: Bool {
        get {
            stateLock.lock()
            defer { stateLock.unlock() }
            return _isThrottled
        }
        set {
            stateLock.lock()
            _isThrottled = newValue
            stateLock.unlock()
        }
    }

    static var isSupported: Bool {
        ARWorldTrackingConfiguration.isSupported
    }

    static var hasLiDAR: Bool {
        ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)
    }

    override init() {
        super.init()
        session.delegate = self
        session.delegateQueue = arQueue
    }

    /// Only valid once `stop(completion:)` has fired, i.e. after the writer has
    /// been configured by the first frame and finalised.
    var videoInfo: SessionManifest.VideoInfo? { videoWriter?.info }

    // MARK: - Lifecycle

    func start(config: CaptureConfig, sessionDirectory: URL) {
        arQueue.async { [weak self] in
            guard let self = self, !self.running else { return }
            self.config = config
            self.running = true
            self.frameIndex = 0
            self.lastDepthTime = -.infinity
            self.lastStillTime = -.infinity
            self.stateLock.lock()
            self._snapshot = Snapshot()
            self.stateLock.unlock()

            switch config.captureMode {
            case .video:
                self.videoWriter = VideoWriter(
                    url: sessionDirectory.appendingPathComponent(SessionStore.Filename.video),
                    bitrate: config.videoBitrate,
                    fps: config.videoFPS)
            case .stills:
                let writer = StillsWriter(
                    directory: sessionDirectory.appendingPathComponent(SessionStore.Filename.framesDirectory),
                    relativePrefix: SessionStore.Filename.framesDirectory,
                    quality: config.stillQuality)
                writer.onWrite = { [weak self] entry in self?.onStill?(entry) }
                writer.onError = { [weak self] message in self?.onEvent?("stills.error", message) }
                do {
                    try writer.prepare()
                    self.stillsWriter = writer
                } catch {
                    self.onEvent?("stills.error",
                                  "could not create frames directory: \(error.localizedDescription)")
                }
            }

            let arConfig = ARWorldTrackingConfiguration()
            // Gravity-aligned world frame with an arbitrary yaw origin. The
            // heading comes from GPS and the magnetometer; asking ARKit for
            // `.gravityAndHeading` costs a compass lock we don't need.
            arConfig.worldAlignment = .gravity
            arConfig.planeDetection = config.detectPlanes ? [.horizontal, .vertical] : []
            arConfig.environmentTexturing = .none
            // Pure power savings — nothing downstream consumes either of these.
            arConfig.isLightEstimationEnabled = false
            if #available(iOS 16.0, *) {
                // HDR retimes exposure frame to frame, so brightness stops
                // being comparable across a drive. Off, for consistency.
                arConfig.isVideoHDRAllowed = false
            }

            if config.recordDepth, ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
                // Raw scene depth, not `.smoothedSceneDepth`: the smoothed
                // variant is temporally filtered, which is exactly the kind of
                // hidden preprocessing that should not be baked into training
                // data.
                arConfig.frameSemantics.insert(.sceneDepth)
            } else if config.recordDepth {
                self.onEvent?("depth.unsupported", "sceneDepth not supported on this device")
            }

            if let format = Self.preferredVideoFormat(config: config) {
                arConfig.videoFormat = format
            }

            self.session.run(arConfig, options: [.resetTracking, .removeExistingAnchors])
            self.onEvent?("ar.started",
                          "format=\(Int(arConfig.videoFormat.imageResolution.width))x\(Int(arConfig.videoFormat.imageResolution.height))@\(arConfig.videoFormat.framesPerSecond) depth=\(arConfig.frameSemantics.contains(.sceneDepth))")
        }
    }

    /// Stops capture and finalises the video file. `completion` runs once the
    /// .mov is playable — an un-finalised AVAssetWriter output is unreadable,
    /// so the session must not be marked complete before this fires.
    func stop(completion: @escaping () -> Void) {
        arQueue.async { [weak self] in
            guard let self = self else { completion(); return }
            guard self.running else { completion(); return }
            self.running = false
            self.session.pause()

            // Drain any encodes still in flight before the session is allowed
            // to be marked complete.
            self.stillsWriter?.finish()

            guard let writer = self.videoWriter else {
                completion()
                return
            }
            writer.finish {
                completion()
            }
        }
    }

    // MARK: - ARSessionDelegate

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard running else { return }
        let t = frame.timestamp
        let index = frameIndex
        frameIndex += 1

        // Pose first and unconditionally: it is a few dozen bytes and remains
        // useful even when video and depth have been throttled off.
        let transform = frame.camera.transform
        let translation = transform.translation
        let q = transform.rotationQuaternion
        let intrinsics = frame.camera.intrinsics
        let gravity = transform.gravityInCameraFrame

        onPose?(PoseSample(
            t: t,
            frame: index,
            tx: translation.x, ty: translation.y, tz: translation.z,
            qx: q.imag.x, qy: q.imag.y, qz: q.imag.z, qw: q.real,
            fx: intrinsics[0][0], fy: intrinsics[1][1],
            cx: intrinsics[2][0], cy: intrinsics[2][1],
            tracking: Self.describe(frame.camera.trackingState),
            exposure: frame.camera.exposureDuration,
            gravX: gravity.x, gravY: gravity.y, gravZ: gravity.z))

        if !isThrottled {
            switch config.captureMode {
            case .video:
                videoWriter?.append(frame.capturedImage, pts: t)
            case .stills:
                let interval = 1.0 / max(0.1, config.stillsHz)
                if t - lastStillTime >= interval * 0.5 {
                    lastStillTime = t
                    stillsWriter?.capture(frame.capturedImage, t: t, frame: index)
                }
            }

            if config.recordDepth, let sceneDepth = frame.sceneDepth {
                let interval = 1.0 / max(0.1, config.depthHz)
                if t - lastDepthTime >= interval * 0.5 {
                    lastDepthTime = t
                    if let (data, width, height) = Self.float16Depth(sceneDepth.depthMap) {
                        let confidence = config.recordConfidence
                            ? sceneDepth.confidenceMap.flatMap { Self.confidenceBytes($0) }
                            : nil
                        onDepth?(data, confidence, t, index, width, height)
                    }
                }
            }
        }

        stateLock.lock()
        _snapshot.poses += 1
        _snapshot.encodedFrames = videoWriter?.frameCount ?? stillsWriter?.written ?? 0
        _snapshot.droppedFrames = videoWriter?.droppedCount ?? stillsWriter?.failed ?? 0
        stateLock.unlock()

        // Preview is throttled hard — it exists to aim the camera at the road,
        // not to be watched.
        if t - lastPreviewTime >= 0.2, onPreview != nil {
            lastPreviewTime = t
            if let cg = makePreview(frame.capturedImage) {
                onPreview?(cg)
            }
        }
    }

    func session(_ session: ARSession, cameraDidChangeTrackingState camera: ARCamera) {
        onEvent?("ar.tracking", Self.describe(camera.trackingState))
    }

    func session(_ session: ARSession, didAdd anchors: [ARAnchor]) {
        emitPlanes(anchors, event: "added", throttled: false)
    }

    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        emitPlanes(anchors, event: "updated", throttled: true)
    }

    func session(_ session: ARSession, didRemove anchors: [ARAnchor]) {
        for anchor in anchors {
            if let plane = anchor as? ARPlaneAnchor {
                lastPlaneUpdate.removeValue(forKey: plane.identifier)
            }
        }
        emitPlanes(anchors, event: "removed", throttled: false)
    }

    private func emitPlanes(_ anchors: [ARAnchor], event: String, throttled: Bool) {
        guard running, onPlane != nil else { return }
        let t = Clock.now()
        for anchor in anchors {
            guard let plane = anchor as? ARPlaneAnchor else { continue }
            if throttled {
                let last = lastPlaneUpdate[plane.identifier] ?? -.infinity
                guard t - last >= Self.planeUpdateInterval else { continue }
                lastPlaneUpdate[plane.identifier] = t
            }

            let translation = plane.transform.translation
            let q = plane.transform.rotationQuaternion
            onPlane?(PlaneSample(
                t: t,
                id: plane.identifier.uuidString,
                event: event,
                alignment: plane.alignment == .horizontal ? "horizontal" : "vertical",
                classification: Self.describe(plane.classification),
                tx: translation.x, ty: translation.y, tz: translation.z,
                qx: q.imag.x, qy: q.imag.y, qz: q.imag.z, qw: q.real,
                cx: plane.center.x, cy: plane.center.y, cz: plane.center.z,
                width: plane.planeExtent.width,
                height: plane.planeExtent.height,
                rotationOnYAxis: plane.planeExtent.rotationOnYAxis))
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        onEvent?("ar.error", error.localizedDescription)
    }

    func sessionWasInterrupted(_ session: ARSession) {
        // Fires when the app is backgrounded or a call comes in. Video and
        // depth are dead until it resumes; GPS and IMU carry on.
        onEvent?("ar.interrupted", "camera capture suspended")
    }

    func sessionInterruptionEnded(_ session: ARSession) {
        // ARKit resets the world origin here, so poses before and after this
        // event are in *different* coordinate frames. Readers must treat this
        // as a hard break in the odometry track.
        onEvent?("ar.interruptionEnded", "world origin reset; pose frame discontinuity")
    }

    // MARK: - Pixel buffer conversion

    /// Converts ARKit's Float32 depth map to row-major float16.
    ///
    /// Halves the payload for a quantisation error of roughly a millimetre at
    /// 10 m — far below the sensor's own noise floor, and the difference
    /// between ~5 GB and ~10 GB across a long drive.
    static func float16Depth(_ buffer: CVPixelBuffer) -> (Data, Int, Int)? {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }

        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)

        var out = [Float16](repeating: 0, count: width * height)
        out.withUnsafeMutableBufferPointer { dst in
            for y in 0..<height {
                let row = base.advanced(by: y * bytesPerRow).assumingMemoryBound(to: Float32.self)
                let rowStart = y * width
                for x in 0..<width {
                    dst[rowStart + x] = Float16(row[x])
                }
            }
        }
        let data = out.withUnsafeBufferPointer { Data(buffer: $0) }
        return (data, width, height)
    }

    /// One byte per pixel, `ARConfidenceLevel` raw values, rows packed tight.
    static func confidenceBytes(_ buffer: CVPixelBuffer) -> Data? {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }

        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)

        var out = Data(capacity: width * height)
        for y in 0..<height {
            let row = base.advanced(by: y * bytesPerRow).assumingMemoryBound(to: UInt8.self)
            out.append(UnsafeBufferPointer(start: row, count: width))
        }
        return out
    }

    private func makePreview(_ buffer: CVPixelBuffer) -> CGImage? {
        let image = CIImage(cvPixelBuffer: buffer)
            .transformed(by: CGAffineTransform(scaleX: 0.2, y: 0.2))
        return ciContext.createCGImage(image, from: image.extent)
    }

    // MARK: - Helpers

    /// Picks a capture format, preferring vertical field of view over pixels.
    ///
    /// The camera's horizontal FOV is fixed by the lens — about 62 degrees — and
    /// no format changes it. What formats *do* change is the aspect ratio, and
    /// a 16:9 format (including the 4K one) gets there by cropping the top and
    /// bottom off the 4:3 sensor readout. That trades away roughly 11 degrees of
    /// vertical FOV for pixels nothing downstream needs, and for navigation data
    /// seeing more floor and ceiling is worth far more than resolution.
    ///
    /// So: 4:3 first, then the largest of those.
    private static func preferredVideoFormat(config: CaptureConfig) -> ARConfiguration.VideoFormat? {
        let formats = ARWorldTrackingConfiguration.supportedVideoFormats
        guard !formats.isEmpty else { return nil }

        // In stills mode the sensor frame rate is irrelevant — frames are
        // sampled down to `stillsHz` anyway — so only video mode filters on it.
        let rateFiltered: [ARConfiguration.VideoFormat]
        if config.captureMode == .video {
            let usable = formats.filter { $0.framesPerSecond >= config.videoFPS }
            rateFiltered = usable.isEmpty ? formats : usable
        } else {
            rateFiltered = formats
        }

        func isFourThree(_ format: ARConfiguration.VideoFormat) -> Bool {
            let size = format.imageResolution
            guard size.height > 0 else { return false }
            return abs(size.width / size.height - 4.0 / 3.0) < 0.02
        }

        let fourThree = rateFiltered.filter(isFourThree)
        let pool = fourThree.isEmpty ? rateFiltered : fourThree
        return pool.max { a, b in
            let areaA = a.imageResolution.width * a.imageResolution.height
            let areaB = b.imageResolution.width * b.imageResolution.height
            return areaA < areaB
        }
    }

    static func describe(_ classification: ARPlaneAnchor.Classification) -> String {
        switch classification {
        case .wall: return "wall"
        case .floor: return "floor"
        case .ceiling: return "ceiling"
        case .table: return "table"
        case .seat: return "seat"
        case .door: return "door"
        case .window: return "window"
        case .none(let reason):
            switch reason {
            case .notAvailable: return "none:notAvailable"
            case .undetermined: return "none:undetermined"
            case .unknown: return "none:unknown"
            @unknown default: return "none:unhandled"
            }
        @unknown default:
            return "unknown"
        }
    }

    static func describe(_ state: ARCamera.TrackingState) -> String {
        switch state {
        case .notAvailable:
            return "notAvailable"
        case .normal:
            return "normal"
        case .limited(let reason):
            switch reason {
            case .initializing: return "limited:initializing"
            case .relocalizing: return "limited:relocalizing"
            case .excessiveMotion: return "limited:excessiveMotion"
            case .insufficientFeatures: return "limited:insufficientFeatures"
            @unknown default: return "limited:unknown"
            }
        }
    }
}
