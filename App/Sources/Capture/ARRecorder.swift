import ARKit
import AVFoundation
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
    private var config = CaptureConfig.default
    private var running = false
    private var frameIndex = 0
    private var captureGate = RateGate(hz: 5)
    private var depthGate = RateGate(hz: 5)
    private var lastPreviewTime: Double = -.infinity

    /// Called on `arQueue`.
    var onPose: ((PoseSample) -> Void)?
    /// `(depthData, confidenceData?, t, frameIndex, width, height,
    ///   frameCondition, weakAxis, conditioningSamples)`.
    var onDepth: ((Data, Data?, Double, Int, Int, Int, Double?, [Double]?, Int) -> Void)?
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
        /// Fraction of the last depth map ARKit rated medium or high.
        ///
        /// Surfaced live because it is not recoverable later by looking at the
        /// screen: ARKit's depth is guided by the colour image, so a dim room
        /// returns a full, plausible-looking depth map that is almost entirely
        /// `ARConfidenceLevel.low`. A 30 m session measured 98.4% low and was
        /// worthless for registration, and nothing about holding the phone said
        /// so at the time.
        var depthUsable: Double = 0
        /// Latest in-image roll derived from gravity, in degrees.
        var currentRoll: Double?
        /// Frame-only conditioning for the newest depth map, the direction it
        /// leaves unconstrained, and how much of the recent past sat below the
        /// floor. Surfaced for the same reason as `depthUsable`: it is the one
        /// signal measured to track how a session actually turns out — median
        /// conditioning against the depth estimator's loop error runs
        /// r = -0.68 over thirteen sessions, where per-pair error, ICP
        /// conditioning, image sharpness and reference distance each failed.
        /// It is a property of the frame's own geometry, so unlike a residual
        /// or an inlier count nothing the estimator does can flatter it.
        var frameConditioning: Double?
        var conditioningWeakAxis: [Double]?
        var conditioningBelowFloor: Double = 0
        /// Recent camera positions in ARKit's gravity-aligned world, thinned
        /// for drawing. A walk that has not returned to where it started is
        /// visible here and nowhere else on the screen.
        var track: [SIMD3<Float>] = []
    }

    /// Below this, the depth estimator's loop error starts to run away. Soft:
    /// the good and bad sessions overlap around it, so it ranks a capture
    /// rather than sorting it, which is all a warning needs to do.
    static let conditioningFloor = 0.007

    private struct FrameConditioning {
        let cond: Double?
        let weakAxis: [Double]?
        let samples: Int
    }

    private let stateLock = NSLock()
    private var _snapshot = Snapshot()
    private var _isThrottled = false
    /// About five seconds of depth frames. Long enough that a single awkward
    /// frame does not raise a warning, short enough that walking out of a bad
    /// spot clears it while the phone is still up.
    private var _recentConditioning: [Double] = []
    private var _track: [SIMD3<Float>] = []
    private var _trackTick = 0

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
            self.captureGate = RateGate(hz: config.captureMode == .stills
                                        ? config.stillsHz
                                        : Double(config.videoFPS))
            self.depthGate = RateGate(hz: config.depthHz)
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
                arConfig.videoHDRAllowed = false
            }
            // Autofocus failures are finite, self-report through fx, and are
            // removed by the export gate. Fixed focus instead fails silently
            // for the rest of a session at the wrong distance.
            arConfig.isAutoFocusEnabled = true

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
                          "format=\(Self.describe(arConfig.videoFormat)) "
                          + "depth=\(arConfig.frameSemantics.contains(.sceneDepth)) "
                          + "autofocus=\(arConfig.isAutoFocusEnabled)")
            // Logged once per session so a recording is self-describing about
            // what the hardware actually offered — in particular whether any
            // ultra-wide format is available to world tracking, which is not
            // something a spec sheet answers.
            self.onEvent?("ar.formats",
                          ARWorldTrackingConfiguration.supportedVideoFormats
                              .map(Self.describe).joined(separator: " | "))
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
        let roll = atan2(Double(gravity.x), -Double(gravity.y))
            * 180.0 / Double.pi
        // Thinned to about 5 Hz and capped: this is drawn, not recorded, and
        // the file already holds every pose at full rate.
        _trackTick += 1
        stateLock.lock()
        _snapshot.currentRoll = roll
        if _trackTick % 6 == 0 {
            _track.append(SIMD3(translation.x, translation.y, translation.z))
            if _track.count > 900 { _track.removeFirst(_track.count - 900) }
            _snapshot.track = _track
        }
        stateLock.unlock()

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
                // Video keeps its own cadence, and depth is sampled separately:
                // there is no frame-level pairing to preserve because the video
                // track is addressed by presentation time, not by index.
                videoWriter?.append(frame.capturedImage, pts: t)
                if config.recordDepth, depthGate.shouldFire(at: t) {
                    emitDepth(from: frame, t: t, index: index)
                }

            case .stills:
                // Depth runs faster than the images and every captured image is
                // guaranteed a depth map on its own `frame`.
                //
                // Independent gates do not work and fail quietly: `sceneDepth`
                // is nil on some frames, so a depth-only gate does not advance
                // in lockstep and the two streams drift onto different frames
                // within seconds, leaving the `frame` join matching almost
                // nothing. The fix is a superset rather than a second gate —
                // depth fires on its own schedule *or* whenever an image does.
                //
                // The rates are decoupled because the export rate is not the
                // capture rate. Episodes want 5 Hz; frame-to-frame depth
                // registration wants every frame it can get, since ICP
                // converges on small motion and 5 Hz at walking pace is ~10 cm
                // and several degrees apart — far outside where it works.
                // Decimating later is free; the frames not captured are gone.
                let wantImage = captureGate.shouldFire(at: t)
                let wantDepth = depthGate.shouldFire(at: t)
                if wantImage {
                    stillsWriter?.capture(frame.capturedImage, t: t, frame: index)
                }
                if config.recordDepth, wantImage || wantDepth {
                    emitDepth(from: frame, t: t, index: index)
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

    /// Extracts and forwards the depth map for a frame, if it has one.
    /// Silently does nothing when ARKit did not attach depth to this frame,
    /// which happens routinely — the image is still worth keeping.
    private func emitDepth(from frame: ARFrame, t: Double, index: Int) {
        guard let sceneDepth = frame.sceneDepth else { return }
        guard let (data, width, height, depthValues) = Self.float16Depth(sceneDepth.depthMap) else {
            return
        }
        // Compute from the same float16 values that are written to depth.bin,
        // so the offline reader and the live measurement see identical input.
        let confidenceBytes = sceneDepth.confidenceMap.flatMap { Self.confidenceBytes($0) }
        let confidence = config.recordConfidence ? confidenceBytes : nil
        let conditioning = Self.frameConditioning(
            depth: depthValues,
            confidence: confidence,
            width: width,
            height: height,
            intrinsics: frame.camera.intrinsics)
        if let confidence = confidence, !confidence.isEmpty {
            // Sampled rather than counted in full: this runs on the AR queue at
            // 30 Hz, and every 16th byte settles a percentage well enough.
            var usable = 0, seen = 0
            var i = 0
            while i < confidence.count {
                if confidence[i] >= 1 { usable += 1 }
                seen += 1
                i += 16
            }
            let fraction = seen > 0 ? Double(usable) / Double(seen) : 0
            stateLock.lock()
            _snapshot.depthUsable = fraction
            stateLock.unlock()
        }
        stateLock.lock()
        _snapshot.frameConditioning = conditioning.cond
        _snapshot.conditioningWeakAxis = conditioning.weakAxis
        if let value = conditioning.cond {
            _recentConditioning.append(value)
            if _recentConditioning.count > 150 {
                _recentConditioning.removeFirst(_recentConditioning.count - 150)
            }
            let below = _recentConditioning.filter { $0 < Self.conditioningFloor }.count
            _snapshot.conditioningBelowFloor =
                Double(below) / Double(_recentConditioning.count)
        }
        stateLock.unlock()
        onDepth?(data, confidence, t, index, width, height,
                 conditioning.cond, conditioning.weakAxis, conditioning.samples)
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
    static func float16Depth(_ buffer: CVPixelBuffer) -> (Data, Int, Int, [Float16])? {
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
        return (data, width, height, out)
    }

    /// Computes the conditioning of one depth frame without frame matching.
    ///
    /// This intentionally mirrors `frame_points` and `frame_conditioning` in
    /// tools/depth_odometry.py: 5x5 edge-aware smoothing, normals from pixels
    /// two apart, confidence < 1 discarded, then every fourth pixel contributes
    /// one row `[p x n, n]`. `weakAxis` is in the depth frame (+Z forward,
    /// +Y down), not ARKit's camera frame (-Z forward, +Y up).
    private static func frameConditioning(
        depth: [Float16],
        confidence: Data?,
        width: Int,
        height: Int,
        intrinsics: simd_float3x3
    ) -> FrameConditioning {
        let count = width * height
        guard depth.count == count, width > 0, height > 0 else {
            return FrameConditioning(cond: nil, weakAxis: nil, samples: 0)
        }

        let fx = Double(intrinsics[0][0])
        let fy = Double(intrinsics[1][1])
        let cx = Double(intrinsics[2][0])
        let cy = Double(intrinsics[2][1])
        guard fx > 0, fy > 0 else {
            return FrameConditioning(cond: nil, weakAxis: nil, samples: 0)
        }
        let scale = Double(width) / max(2.0 * cx, 1e-12)
        let depthFx = fx * scale
        let depthFy = fy * scale
        let depthCx = cx * scale
        let depthCy = cy * scale

        var base = [Double](repeating: 0, count: count)
        var valid = [Bool](repeating: false, count: count)
        let hasConfidence = confidence?.count == count
        for i in 0..<count {
            let value = Double(depth[i])
            // `frame_points` uses nan_to_num before backprojection.  Preserve
            // finite out-of-range values for neighbouring normals, but turn
            // NaN/no-return and confidence-rejected pixels into zero.
            let confidenceOK = !hasConfidence || confidence![i] >= 1
            base[i] = value.isFinite && confidenceOK ? value : 0
            valid[i] = value.isFinite && value > 0.1 && value < 5.0 && confidenceOK
        }

        // Integral images reproduce _box_sum's clipped 5x5 neighbourhoods.
        let integralWidth = width + 1
        var depthIntegral = [Double](repeating: 0, count: (height + 1) * integralWidth)
        var validIntegral = [Double](repeating: 0, count: (height + 1) * integralWidth)
        for y in 0..<height {
            var depthRow = 0.0
            var validRow = 0.0
            for x in 0..<width {
                let i = y * width + x
                if valid[i] {
                    depthRow += base[i]
                    validRow += 1
                }
                let out = (y + 1) * integralWidth + x + 1
                let above = y * integralWidth + x + 1
                depthIntegral[out] = depthIntegral[above] + depthRow
                validIntegral[out] = validIntegral[above] + validRow
            }
        }

        var smoothed = [Double](repeating: 0, count: count)
        for y in 0..<height {
            let y0 = max(0, y - 2)
            let y1 = min(height, y + 3)
            for x in 0..<width {
                let x0 = max(0, x - 2)
                let x1 = min(width, x + 3)
                let a = y0 * integralWidth + x0
                let b = y0 * integralWidth + x1
                let c = y1 * integralWidth + x0
                let d = y1 * integralWidth + x1
                let total = depthIntegral[d] - depthIntegral[b]
                    - depthIntegral[c] + depthIntegral[a]
                let neighbours = validIntegral[d] - validIntegral[b]
                    - validIntegral[c] + validIntegral[a]
                let i = y * width + x
                let average = neighbours > 0 ? total / neighbours : base[i]
                smoothed[i] = abs(average - base[i]) > 0.05 ? base[i] : average
            }
        }

        let sampleStride = 4
        var hessian = [Double](repeating: 0, count: 36)
        var samples = 0
        for y in Swift.stride(from: 0, to: height, by: sampleStride) {
            for x in Swift.stride(from: 0, to: width, by: sampleStride) {
                let i = y * width + x
                guard valid[i], x > 0, x < width - 1,
                      y > 0, y < height - 1 else { continue }

                let p = Self.depthPoint(x: x, y: y, z: smoothed[i],
                                        fx: depthFx, fy: depthFy,
                                        cx: depthCx, cy: depthCy)
                let left = Self.depthPoint(x: x - 1, y: y,
                                           z: smoothed[y * width + x - 1],
                                           fx: depthFx, fy: depthFy,
                                           cx: depthCx, cy: depthCy)
                let right = Self.depthPoint(x: x + 1, y: y,
                                            z: smoothed[y * width + x + 1],
                                            fx: depthFx, fy: depthFy,
                                            cx: depthCx, cy: depthCy)
                let up = Self.depthPoint(x: x, y: y - 1,
                                         z: smoothed[(y - 1) * width + x],
                                         fx: depthFx, fy: depthFy,
                                         cx: depthCx, cy: depthCy)
                let down = Self.depthPoint(x: x, y: y + 1,
                                           z: smoothed[(y + 1) * width + x],
                                           fx: depthFx, fy: depthFy,
                                           cx: depthCx, cy: depthCy)
                let normalRaw = simd_cross(right - left, down - up)
                let normalLength = simd_length(normalRaw)
                guard normalLength > 1e-9 else { continue }
                let normal = normalRaw / normalLength
                let a = [
                    p.y * normal.z - p.z * normal.y,
                    p.z * normal.x - p.x * normal.z,
                    p.x * normal.y - p.y * normal.x,
                    normal.x, normal.y, normal.z
                ]
                for row in 0..<6 {
                    for column in 0..<6 {
                        hessian[row * 6 + column] += a[row] * a[column]
                    }
                }
                samples += 1
            }
        }

        guard samples > 0 else {
            return FrameConditioning(cond: nil, weakAxis: nil, samples: 0)
        }
        for i in 0..<36 {
            hessian[i] /= Double(samples)
        }
        let (eigenvalues, eigenvectors) = Self.symmetricEigen6(hessian)
        let largest = eigenvalues[5]
        guard largest > 1e-12, largest.isFinite else {
            return FrameConditioning(cond: 0.0, weakAxis: nil, samples: samples)
        }
        var weak = (0..<6).map { eigenvectors[$0 * 6] }
        let length = sqrt(weak.reduce(0.0) { $0 + $1 * $1 })
        if length > 1e-12 {
            weak = weak.map { $0 / length }
            var pivot = 0
            for i in 1..<weak.count where abs(weak[i]) > abs(weak[pivot]) {
                pivot = i
            }
            if weak[pivot] < 0 {
                weak = weak.map { -$0 }
            }
        }
        return FrameConditioning(
            cond: max(0.0, eigenvalues[0] / largest),
            weakAxis: weak,
            samples: samples)
    }

    private static func depthPoint(x: Int, y: Int, z: Double,
                                   fx: Double, fy: Double,
                                   cx: Double, cy: Double) -> SIMD3<Double> {
        SIMD3<Double>((Double(x) - cx) * z / fx,
                      (Double(y) - cy) * z / fy,
                      z)
    }

    /// Jacobi eigendecomposition for a 6x6 real symmetric matrix.
    /// The matrix is tiny and this avoids making the 30 Hz capture path depend
    /// on a platform-specific LAPACK symbol. Eigenvectors are columns in the
    /// returned row-major matrix, ordered by ascending eigenvalue.
    private static func symmetricEigen6(_ input: [Double]) -> ([Double], [Double]) {
        var a = input
        var v = [Double](repeating: 0, count: 36)
        for i in 0..<6 { v[i * 6 + i] = 1 }

        for _ in 0..<128 {
            var p = 0
            var q = 1
            var largest = 0.0
            for row in 0..<6 {
                for column in (row + 1)..<6 {
                    let value = abs(a[row * 6 + column])
                    if value > largest {
                        largest = value
                        p = row
                        q = column
                    }
                }
            }
            if largest <= 1e-12 { break }

            let app = a[p * 6 + p]
            let aqq = a[q * 6 + q]
            let apq = a[p * 6 + q]
            let angle = 0.5 * atan2(2.0 * apq, aqq - app)
            let c = cos(angle)
            let s = sin(angle)
            for k in 0..<6 where k != p && k != q {
                let akp = a[k * 6 + p]
                let akq = a[k * 6 + q]
                a[k * 6 + p] = c * akp - s * akq
                a[p * 6 + k] = a[k * 6 + p]
                a[k * 6 + q] = s * akp + c * akq
                a[q * 6 + k] = a[k * 6 + q]
            }
            a[p * 6 + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq
            a[q * 6 + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq
            a[p * 6 + q] = 0
            a[q * 6 + p] = 0

            for k in 0..<6 {
                let vkp = v[k * 6 + p]
                let vkq = v[k * 6 + q]
                v[k * 6 + p] = c * vkp - s * vkq
                v[k * 6 + q] = s * vkp + c * vkq
            }
        }

        let order = (0..<6).sorted { a[$0 * 6 + $0] < a[$1 * 6 + $1] }
        var values = [Double](repeating: 0, count: 6)
        var vectors = [Double](repeating: 0, count: 36)
        for column in 0..<6 {
            let source = order[column]
            values[column] = a[source * 6 + source]
            for row in 0..<6 {
                vectors[row * 6 + column] = v[row * 6 + source]
            }
        }
        return (values, vectors)
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

    /// Picks a capture format according to `CaptureConfig.formatPreference`.
    ///
    /// The default prefers 4:3 over raw pixel count. The horizontal field of
    /// view is fixed by the lens, so the 16:9 formats buy horizontal detail by
    /// giving up sensor height — and for indoor navigation, seeing more floor
    /// and ceiling is generally worth more than resolution a 224-pixel encoder
    /// throws away anyway.
    ///
    /// That is a judgement, not a measurement, and the app records per-frame
    /// intrinsics precisely so it can be checked: switch the preference, record,
    /// and compare the field of view the reader reports.
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

        let pool: [ARConfiguration.VideoFormat]
        switch config.formatPreference {
        case .tallest:
            let fourThree = rateFiltered.filter(isFourThree)
            pool = fourThree.isEmpty ? rateFiltered : fourThree
        case .highestResolution:
            pool = rateFiltered
        }
        return pool.max { a, b in
            let areaA = a.imageResolution.width * a.imageResolution.height
            let areaB = b.imageResolution.width * b.imageResolution.height
            return areaA < areaB
        }
    }

    static func describe(_ format: ARConfiguration.VideoFormat) -> String {
        let size = format.imageResolution
        // The capture device matters as much as the resolution: FOV is a
        // property of the lens, and every world-tracking format is expected to
        // report the wide-angle camera.
        let device = format.captureDeviceType.rawValue
            .replacingOccurrences(of: "AVCaptureDeviceTypeBuiltIn", with: "")
        return "\(Int(size.width))x\(Int(size.height))@\(format.framesPerSecond)/\(device)"
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
