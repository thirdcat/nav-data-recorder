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
///
/// **Both lenses, one walk.** The LiDAR device is the wide camera plus a
/// scanner and emits wide video beside the depth, so it is recorded too, into
/// `frames_wide/`. Every ultra-wide comparison so far scored three ultra-wide
/// walks against three *different* wide walks — path and lens moved together,
/// and `docs/POSE.md` says so in its limits. One walk through both lenses holds
/// the path fixed, and the depth is already in the wide camera's frame, so that
/// arm needs no reprojection at all.
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

    /// A small ultra-wide frame for the aiming preview, about 5 Hz, delivered
    /// on the **main** queue.
    ///
    /// `ARRecorder.onPreview` hands its image over on the AR queue and leaves
    /// the hop to the receiver; this class hops for the caller, because every
    /// other callback here already does (`status`, `finish`) and a screen that
    /// had to know which of two conventions applied to which closure is how a
    /// `@State` write ends up off the main thread.
    ///
    /// **What it costs, and why it is on the capture queue.** See
    /// `captureOutput`: one downscale render and a ~0.3 MB readback, only on
    /// frames that are already being encoded to JPEG. It runs inline on `queue`
    /// rather than being handed to a preview queue because moving it would mean
    /// retaining the 4K `CVPixelBuffer` past the delegate callback, and that
    /// starves the capture pool — the same reason `ARRecorder` never keeps an
    /// `ARFrame`.
    var onPreview: ((CGImage) -> Void)?

    /// The two lenses' active geometry, once, on the main queue, as soon as the
    /// session is configured and before any frame arrives.
    var onLensGeometry: ((LensGeometry) -> Void)?

    /// What the aiming overlay needs to place the wide camera's footprint
    /// inside the ultra-wide frame.
    ///
    /// Reported rather than assumed, because neither format is a constant this
    /// recorder chose once and for all. `selectUltraWideFormat` pins the widest
    /// field of view *available on the device*, and `selectDepthFormat` ranks
    /// the wide arm on depth resolution alone and says in its own comment that
    /// the tie-break is likely to change. A screen that drew the footprint from
    /// numbers compiled into it would keep drawing the old box after either
    /// moved, and nothing would say so.
    ///
    /// Fields of view are `AVCaptureDeviceFormat.videoFieldOfView`: horizontal,
    /// in degrees, and a **whole-frame** angle that already contains the lens
    /// distortion (`docs/UW_TRANSFER_PREREG.md`). They are field angles at the
    /// frame's horizontal edge and belong inside a tangent, never under a
    /// half-width.
    struct LensGeometry {
        let ultraWideWidth: Double
        let ultraWideHeight: Double
        let ultraWideFieldOfView: Double
        /// The LiDAR device's video format — the wide lens, and the frame the
        /// depth map is already in.
        let wideWidth: Double
        let wideHeight: Double
        let wideFieldOfView: Double
    }

    private let session = AVCaptureMultiCamSession()
    private let depthOutput = AVCaptureDepthDataOutput()
    private let ultraWideOutput = AVCaptureVideoDataOutput()
    /// The LiDAR device is the wide camera plus a scanner, so it emits wide
    /// video alongside the depth. Recording it costs one more output and no
    /// extra camera, and it removes the confound every comparison so far has
    /// carried: three ultra-wide walks were scored against three *different*
    /// wide walks, so path and lens moved together. One walk through both
    /// lenses holds the path fixed.
    private let wideOutput = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "nav.multicam.recorder")
    private lazy var motion = MotionRecorder(queue: queue)
    /// Built once, not per frame.
    ///
    /// It used to be constructed inside `captureOutput`, which at 5 Hz across
    /// two lenses was ten throwaway contexts a second and got away with it. A
    /// `CIContext` carries a Metal device, a command queue and a shader cache;
    /// rebuilding it per frame throws all three away and is the first thing to
    /// break when the rate goes up. Matching `ARRecorder`'s own context.
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    private var directory: URL?
    private var frameIndex: JSONLWriter?
    private var wideFrameIndex: JSONLWriter?
    private var depthIndex: JSONLWriter?
    private var depthData: BufferedFileWriter?
    private var motionWriter: JSONLWriter?
    private var eventWriter: JSONLWriter?

    private var ultraWide: AVCaptureDevice?
    private var lidar: AVCaptureDevice?
    /// `isGeometricDistortionCorrectionEnabled` as each device reported it
    /// after configuration, keyed by short device-type name. Written on the
    /// capture queue during `configure()`, read once at `closeFiles()`.
    private var distortionCorrection: [String: Bool] = [:]
    private var running = false
    private var stopping = false
    private var imageCount = 0
    private var wideCount = 0
    private var depthCount = 0
    private var droppedImages = 0
    /// Frames AVFoundation discarded because this delegate was still busy with
    /// the previous one, per lens.
    ///
    /// Not the same number as `droppedImages`, which counts the frames the rate
    /// gate throws away on purpose and is enormous by design — at a 30 fps
    /// sensor and a 5 Hz gate it is 25 a second, so it can say nothing about
    /// load. This is the one that can: `alwaysDiscardsLateVideoFrames` is true
    /// on both outputs, so a queue that falls behind loses frames **silently**,
    /// and until now nothing counted them at all.
    ///
    /// It exists because of the preview. The preview is the first work on this
    /// queue that is there for the screen rather than for the session, so it is
    /// the first thing that has to prove it costs nothing, and "the last
    /// capture dropped no frames" was not a number anyone could read off a
    /// session. See `docs/SCAN_PRESET.md` for the pre-registered reading.
    private var lateDrops = 0
    private var lateWideDrops = 0
    private var notes: [String] = []
    /// Filled from the first depth frame that carries calibration. See
    /// `captureDepthCalibration(from:)` for why it is worth keeping.
    private var depthCalibration: [String: Any]?

    /// Which capture preset this walk is being recorded under. Decides the
    /// stills rate, whether exposure gets locked, and the duration cap — and is
    /// written into `manifest.json` by name so the session says which it was.
    private let preset: CaptureConfig.Preset

    /// Images are stored at this rate. The sensor runs faster; everything
    /// between is dropped rather than written.
    ///
    /// Taken from the preset rather than matched to the ARKit recorder's, which
    /// is what it used to be. The two paths do not have the same budget: these
    /// frames are 3840x2160 against 1920x1440, and this session logs a
    /// `systemPressureCost` near 2.0 before a single JPEG is encoded.
    private let stillsHz: Double
    private var lastImageStamp: Double = -.infinity
    private var lastWideStamp: Double = -.infinity
    /// Presentation timestamp of the last frame turned into a preview. Touched
    /// on `queue` only, like the two above.
    private var lastPreviewStamp: Double = -.infinity
    /// About 5 Hz, matching `ARRecorder`'s own preview throttle. The preview
    /// exists to aim the lens, not to be watched.
    private static let previewInterval: Double = 0.2
    /// Target width of the preview image, in pixels. 384 is what `ARRecorder`
    /// produces (0.2 of 1920) and is already more than a 400-point-wide view on
    /// a Retina screen can show. Expressed as a width rather than as a scale
    /// factor so it does not depend on which format the ultra-wide pinned.
    private static let previewWidth: Double = 384
    /// Set when the preset's cap ended the session rather than the operator.
    private var terminationReason = "user"

    /// Where the second lens lands. Not `SessionStore.Filename`, because every
    /// other tool expects `frames/` to be the session's images and this is an
    /// addition only this recorder makes.
    static let wideFramesDirectory = "frames_wide"
    static let wideFrameIndexName = "frames_wide.jsonl"

    init(preset: CaptureConfig.Preset = .vln) {
        self.preset = preset
        self.stillsHz = preset.multiCamStillsHz
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
                self.scheduleExposureLock()
                self.scheduleDurationCap()
                self.status("recording — \(self.preset.rawValue) preset, "
                            + "\(Int(self.stillsHz)) Hz per lens, no poses")
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
        try FileManager.default.createDirectory(
            at: dir.appendingPathComponent(Self.wideFramesDirectory, isDirectory: true),
            withIntermediateDirectories: true)
        frameIndex = try JSONLWriter(
            url: dir.appendingPathComponent(SessionStore.Filename.frameIndex))
        wideFrameIndex = try JSONLWriter(
            url: dir.appendingPathComponent(Self.wideFrameIndexName))
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

        // The wide lens, from the LiDAR device's own video port. Optional: if it
        // cannot be added the ultra-wide recording is still complete, and the
        // note says which happened rather than leaving a reader to infer it
        // from a missing directory.
        let widePorts = lidarInput.ports(for: .video,
                                         sourceDeviceType: .builtInLiDARDepthCamera,
                                         sourceDevicePosition: lidar.position)
        if !widePorts.isEmpty, session.canAddOutput(wideOutput) {
            wideOutput.videoSettings =
                [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            wideOutput.alwaysDiscardsLateVideoFrames = true
            session.addOutputWithNoConnections(wideOutput)
            wideOutput.setSampleBufferDelegate(self, queue: queue)
            let wideConnection = AVCaptureConnection(inputPorts: widePorts,
                                                     output: wideOutput)
            if session.canAddConnection(wideConnection) {
                session.addConnection(wideConnection)
                notes.append("wide lens recorded alongside, from the LiDAR device")
            } else {
                notes.append("! the wide video connection was refused — "
                             + "ultra-wide only")
            }
        } else {
            notes.append("! the LiDAR input exposes no video port — ultra-wide only")
        }

        // Pin the ultra-wide's format. Left to itself AVFoundation picked a
        // different one per run — 1920x1080 at 106.2° in one session and
        // 640x480 at 101.0° in the next — which puts resolution and field of
        // view into every comparison between sessions as a confound nobody
        // chose. `manifest.json` records what was actually used either way, so
        // the first two captures are still readable; this stops it recurring.
        try? selectUltraWideFormat(on: ultraWide)

        // Focus locked so the intrinsics in `calib/` stay the intrinsics of
        // these frames. An autofocus pull changes the focal length mid-session
        // and there is no per-frame calibration on a video output to record it.
        try? lockFocus(on: ultraWide)

        // Distortion correction, for the same reason the format is pinned
        // above: a setting nobody chose is a setting that can change without
        // anyone noticing. Pinned on the ultra-wide, where the current value is
        // known; recorded only on the wide, where it is not. See
        // `noteDistortionCorrection`.
        noteDistortionCorrection(on: ultraWide, pinTo: true)
        noteDistortionCorrection(on: lidar, pinTo: nil)

        notes.append("hardware cost \(session.hardwareCost), "
                     + "system pressure cost \(session.systemPressureCost)")
        notes.append("ultra-wide fov \(ultraWide.activeFormat.videoFieldOfView)")

        // Both lenses' active geometry, to the screen, so the aiming overlay
        // draws *this* session's footprint. Read off `activeFormat` after both
        // selectors have run rather than inside them, because both are called
        // through `try?` — on the path where one of them throws, the device
        // keeps whatever format AVFoundation chose, and that is the format the
        // overlay has to describe.
        //
        // The wide arm is the LiDAR device's own video format. Its field of
        // view is taken as the depth map's, which is the assumption the box
        // rests on: the depth format is chosen from `activeFormat`'s supported
        // list, so the two are paired, but AVFoundation does not promise they
        // subtend the same angle. `depthDataOutput` writes the depth camera's
        // measured intrinsics into `manifest.json` on the first frame that
        // carries them, so a session can be checked against this afterwards
        // even though the screen cannot wait for it.
        let uwDimensions = CMVideoFormatDescriptionGetDimensions(
            ultraWide.activeFormat.formatDescription)
        let wideDimensions = CMVideoFormatDescriptionGetDimensions(
            lidar.activeFormat.formatDescription)
        let geometry = LensGeometry(
            ultraWideWidth: Double(uwDimensions.width),
            ultraWideHeight: Double(uwDimensions.height),
            ultraWideFieldOfView: Double(ultraWide.activeFormat.videoFieldOfView),
            wideWidth: Double(wideDimensions.width),
            wideHeight: Double(wideDimensions.height),
            wideFieldOfView: Double(lidar.activeFormat.videoFieldOfView))
        DispatchQueue.main.async { [weak self] in
            self?.onLensGeometry?(geometry)
        }
    }

    private func selectDepthFormat(on device: AVCaptureDevice) throws {
        // The largest depth map that can run in a multi-cam session, since the
        // depth is the scale reference and there is no reason to throw
        // resolution away.
        //
        // **This format also decides the wide video**, because the wide lens is
        // this same device and rides its active format. Ranking on depth alone
        // left that at 640x480 — not a choice anyone made, just whatever came
        // attached to the biggest depth map. That is small enough that matching
        // the wide against the ultra-wide runs out of pixels near the centre,
        // where lens distortion is least and the comparison is most trustworthy.
        //
        // Breaking ties on video size would cost no depth resolution and is the
        // obvious fix, but it is deliberately **not** made here: it changes what
        // the recorder captures, and this commit changes what the recorder
        // *reports*. Doing both at once would leave the next capture unable to
        // say which one moved it. The note below is what makes that follow-up
        // measurable, so it comes first.
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

        // The wide lens's own format and field of view. The ultra-wide has said
        // this about itself since the format was pinned, and the asymmetry cost
        // real work: `eval/pi3_poseless.py` can trust the logged number for the
        // ultra-wide and has to *assume* the factory calibration for the wide,
        // which is the exact assumption the ultra-wide path was fixed to stop
        // making. Now neither arm has to be inferred.
        //
        // Two notes, matching what the ultra-wide already writes: one that reads
        // as prose and one a parser can take the number from without guessing.
        let v = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
        notes.append("wide format \(v.width)x\(v.height) "
                     + "at \(format.videoFieldOfView) degrees")
        notes.append("wide fov \(format.videoFieldOfView)")
    }

    private func selectUltraWideFormat(on device: AVCaptureDevice) throws {
        // Widest field first, then most pixels. Field of view is what this lens
        // is here for, and a format that trades it away for resolution defeats
        // the reason for leaving ARKit at all.
        let usable = device.formats.filter { $0.isMultiCamSupported }
        guard let format = usable.max(by: { a, b in
            if abs(a.videoFieldOfView - b.videoFieldOfView) > 0.5 {
                return a.videoFieldOfView < b.videoFieldOfView
            }
            let da = CMVideoFormatDescriptionGetDimensions(a.formatDescription)
            let db = CMVideoFormatDescriptionGetDimensions(b.formatDescription)
            return Int(da.width) * Int(da.height) < Int(db.width) * Int(db.height)
        }) else {
            throw RecorderError.unsupported("The ultra-wide has no multi-cam format.")
        }
        try device.lockForConfiguration()
        device.activeFormat = format
        device.unlockForConfiguration()
        let d = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
        notes.append("ultra-wide format pinned to \(d.width)x\(d.height) "
                     + "at \(format.videoFieldOfView) degrees")
    }

    /// Records `isGeometricDistortionCorrectionEnabled` per lens, and pins it
    /// where the current value is known.
    ///
    /// **Why the ultra-wide is pinned to `true`, and why that is not a change.**
    /// It is already the default for that lens (`docs/ULTRAWIDE.md`), so every
    /// session ever recorded here ran with it on and every measurement on
    /// `docs/3DGS.md` describes frames delivered rectified. Setting it to
    /// `false` would hand over the raw lens and invalidate that corpus; the
    /// point of the write is not to pick a different value but to stop the
    /// value being picked by a default.
    ///
    /// **Why the wide is recorded and not pinned.** Nothing here has measured
    /// what that lens defaults to, and writing a value to find out would change
    /// the frames of every future session against a corpus recorded under the
    /// old one. Recording it costs nothing and is what makes the question
    /// answerable from a session rather than from a guess.
    ///
    /// **Why it matters that nobody wrote it down.** `UltraWideProbe` had to
    /// force this *off* to make the factory distortion tables appear at all,
    /// and wrote `geometric_distortion_correction: false` into `calib/` —
    /// truthfully about the probe, and misleadingly about the recorder, which
    /// never touched the property. Three attempts to give the ultra-wide a
    /// camera model then failed, each one correcting an image AVFoundation had
    /// already corrected. A checkerboard settled it at +9.6 px delivered
    /// against the 113.8 px a raw lens would bow.
    ///
    /// So the value goes in the manifest, per lens, as the device reports it
    /// rather than as it was requested — a device that refuses the write should
    /// not be able to look like one that accepted it.
    private func noteDistortionCorrection(on device: AVCaptureDevice,
                                          pinTo wanted: Bool?) {
        let name = shortName(device.deviceType)
        guard device.isGeometricDistortionCorrectionSupported else {
            notes.append("geometric distortion correction not supported on \(name)")
            return
        }
        if let wanted {
            do {
                try device.lockForConfiguration()
                device.isGeometricDistortionCorrectionEnabled = wanted
                device.unlockForConfiguration()
            } catch {
                notes.append("! could not set geometric distortion correction on "
                             + "\(name): \(error.localizedDescription)")
            }
        }
        // Read back rather than assumed. A device that refused the write should
        // not be able to look like one that took it, and the lens this is not
        // pinned on has to report something too.
        let actual = device.isGeometricDistortionCorrectionEnabled
        distortionCorrection[name] = actual
        var note = "geometric distortion correction \(actual) on \(name)"
        if let wanted {
            if actual != wanted { note += " — asked for \(wanted) and did not get it" }
        } else {
            note += " — read, not pinned"
        }
        notes.append(note)
    }

    private func lockFocus(on device: AVCaptureDevice) throws {
        try device.lockForConfiguration()
        if device.isFocusModeSupported(.locked) { device.focusMode = .locked }
        if device.isExposureModeSupported(.continuousAutoExposure) {
            device.exposureMode = .continuousAutoExposure
        }
        device.unlockForConfiguration()
    }

    /// Freezes exposure on both lenses once auto-exposure has had time to
    /// settle, if the preset asks for it.
    ///
    /// **Why a scan wants this and an episode does not.** A room does not fit
    /// in one walk, so several fragments have to merge into one map, and
    /// `tools/align_sessions.py` scores a merge partly on whether the
    /// photographs agree. Two lenses each running their own continuous
    /// auto-exposure means brightness drifts within a walk and between walks,
    /// and the reconstruction inherits it — the downstream remedy is a
    /// per-image appearance embedding, which is a model learning to undo
    /// something the capture chose to do.
    ///
    /// **Why it is delayed rather than set at configuration time.** `.locked`
    /// freezes whatever the device has converged on *now*. Locking before the
    /// session has seen the room pins a value measured against the inside of a
    /// pocket.
    ///
    /// **Why the measured values go into the notes.** A lock that caught a bad
    /// moment produces a whole session at the wrong exposure and looks
    /// identical on disk to one that worked. The duration and ISO it settled on
    /// are the only way to tell afterwards.
    private func scheduleExposureLock() {
        guard let delay = preset.lockExposureAfterSeconds else {
            notes.append("exposure left on continuous auto — the "
                         + "\(preset.rawValue) preset does not lock it")
            return
        }
        queue.asyncAfter(deadline: .now() + delay) { [weak self] in
            guard let self, self.running, !self.stopping else { return }
            for device in [self.lidar, self.ultraWide].compactMap({ $0 }) {
                self.lockExposure(on: device, after: delay)
            }
        }
    }

    private func lockExposure(on device: AVCaptureDevice, after delay: Double) {
        let name = shortName(device.deviceType)
        guard device.isExposureModeSupported(.locked) else {
            notes.append("! \(name) does not support an exposure lock — "
                         + "left on auto")
            return
        }
        // Recorded, not waited on. Blocking the capture queue until the device
        // stops adjusting would stall depth writes, and a lock taken while it
        // was still moving is worth knowing about rather than worth avoiding at
        // that price.
        let stillMoving = device.isAdjustingExposure
        do {
            try device.lockForConfiguration()
            device.exposureMode = .locked
            device.unlockForConfiguration()
        } catch {
            notes.append("! could not lock exposure on \(name): "
                         + error.localizedDescription)
            return
        }
        let milliseconds: Double = round(device.exposureDuration.seconds * 100_000) / 100
        let iso = Int(device.iso.rounded())
        let caveat = stillMoving ? " — still adjusting when locked" : ""
        notes.append("exposure locked on \(name) after \(delay) s "
                     + "at \(milliseconds) ms, ISO \(iso)\(caveat)")
    }

    /// Ends the session on the preset's cap.
    ///
    /// A one-shot timer rather than a check per frame: the delegate callbacks
    /// are what the cap exists to bound, so hanging the bound off them would
    /// make it silent exactly when capture had already stopped delivering.
    private func scheduleDurationCap() {
        guard let cap = preset.maxDurationSeconds else { return }
        queue.asyncAfter(deadline: .now() + cap) { [weak self] in
            guard let self, self.running, !self.stopping else { return }
            self.terminationReason = "durationCap"
            self.notes.append("stopped by the \(self.preset.rawValue) preset's "
                              + "\(Int(cap)) s cap")
            self.stop()
        }
    }

    private func shortName(_ type: AVCaptureDevice.DeviceType) -> String {
        type.rawValue.replacingOccurrences(of: "AVCaptureDeviceTypeBuiltIn",
                                           with: "")
    }

    // MARK: - Finishing

    private func closeFiles() throws {
        frameIndex?.close()
        wideFrameIndex?.close()
        depthIndex?.close()
        depthData?.close()
        motionWriter?.close()
        guard let dir = directory else { return }
        for note in notes {
            try? eventWriter?.write(SessionEvent(t: Clock.now(), kind: "multicam", detail: note))
        }
        eventWriter?.close()

        // Built outside the dictionary literal. That literal is already a large
        // heterogeneous `[String: Any]`, which is where Swift's type checker
        // gets expensive, and an optional-chained expression inside one is a
        // cheap way to turn a build into a timeout.
        let exposureLock: String
        if let delay = preset.lockExposureAfterSeconds {
            exposureLock = "locked after \(delay) s on both lenses — see the "
                + "notes for the duration and ISO it settled on"
        } else {
            exposureLock = "none — continuous auto-exposure on both lenses"
        }
        let durationCap: Double = preset.maxDurationSeconds ?? 0

        // The manifest says what is *not* here. A reader that finds no
        // `pose.jsonl` should not have to guess whether the recorder crashed.
        let manifest: [String: Any] = [
            "id": dir.lastPathComponent,
            "kind": "multicam",
            // Which preset, and the rate it actually asked for. Both, because
            // one without the other is not enough: a name with no number is
            // unreadable once the preset's definition moves, and a number with
            // no name does not say what else moved with it.
            "preset": preset.rawValue,
            "stills_hz": stillsHz,
            "exposure_lock": exposureLock,
            // Per lens, as the device reported it after configuration. Absent
            // from every session recorded before this build, which is exactly
            // the gap it closes — see `pinDistortionCorrection`.
            "geometric_distortion_correction": distortionCorrection,
            "max_duration_s": durationCap,
            "termination": terminationReason,
            // `pi3_chain.py` reads `pose.jsonl`, which is the one file this
            // recorder deliberately does not write, so it cannot run here.
            // `pi3_poseless.py` is the tool for a session with no tracker.
            "poses": "none — this session has no tracker; run eval/pi3_poseless.py",
            // Which build wrote this. Without it the only way to date a session
            // was to look for the absence of a note, and that is how a capture
            // came back on the old build and took a day to notice.
            "appVersion": Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "?",
            "appBuild": Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "?",
            "images": imageCount,
            "wide_images": wideCount,
            "depth_frames": depthCount,
            "images_dropped_for_rate": droppedImages,
            // Late drops, which are the load number. `images_dropped_for_rate`
            // above is the rate gate doing its job and is large on every
            // healthy session; these two should be **zero**, and a session
            // where they are not is a session whose capture queue could not
            // keep up with what this app asked of it.
            "images_dropped_late": lateDrops,
            "wide_images_dropped_late": lateWideDrops,
            "lens": "UltraWideCamera",
            "second_lens": wideCount > 0
                ? "WideAngleCamera in \(Self.wideFramesDirectory)/, the same walk "
                  + "through the other lens — the depth is already in its frame"
                : "none",
            "depth_from": "LiDARDepthCamera, filtering disabled, holes mean no return",
            // Present unless the device withheld calibration for this format,
            // and the string says which — a reader must not have to tell an
            // absent measurement from one that came back empty.
            "depth_calibration": depthCalibration
                ?? "none — no depth frame this session carried "
                 + "cameraCalibrationData; fall back to scaling calib/",
            // This used to say "scale intrinsics by the image size". That is
            // wrong whenever the active format is not the calibrated one, which
            // is the normal case: `calib/` is measured at 4032x3024 and implies
            // 102.5 degrees, and the pinned ultra-wide format reports 106.2. The
            // notes carry each lens's own active field of view; use those.
            "calibration": "calib/iphone17-1_ultrawide.json — take the focal "
                         + "length from this session's logged field of view, not "
                         + "by scaling the calibration to the image size, and "
                         + "apply R x + t to reach the wide camera's frame, "
                         + "which is where the depth already is",
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

    /// Record the depth camera's own intrinsics, once, from the first frame
    /// that carries them.
    ///
    /// **This is measured, not inferred.** Everything downstream that puts
    /// depth on an image has so far derived the depth grid's intrinsics by
    /// scaling `calib/iphone17-1_wide.json` — factory numbers for 4032x3024 —
    /// down to the depth resolution. That is the same assumption the ultra-wide
    /// path was fixed to stop making, because the active format is not the
    /// calibrated one, and nothing checked it for the depth camera. iOS hands
    /// the real thing over on every depth frame and the recorder was dropping
    /// it.
    ///
    /// `intrinsicMatrixReferenceDimensions` is the half that makes it usable:
    /// the intrinsics belong to that frame, not to the depth map's own
    /// dimensions, so a reader can scale them correctly instead of guessing
    /// which frame they were meant for. The extrinsic is here too — `calib/`
    /// argues from a probe capture that this camera is the reference and gets
    /// the identity, and this lets a session confirm it rather than inherit it.
    private func captureDepthCalibration(from depthData: AVDepthData) {
        guard depthCalibration == nil,
              let c = depthData.cameraCalibrationData else { return }
        let k = c.intrinsicMatrix
        let ref = c.intrinsicMatrixReferenceDimensions
        let e = c.extrinsicMatrix
        depthCalibration = [
            "fx": Double(k.columns.0.x), "fy": Double(k.columns.1.y),
            "cx": Double(k.columns.2.x), "cy": Double(k.columns.2.y),
            "reference_dimensions": [Double(ref.width), Double(ref.height)],
            "pixel_size_mm": Double(c.pixelSize),
            "lens_distortion_center": [Double(c.lensDistortionCenter.x),
                                       Double(c.lensDistortionCenter.y)],
            // Columns, matching how `calib/` writes the same thing: three
            // rotation columns then the translation, in millimetres.
            "extrinsic_matrix_columns": [
                [Double(e.columns.0.x), Double(e.columns.0.y), Double(e.columns.0.z)],
                [Double(e.columns.1.x), Double(e.columns.1.y), Double(e.columns.1.z)],
                [Double(e.columns.2.x), Double(e.columns.2.y), Double(e.columns.2.z)],
                [Double(e.columns.3.x), Double(e.columns.3.y), Double(e.columns.3.z)],
            ],
            "note": "the depth camera's own intrinsics for the format this "
                  + "session actually ran, read off AVDepthData rather than "
                  + "scaled from calib/ — scale these by the depth map size "
                  + "over reference_dimensions",
        ]
        let hfov = 2 * atan(Double(ref.width) / (2 * Double(k.columns.0.x)))
        notes.append("depth fov \(hfov * 180 / .pi)")
    }

    func depthDataOutput(_ output: AVCaptureDepthDataOutput,
                         didOutput depthData: AVDepthData,
                         timestamp: CMTime,
                         connection: AVCaptureConnection) {
        guard running, !stopping, let writer = self.depthData else { return }
        captureDepthCalibration(from: depthData)
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
        let isWide = output === wideOutput
        let stamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds
        let last = isWide ? lastWideStamp : lastImageStamp
        guard stamp - last >= 1.0 / stillsHz else {
            if !isWide { droppedImages += 1 }
            return
        }
        guard let buffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let image = CIImage(cvPixelBuffer: buffer)

        // The aiming preview. Ultra-wide only, and deliberately *inside* the
        // rate gate above: it therefore never touches a frame that this
        // recorder was not already going to encode, so the frames the gate
        // discards stay as cheap as they are today. At the vln preset the two
        // rates coincide and every encoded frame is previewed; at scan (10 Hz)
        // every second one is.
        //
        // Cost, on this frame, on `queue`, against the JPEG encode three lines
        // below that the same frame is already paying for: one downscale render
        // through the hoisted `CIContext` and a readback of 384x216 BGRA, about
        // 0.33 MB, against a 3840x2160 encode whose *output* alone has a median
        // of 1386 KB (docs/SCAN_PRESET.md). At most five of these a second.
        //
        // **Why not `AVCaptureVideoPreviewLayer`.** It is cheaper per frame —
        // no readback at all — and it was still not taken, for three reasons in
        // descending order. (1) It needs its own `AVCaptureConnection` from the
        // ultra-wide port, and a multi-cam session prices connections: this one
        // already logs `systemPressureCost` 1.985, and a configuration that is
        // legal today could stop being legal, on the device, in a way no review
        // here would catch. (2) The overlay has to be registered to the *frame*,
        // and a preview layer's drawn rectangle is decided by `videoGravity`
        // inside a layer SwiftUI does not own — a second geometry convention
        // beside the aspect-fit one this app already has. (3) It would need a
        // `UIViewRepresentable` and its own orientation handling, where this
        // path reuses `RecordView`'s. The readback is 1.6 MB/s against a session
        // already writing 11 MB/s to NAND, so it is not the binding cost.
        if !isWide, let onPreview = onPreview,
           stamp - lastPreviewStamp >= Self.previewInterval {
            lastPreviewStamp = stamp
            let width = Double(CVPixelBufferGetWidth(buffer))
            let scale = width > 0 ? min(1.0, Self.previewWidth / width) : 1.0
            let small = image.transformed(
                by: CGAffineTransform(scaleX: CGFloat(scale), y: CGFloat(scale)))
            if let preview = ciContext.createCGImage(small, from: small.extent) {
                DispatchQueue.main.async { onPreview(preview) }
            }
        }

        guard let jpeg = ciContext.jpegRepresentation(
            of: image, colorSpace: CGColorSpaceCreateDeviceRGB(),
            options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 0.85])
        else { return }

        let index = isWide ? wideCount : imageCount
        let folder = isWide ? Self.wideFramesDirectory
                            : SessionStore.Filename.framesDirectory
        let relative = "\(folder)/\(String(format: "%06d.jpg", index))"
        do {
            try jpeg.write(to: dir.appendingPathComponent(relative), options: .atomic)
            let entry = FrameIndexEntry(
                t: stamp, frame: index, file: relative,
                width: CVPixelBufferGetWidth(buffer),
                height: CVPixelBufferGetHeight(buffer),
                bytes: jpeg.count)
            if isWide {
                try wideFrameIndex?.write(entry)
                wideCount += 1
                lastWideStamp = stamp
            } else {
                try frameIndex?.write(entry)
                imageCount += 1
                lastImageStamp = stamp
            }
        } catch {
            status("image write failed: \(error.localizedDescription)")
        }
    }

    /// AVFoundation discarded a frame because this queue was still busy.
    ///
    /// Runs on `queue`, like `captureOutput(_:didOutput:from:)`, so the two
    /// counters need no lock. Counting only — a status line per drop would
    /// arrive by the thousand on a session that had a bad second, and the
    /// session-wide totals land in `manifest.json`.
    func captureOutput(_ output: AVCaptureOutput,
                       didDrop sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        if output === wideOutput {
            lateWideDrops += 1
        } else {
            lateDrops += 1
        }
    }
}
