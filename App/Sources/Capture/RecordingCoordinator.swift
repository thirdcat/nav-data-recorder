import Combine
import CoreLocation
import Foundation
import UIKit

/// Owns a recording session end to end: opens the files, starts the three
/// capture sources, funnels every sample onto one serial IO queue, and closes
/// everything down in an order that leaves a readable session on disk.
///
/// Threading contract, which the rest of the capture layer depends on:
///
/// - every writer is created, written, flushed and closed on `ioQueue`
/// - every `@Published` property is mutated on the main thread only
/// - the recorders deliver samples on their own queues and hop to `ioQueue` here
///
/// This is deliberately GCD rather than actors: the capture path is a fixed set
/// of serial queues with hard real-time-ish deadlines, and modelling it that
/// way is clearer than fighting actor hops at 100 Hz.
final class RecordingCoordinator: ObservableObject {

    enum State: Equatable {
        case idle
        case starting
        case recording
        case stopping
    }

    struct Stats: Equatable {
        var locations = 0
        var headings = 0
        var motion = 0
        var poses = 0
        var depthFrames = 0
        var videoFrames = 0
        var droppedVideoFrames = 0
        /// Fraction of the newest depth map ARKit rates medium or high.
        var depthUsable: Double = 0
        var currentRoll: Double?
        var bytesOnDisk: UInt64 = 0
    }

    // MARK: - Published state (main thread only)

    @Published private(set) var state: State = .idle
    @Published private(set) var sessionID: String?
    @Published private(set) var elapsed: TimeInterval = 0
    @Published private(set) var stats = Stats()
    @Published private(set) var lastLocation: CLLocation?
    @Published private(set) var thermalState: ProcessInfo.ThermalState = .nominal
    @Published private(set) var isThrottled = false
    @Published private(set) var freeBytes: Int64 = 0
    @Published private(set) var previewImage: CGImage?
    @Published var statusMessage: String?

    @Published var config: CaptureConfig {
        didSet { AppSettings.shared.captureConfig = config }
    }

    // MARK: - Internals

    private let ioQueue = DispatchQueue(label: "nav.io", qos: .utility)
    private let locationRecorder: LocationRecorder
    private let motionRecorder: MotionRecorder
    private let arRecorder = ARRecorder()
    private let uploadManager: UploadManager

    // Touched on ioQueue only.
    private var locationWriter: JSONLWriter?
    private var headingWriter: JSONLWriter?
    private var motionWriter: JSONLWriter?
    private var poseWriter: JSONLWriter?
    private var planeWriter: JSONLWriter?
    private var frameWriter: JSONLWriter?
    private var depthIndexWriter: JSONLWriter?
    private var depthData: BufferedFileWriter?
    private var confidenceData: BufferedFileWriter?
    private var eventWriter: JSONLWriter?

    // Touched on main only.
    private var manifest: SessionManifest?
    private var startClock: Double = 0
    private var anchor: Double = 0
    private var uiTimer: Timer?
    private var cancellables = Set<AnyCancellable>()

    /// Stop cleanly rather than let writes start failing mid-drive.
    private static let minimumFreeBytes: Int64 = 2 * 1024 * 1024 * 1024

    init(uploadManager: UploadManager) {
        self.uploadManager = uploadManager
        self.config = AppSettings.shared.captureConfig
        self.locationRecorder = LocationRecorder(queue: ioQueue)
        self.motionRecorder = MotionRecorder(queue: ioQueue)

        UIDevice.current.isBatteryMonitoringEnabled = true
        self.thermalState = ProcessInfo.processInfo.thermalState
        self.freeBytes = SessionStore.availableCapacity()

        wireRecorders()
        observeSystem()
    }

    var isRecording: Bool { state == .recording || state == .starting }
    var canRecord: Bool { ARRecorder.isSupported }
    var hasLiDAR: Bool { ARRecorder.hasLiDAR }

    // MARK: - Permissions

    func requestPermissions() {
        locationRecorder.requestAuthorization()
    }

    var locationAuthorization: CLAuthorizationStatus {
        locationRecorder.authorizationStatus
    }

    // MARK: - Start

    func start() {
        guard state == .idle else { return }
        guard ARRecorder.isSupported else {
            statusMessage = "This device does not support ARKit world tracking."
            return
        }
        guard SessionStore.availableCapacity() > Self.minimumFreeBytes else {
            statusMessage = "Not enough free space to start a recording."
            return
        }

        state = .starting
        statusMessage = nil

        let id = SessionStore.makeSessionID()
        let dir: URL
        do {
            dir = try SessionStore.createDirectory(for: id)
        } catch {
            state = .idle
            statusMessage = "Could not create the session folder: \(error.localizedDescription)"
            return
        }

        anchor = Clock.wallClockAnchor()
        startClock = Clock.now()

        let newManifest = SessionManifest(
            id: id,
            startedAt: Date().timeIntervalSince1970,
            endedAt: nil,
            clockAnchor: anchor,
            startClock: startClock,
            device: Self.deviceInfo(config: config),
            config: config,
            video: nil,
            appVersion: Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "?",
            appBuild: Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "?")
        self.manifest = newManifest
        self.sessionID = id

        let cfg = config
        ioQueue.async { [weak self] in
            guard let self = self else { return }
            do {
                self.locationWriter = try JSONLWriter(url: dir.appendingPathComponent(SessionStore.Filename.location))
                self.headingWriter = try JSONLWriter(url: dir.appendingPathComponent(SessionStore.Filename.heading))
                self.motionWriter = try JSONLWriter(url: dir.appendingPathComponent(SessionStore.Filename.motion))
                self.eventWriter = try JSONLWriter(url: dir.appendingPathComponent(SessionStore.Filename.events))
                self.poseWriter = try JSONLWriter(url: dir.appendingPathComponent(SessionStore.Filename.pose))
                if cfg.captureMode == .stills {
                    self.frameWriter = try JSONLWriter(
                        url: dir.appendingPathComponent(SessionStore.Filename.frameIndex))
                }
                if cfg.detectPlanes {
                    self.planeWriter = try JSONLWriter(
                        url: dir.appendingPathComponent(SessionStore.Filename.planes))
                }
                if cfg.recordDepth {
                    self.depthIndexWriter = try JSONLWriter(
                        url: dir.appendingPathComponent(SessionStore.Filename.depthIndex))
                    self.depthData = try BufferedFileWriter(
                        url: dir.appendingPathComponent(SessionStore.Filename.depthData),
                        bufferSize: 1024 * 1024)
                    if cfg.recordConfidence {
                        self.confidenceData = try BufferedFileWriter(
                            url: dir.appendingPathComponent(SessionStore.Filename.confidenceData),
                            bufferSize: 512 * 1024)
                    }
                }
            } catch {
                let message = error.localizedDescription
                DispatchQueue.main.async {
                    self.statusMessage = "Could not open the session files: \(message)"
                    self.abortStart()
                }
                return
            }

            // Non-fatal if this fails: the manifest is rewritten on stop.
            try? SessionStore.writeManifest(newManifest)

            DispatchQueue.main.async {
                self.beginCapture(dir: dir)
            }
        }
    }

    private func beginCapture(dir: URL) {
        guard state == .starting else { return }

        let startingThermal = ProcessInfo.processInfo.thermalState
        logEvent("session.start",
                 "id=\(sessionID ?? "?") battery=\(UIDevice.current.batteryLevel) "
                 + "thermal=\(Self.describe(startingThermal))")
        // Seed the throttle from the state we are actually starting in.
        applyThermalState(startingThermal)

        locationRecorder.start(anchor: anchor)
        motionRecorder.start(hz: config.motionHz,
                             magnetometerCorrected: config.useMagnetometerCorrection)
        arRecorder.start(config: config, sessionDirectory: dir)

        // No camera with the screen off, so the screen stays on for the whole
        // drive. This is why the app assumes a mount and a charger.
        UIApplication.shared.isIdleTimerDisabled = true

        state = .recording
        startUITimer()
    }

    private func abortStart() {
        state = .idle
        sessionID = nil
        closeWriters(completion: nil)
    }

    // MARK: - Stop

    func stop(reason: String? = nil) {
        guard state == .recording || state == .starting else { return }
        state = .stopping

        uiTimer?.invalidate()
        uiTimer = nil
        UIApplication.shared.isIdleTimerDisabled = false

        locationRecorder.stop()
        motionRecorder.stop()
        logEvent("session.stop", reason ?? "user")

        manifest?.endedAt = Date().timeIntervalSince1970
        manifest?.terminationReason = reason

        // The video has to be finalised before anything marks the session
        // complete: an un-finalised .mov is unreadable, and the upload queue
        // keys off that marker.
        arRecorder.stop { [weak self] in
            DispatchQueue.main.async {
                guard let self = self else { return }
                self.manifest?.video = self.arRecorder.videoInfo
                self.finishSession()
            }
        }
    }

    private func finishSession() {
        guard let id = sessionID else {
            state = .idle
            return
        }
        var finished = self.manifest

        closeWriters { counts in
            finished?.counts = counts
            if let m = finished {
                try? SessionStore.writeManifest(m)
            }
            SessionStore.markComplete(id: id)

            DispatchQueue.main.async { [weak self] in
                guard let self = self else { return }
                self.state = .idle
                self.sessionID = nil
                self.manifest = nil
                self.previewImage = nil
                self.elapsed = 0
                self.freeBytes = SessionStore.availableCapacity()
                self.statusMessage = "Session \(id) saved."
                self.uploadManager.enqueue(sessionID: id)
            }
        }
    }

    /// Closes every writer on the IO queue and reports row counts, which land in
    /// the manifest as an integrity check against the files themselves.
    /// `completion` runs on `ioQueue`.
    private func closeWriters(completion: (([String: Int]) -> Void)?) {
        ioQueue.async { [weak self] in
            guard let self = self else {
                completion?([:])
                return
            }
            var counts: [String: Int] = [:]
            counts["location"] = self.locationWriter?.count ?? 0
            counts["heading"] = self.headingWriter?.count ?? 0
            counts["motion"] = self.motionWriter?.count ?? 0
            counts["pose"] = self.poseWriter?.count ?? 0
            counts["planes"] = self.planeWriter?.count ?? 0
            counts["frames"] = self.frameWriter?.count ?? 0
            counts["depth"] = self.depthIndexWriter?.count ?? 0
            counts["events"] = self.eventWriter?.count ?? 0

            self.locationWriter?.close(); self.locationWriter = nil
            self.headingWriter?.close(); self.headingWriter = nil
            self.motionWriter?.close(); self.motionWriter = nil
            self.poseWriter?.close(); self.poseWriter = nil
            self.planeWriter?.close(); self.planeWriter = nil
            self.frameWriter?.close(); self.frameWriter = nil
            self.depthIndexWriter?.close(); self.depthIndexWriter = nil
            self.depthData?.close(); self.depthData = nil
            self.confidenceData?.close(); self.confidenceData = nil
            self.eventWriter?.close(); self.eventWriter = nil

            completion?(counts)
        }
    }

    // MARK: - Wiring

    private func wireRecorders() {
        // Location and motion callbacks already arrive on ioQueue.
        locationRecorder.onLocation = { [weak self] sample in
            try? self?.locationWriter?.write(sample)
        }
        locationRecorder.onHeading = { [weak self] sample in
            try? self?.headingWriter?.write(sample)
        }
        locationRecorder.onEvent = { [weak self] kind, detail in
            self?.writeEvent(kind: kind, detail: detail)
        }

        motionRecorder.onMotion = { [weak self] sample in
            try? self?.motionWriter?.write(sample)
        }
        motionRecorder.onEvent = { [weak self] kind, detail in
            self?.writeEvent(kind: kind, detail: detail)
        }

        // ARKit callbacks arrive on the AR queue and hop over.
        arRecorder.onPose = { [weak self] sample in
            guard let self = self else { return }
            self.ioQueue.async {
                try? self.poseWriter?.write(sample)
            }
        }
        arRecorder.onDepth = { [weak self] depth, confidence, t, frame, width, height in
            guard let self = self else { return }
            self.ioQueue.async {
                guard let bin = self.depthData else { return }
                var confidenceOffset: UInt64?
                var confidenceLength: Int?
                if let confidence = confidence, let confBin = self.confidenceData {
                    confidenceOffset = confBin.nextOffset
                    confidenceLength = confidence.count
                    try? confBin.append(confidence)
                }
                // Read the offset before appending — it is where this payload
                // starts, not where it ends.
                let offset = bin.nextOffset
                try? bin.append(depth)
                try? self.depthIndexWriter?.write(DepthIndexEntry(
                    t: t, frame: frame,
                    offset: offset, length: depth.count,
                    width: width, height: height,
                    format: "float16",
                    confidenceOffset: confidenceOffset,
                    confidenceLength: confidenceLength))
            }
        }
        arRecorder.onPlane = { [weak self] sample in
            guard let self = self else { return }
            self.ioQueue.async {
                try? self.planeWriter?.write(sample)
            }
        }
        arRecorder.onStill = { [weak self] entry in
            guard let self = self else { return }
            self.ioQueue.async {
                try? self.frameWriter?.write(FrameIndexEntry(
                    t: entry.t, frame: entry.frame, file: entry.file,
                    width: entry.width, height: entry.height, bytes: entry.bytes))
            }
        }
        arRecorder.onEvent = { [weak self] kind, detail in
            self?.writeEvent(kind: kind, detail: detail)
        }
        arRecorder.onPreview = { [weak self] image in
            DispatchQueue.main.async {
                self?.previewImage = image
            }
        }
    }

    private func observeSystem() {
        NotificationCenter.default.publisher(for: ProcessInfo.thermalStateDidChangeNotification)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in self?.handleThermalChange() }
            .store(in: &cancellables)

        NotificationCenter.default.publisher(for: UIApplication.didEnterBackgroundNotification)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                guard let self = self, self.isRecording else { return }
                // Camera capture is about to die; GPS and IMU keep going.
                self.logEvent("app.background", "camera and depth suspended by iOS")
                self.flushWriters()
            }
            .store(in: &cancellables)

        NotificationCenter.default.publisher(for: UIApplication.willEnterForegroundNotification)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                guard let self = self, self.isRecording else { return }
                self.logEvent("app.foreground", "")
            }
            .store(in: &cancellables)

        NotificationCenter.default.publisher(for: UIApplication.willTerminateNotification)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                guard let self = self, self.isRecording else { return }
                // Best effort. The video will not be finalised, but every JSONL
                // line already flushed stays readable.
                self.flushWriters()
            }
            .store(in: &cancellables)
    }

    private func handleThermalChange() {
        let current = ProcessInfo.processInfo.thermalState
        logEvent("thermal", Self.describe(current))
        applyThermalState(current)
    }

    /// Bring the throttle in line with a thermal state.
    ///
    /// Called both on the change notification and at the start of a recording.
    /// The start case matters: `thermalStateDidChangeNotification` only fires on
    /// a *transition*, so a session begun while the device is already hot would
    /// otherwise run unthrottled — at full capture rate, making it hotter —
    /// until the state happened to move on its own.
    private func applyThermalState(_ current: ProcessInfo.ThermalState) {
        thermalState = current

        guard config.degradeOnThermalPressure else { return }

        switch current {
        case .serious, .critical:
            if !isThrottled {
                isThrottled = true
                arRecorder.isThrottled = true
                logEvent("thermal.throttle", "video and depth paused; GPS and IMU continue")
                statusMessage = "Device is hot — video and depth paused."
            }
        case .nominal, .fair:
            if isThrottled {
                isThrottled = false
                arRecorder.isThrottled = false
                logEvent("thermal.resume", "video and depth resumed")
                statusMessage = nil
            }
        @unknown default:
            break
        }
    }

    // MARK: - UI tick

    private func startUITimer() {
        uiTimer?.invalidate()
        let timer = Timer(timeInterval: 1.0, repeats: true) { [weak self] _ in
            self?.tick()
        }
        // .common so the clock keeps ticking while a list is being scrolled.
        RunLoop.main.add(timer, forMode: .common)
        uiTimer = timer
    }

    private func tick() {
        guard state == .recording else { return }
        elapsed = Clock.now() - startClock
        lastLocation = locationRecorder.lastLocation

        let free = SessionStore.availableCapacity()
        freeBytes = free
        if free < Self.minimumFreeBytes {
            statusMessage = "Stopped: the device is out of storage."
            stop(reason: "outOfStorage")
            return
        }

        let id = sessionID
        let ar = arRecorder.snapshot()

        ioQueue.async { [weak self] in
            guard let self = self else { return }
            var snapshot = Stats()
            snapshot.locations = self.locationWriter?.count ?? 0
            snapshot.headings = self.headingWriter?.count ?? 0
            snapshot.motion = self.motionWriter?.count ?? 0
            snapshot.depthFrames = self.depthIndexWriter?.count ?? 0
            snapshot.poses = ar.poses
            snapshot.videoFrames = ar.encodedFrames
            snapshot.droppedVideoFrames = ar.droppedFrames
            snapshot.depthUsable = ar.depthUsable
            snapshot.currentRoll = ar.currentRoll
            snapshot.bytesOnDisk = id.map { SessionStore.totalBytes(id: $0) } ?? 0
            DispatchQueue.main.async {
                self.stats = snapshot
            }
        }

        // Cheap, and occasionally the only explanation for why a drive cut out.
        if Int(elapsed) % 60 == 0 {
            logEvent("battery", "level=\(UIDevice.current.batteryLevel) state=\(UIDevice.current.batteryState.rawValue)")
        }
    }

    private func flushWriters() {
        ioQueue.async { [weak self] in
            guard let self = self else { return }
            try? self.locationWriter?.flush()
            try? self.headingWriter?.flush()
            try? self.motionWriter?.flush()
            try? self.poseWriter?.flush()
            try? self.planeWriter?.flush()
            try? self.frameWriter?.flush()
            try? self.depthIndexWriter?.flush()
            try? self.depthData?.flush()
            try? self.confidenceData?.flush()
            try? self.eventWriter?.flush()
        }
    }

    private func logEvent(_ kind: String, _ detail: String) {
        writeEvent(kind: kind, detail: detail)
    }

    /// Safe to call from any queue.
    private func writeEvent(kind: String, detail: String) {
        let t = Clock.now()
        ioQueue.async { [weak self] in
            try? self?.eventWriter?.write(SessionEvent(t: t, kind: kind, detail: detail))
        }
    }

    // MARK: - Helpers

    private static func deviceInfo(config: CaptureConfig) -> SessionManifest.DeviceInfo {
        SessionManifest.DeviceInfo(
            model: UIDevice.current.modelIdentifier,
            systemVersion: UIDevice.current.systemVersion,
            name: UIDevice.current.name,
            hasLiDAR: ARRecorder.hasLiDAR,
            // Recorded, not assumed: the attitude quaternions in motion.jsonl
            // are meaningless without knowing which frame they are relative to.
            attitudeReferenceFrame: MotionRecorder.describe(
                MotionRecorder.referenceFrame(
                    magnetometerCorrected: config.useMagnetometerCorrection)))
    }

    static func describe(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }
}

extension UIDevice {
    /// e.g. `iPhone17,1`. `UIDevice.model` only ever says "iPhone", which is not
    /// enough to know which sensors produced a session.
    var modelIdentifier: String {
        var info = utsname()
        uname(&info)
        let mirror = Mirror(reflecting: info.machine)
        return mirror.children.reduce(into: "") { result, element in
            guard let value = element.value as? Int8, value != 0 else { return }
            result.append(Character(UnicodeScalar(UInt8(bitPattern: value))))
        }
    }
}
