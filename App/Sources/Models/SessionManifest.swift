import Foundation

/// Written to `manifest.json` at session start and rewritten on stop. It is the
/// only file a downstream consumer has to read to understand the rest.
struct SessionManifest: Codable {
    /// Bump when the on-disk layout changes in a way that breaks readers.
    static let currentSchemaVersion = 1

    var schemaVersion: Int = SessionManifest.currentSchemaVersion
    var id: String
    /// Unix seconds.
    var startedAt: Double
    var endedAt: Double?
    /// `unixTime = sample.t + clockAnchor`. See `Clock`.
    var clockAnchor: Double
    /// Value of the master clock when recording started, so `t - startClock`
    /// gives seconds-into-session without needing wall time.
    var startClock: Double

    var device: DeviceInfo
    var config: CaptureConfig
    var video: VideoInfo?

    /// Rows written per stream, filled in on stop. A quick integrity check
    /// against the actual line counts.
    var counts: [String: Int] = [:]

    var appVersion: String
    var appBuild: String

    /// Set when the session ended for a reason other than the user tapping stop.
    var terminationReason: String?

    struct DeviceInfo: Codable {
        var model: String
        var systemVersion: String
        var name: String
        var hasLiDAR: Bool
        /// `CMAttitudeReferenceFrame` used for `MotionSample` quaternions.
        var attitudeReferenceFrame: String
    }

    struct VideoInfo: Codable {
        var file: String
        var width: Int
        var height: Int
        var codec: String
        /// Nominal capture rate; actual per-frame times live in pose.jsonl.
        var nominalFPS: Int
        var bitrate: Int
        /// PTS of the first frame, in the master clock domain. Video PTS values
        /// are written in that domain directly, so a frame's PTS equals the
        /// matching `PoseSample.t`.
        var firstFramePTS: Double
    }
}

/// User-facing capture settings, snapshotted into each session's manifest so a
/// recording can always be interpreted with the settings it was made under.
///
/// Defaults are tuned for **indoor, room-scale capture**. That is not a neutral
/// choice — see the individual notes. An outdoor or automotive profile would
/// want the opposite on several of these.
struct CaptureConfig: Codable, Equatable {

    /// How RGB is stored.
    enum CaptureMode: String, Codable, CaseIterable {
        /// Individual JPEGs, one file per captured frame.
        ///
        /// The right choice when the consumer is a training pipeline that wants
        /// posed images: each frame is addressable without decoding a video,
        /// and there is no seek-and-decode step between the recorder and the
        /// dataloader. Costs roughly twice the bytes of HEVC at the same rate.
        case stills
        /// A single HEVC .mov. Far smaller, but every frame has to be decoded
        /// out again before it can be used.
        case video
    }

    /// Stills by default: this recorder exists to produce posed RGB frames, and
    /// storing them as video only to extract them again is a lossy round trip.
    var captureMode: CaptureMode = .stills
    /// Hz for stills capture. Unused in video mode.
    var stillsHz: Double = 5
    /// JPEG quality, 0…1.
    var stillQuality: Double = 0.85

    var recordDepth: Bool = true
    var recordConfidence: Bool = false
    /// Record ARKit's detected floors, walls, tables and so on. Indoors these
    /// are real structure worth keeping; outdoors they are mostly noise.
    var detectPlanes: Bool = true

    /// Hz. ARKit delivers 60 on a 16 Pro; frames beyond this are dropped from
    /// the video track but their poses are still recorded. Video mode only.
    var videoFPS: Int = 30
    /// Hz. Independent of video because depth is ~100x more expensive per frame.
    ///
    /// Defaults to matching `stillsHz` so every captured image has a depth map
    /// taken from the same `ARFrame`.
    var depthHz: Double = 5
    /// Hz for `CMDeviceMotion`.
    var motionHz: Double = 100

    var videoBitrate: Int = 12_000_000

    /// Whether device attitude is magnetometer-corrected.
    ///
    /// Off indoors, deliberately. Correction pulls yaw towards magnetic north,
    /// and a home is full of things that lie about where that is — steel studs,
    /// wiring in the walls, appliances, speaker magnets. Uncorrected attitude
    /// has an arbitrary yaw origin but drifts smoothly, which is far easier to
    /// work with than one that snaps as you walk past a refrigerator.
    var useMagnetometerCorrection: Bool = false

    /// Pause video/depth capture when the device gets hot, keeping GPS and IMU
    /// running, rather than losing the session outright.
    var degradeOnThermalPressure: Bool = true

    static let `default` = CaptureConfig()
}
