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
struct CaptureConfig: Codable, Equatable {
    var recordVideo: Bool = true
    var recordDepth: Bool = true
    var recordConfidence: Bool = false

    /// Hz. ARKit delivers 60 on a 16 Pro; frames beyond this are dropped from
    /// the video track but their poses are still recorded.
    var videoFPS: Int = 30
    /// Hz. Independent of video because depth is ~100x more expensive per frame.
    var depthHz: Double = 5
    /// Hz for `CMDeviceMotion`.
    var motionHz: Double = 100

    var videoBitrate: Int = 12_000_000

    /// Pause video/depth capture when the device gets hot, keeping GPS and IMU
    /// running. Without this a long drive throttles into uselessness.
    var degradeOnThermalPressure: Bool = true

    static let `default` = CaptureConfig()
}
