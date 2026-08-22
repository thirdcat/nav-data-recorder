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
    /// Which named preset `config` matches: `"vln"`, `"scan"`, or `"custom"`.
    ///
    /// Derived from the values at session start rather than stored as a choice,
    /// so it cannot disagree with them — a manifest that said `scan` while
    /// carrying 5 Hz would be worse than no label at all.
    ///
    /// Optional because it did not exist before this build, and `nil` is the
    /// honest answer for a session recorded then: every capture before this
    /// point ran the 5 Hz VLN settings, but the manifest never said so and a
    /// reader should not have to infer it from the absence of a note.
    var preset: String?
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

    /// Which capture format to ask ARKit for.
    enum FormatPreference: String, Codable, CaseIterable {
        /// Largest 4:3 format. Keeps the full sensor height, so it sees more
        /// floor and ceiling — the axis indoor navigation cares about.
        case tallest
        /// Largest format by pixel count, which on this hardware means a 16:9
        /// 4K one. More horizontal detail, less vertical field of view.
        case highestResolution
    }

    /// Defaults to 4:3. The reasoning is in docs/DATA_FORMAT.md, and it is
    /// worth re-testing per device: switch this, record, and compare the
    /// `sensor fov` line the reader prints.
    var formatPreference: FormatPreference = .tallest

    /// Stills by default: this recorder exists to produce posed RGB frames, and
    /// storing them as video only to extract them again is a lossy round trip.
    var captureMode: CaptureMode = .stills
    /// Hz for stills capture. Unused in video mode.
    ///
    /// 5 Hz is a **VLN dataset** rate, not a capture limit: episodes are
    /// consumed at 5 Hz and storing more would be decimated away again. Depth
    /// already runs at 30 and pose at 60 on the same frames, so this is the one
    /// stream that is throttled, and `Preset.scan` is what raises it. See
    /// `docs/SCAN_PRESET.md`.
    var stillsHz: Double = 5
    /// JPEG quality, 0…1.
    var stillQuality: Double = 0.85

    var recordDepth: Bool = true
    /// On by default, despite costing half again what depth does.
    ///
    /// ARKit's depth is guided by the colour image, so a dim room returns a
    /// full-looking map that is almost entirely `ARConfidenceLevel.low` — a
    /// whole 30 m session measured 98.4% low and not one high-confidence pixel.
    /// Without this map that is invisible, and depth odometry scored on it
    /// measures the lighting rather than the method. A byte per pixel is a
    /// cheap price for knowing whether the expensive stream is worth anything.
    var recordConfidence: Bool = true
    /// Record ARKit's detected floors, walls, tables and so on. Indoors these
    /// are real structure worth keeping; outdoors they are mostly noise.
    var detectPlanes: Bool = true

    /// Hz. ARKit delivers 60 on a 16 Pro; frames beyond this are dropped from
    /// the video track but their poses are still recorded. Video mode only.
    var videoFPS: Int = 30
    /// Hz for depth, in both capture modes.
    ///
    /// Deliberately higher than `stillsHz`. The export rate and the capture rate
    /// are different questions: episodes want 5 Hz, but frame-to-frame depth
    /// registration wants every frame it can get, because ICP converges on small
    /// motion and 5 Hz at walking pace puts frames ~10 cm and several degrees
    /// apart — outside where it works at all. Decimating afterwards is free;
    /// frames never captured are gone.
    ///
    /// Depth is a superset of the image frames rather than a separate stream:
    /// every captured image still gets a depth map on its own `frame`, so the
    /// join that posed-image consumers rely on is unaffected.
    ///
    /// The cost is real — 256x192 float16 is ~98 KB a frame, so 30 Hz is about
    /// 3 MB/s. Best-effort rather than guaranteed: ARKit delivers `sceneDepth`
    /// on the frames it has it for, and thermal throttling cuts this first.
    var depthHz: Double = 30
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

    /// The stored defaults are the VLN preset, unchanged. Every session in
    /// `~/nav_data` was recorded under exactly these values, so `.vln` is
    /// bit-for-bit what the recorder did before presets existed.
    static let `default` = CaptureConfig()

    // MARK: - Presets

    /// A named bundle of capture settings, applied as a unit.
    ///
    /// **The problem this solves is inheritance, not convenience.** Settings
    /// live in one `UserDefaults` blob shared by every recording. Raise the
    /// stills rate for a scan and the next VLN episode silently records at the
    /// scan rate, with nothing on disk to say the dataset changed underneath
    /// the reader. A preset makes the whole bundle move together and makes the
    /// manifest name it.
    ///
    /// Nothing here is stored. `CaptureConfig` gains no coding key, so every
    /// config already persisted and every manifest already written keeps
    /// decoding exactly as before — which is the point, since Swift's
    /// synthesised `Decodable` does *not* fall back to a property's default
    /// value for a missing key, and one new stored field would have quietly
    /// reset every existing install's settings.
    enum Preset: String, CaseIterable, Hashable {
        /// 5 Hz stills. The rate VLN episodes are consumed at.
        case vln
        /// 30 Hz stills for Gaussian-splat capture, with a duration cap.
        case scan

        var title: String {
            switch self {
            case .vln: return "VLN episodes — 5 Hz"
            case .scan: return "3DGS scan — 30 Hz"
            }
        }

        var summary: String {
            switch self {
            case .vln:
                return "The dataset rate. Depth 30 Hz, pose 60 Hz, no duration cap."
            case .scan:
                return "Every ARKit frame that carries depth. ~16 MB/s, capped at 60 s."
            }
        }

        /// The settings this preset owns, on top of the shared defaults.
        var configuration: CaptureConfig {
            var config = CaptureConfig()
            switch self {
            case .vln:
                break
            case .scan:
                // 30 rather than 60: ARKit delivers 60 Hz on the active
                // 1920x1440 format, but the stills encoder is one serial queue
                // whose own comment prices a JPEG at ~20 ms, which does not fit
                // a 16.7 ms budget. At 30 Hz the image gate and the depth gate
                // share a schedule, so every image still gets a depth map on
                // its own `frame` and depth does not double.
                config.stillsHz = 30
            }
            return config
        }

        /// Stop the recording after this many seconds, or `nil` for no cap.
        ///
        /// Not a thermal *measurement* — see `docs/SCAN_PRESET.md` for what the
        /// events actually show. It is a bound on an untested load: the scan
        /// preset writes about six times the image bytes of a VLN session, and
        /// the longest capture ever recorded is 57.8 s.
        var maxDurationSeconds: Double? {
            switch self {
            case .vln: return nil
            case .scan: return 60
            }
        }

        /// Stills rate for the multi-camera path, which is a different recorder
        /// with a different budget.
        ///
        /// Deliberately far below the ARKit path's 30. That session logs
        /// `system pressure cost 1.985` at 5 Hz before any encoding, its frames
        /// are 3840x2160 at a measured median of 1386 KB, and both lenses plus
        /// depth share one serial queue. 10 Hz is 25 MB/s; 30 Hz would be 47.
        var multiCamStillsHz: Double {
            switch self {
            case .vln: return 5
            case .scan: return 10
            }
        }

        /// Seconds to let auto-exposure settle before locking it, or `nil` to
        /// leave it running.
        ///
        /// Multi-camera only — ARKit exposes no exposure control at all, only
        /// `ARFrame.camera.exposureDuration` to read back. Locking matters for
        /// a scan because several fragments have to merge into one map, and
        /// brightness that drifts between walks is inherited by the merge.
        var lockExposureAfterSeconds: Double? {
            switch self {
            case .vln: return nil
            case .scan: return 3
            }
        }
    }

    /// This config with `preset`'s settings applied, keeping the fields a
    /// preset does not own.
    ///
    /// Written as "start from the preset and carry the rest across" rather than
    /// "overwrite the owned fields", so there is exactly one list and `matches`
    /// can be an equality test. A field added later and forgotten here becomes
    /// preset-owned by default, which reads as `custom` — the safe direction.
    func applying(_ preset: Preset) -> CaptureConfig {
        var result = preset.configuration
        // Operator and device configuration. A preset that moved these would be
        // deciding whether the phone protects itself from heat, or which
        // heading reference a walk was recorded against.
        result.useMagnetometerCorrection = useMagnetometerCorrection
        result.degradeOnThermalPressure = degradeOnThermalPressure
        result.motionHz = motionHz
        result.videoFPS = videoFPS
        result.videoBitrate = videoBitrate
        return result
    }

    func matches(_ preset: Preset) -> Bool {
        applying(preset) == self
    }

    /// The preset this config is, or `nil` if it has been hand-edited away from
    /// all of them.
    var activePreset: Preset? {
        Preset.allCases.first { matches($0) }
    }

    /// What the manifest records. `"custom"` is a real answer, not a failure:
    /// the rates alongside it in `config` still say exactly what was captured.
    var presetName: String {
        activePreset?.rawValue ?? "custom"
    }
}
