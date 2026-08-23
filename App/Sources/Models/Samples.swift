import Foundation
import simd

/// Every `t` below is seconds in the master monotonic clock (see `Clock`).
/// Add `SessionManifest.clockAnchor` to get Unix time.

// MARK: - Location

struct LocationSample: Codable {
    let t: Double
    /// Unix time as reported by CoreLocation, kept verbatim. Useful as a
    /// cross-check on `clockAnchor` and for correlating with external sources.
    let wall: Double
    let lat: Double
    let lon: Double
    /// Mean-sea-level altitude, metres.
    let alt: Double
    /// WGS-84 ellipsoidal altitude, metres. This is the one to use when fusing
    /// with anything geodetic.
    let altEllipsoidal: Double
    /// Horizontal / vertical accuracy, metres. Negative means invalid.
    let hAcc: Double
    let vAcc: Double
    /// Metres per second. Negative means invalid.
    let speed: Double
    let speedAcc: Double
    /// Degrees clockwise from true north. Negative means invalid.
    let course: Double
    let courseAcc: Double
}

struct HeadingSample: Codable {
    let t: Double
    let trueHeading: Double
    let magneticHeading: Double
    /// Degrees. Negative means the reading is invalid.
    let accuracy: Double
}

// MARK: - Motion

/// One fused `CMDeviceMotion` reading. Gravity is already separated out of the
/// acceleration, and attitude is delivered as a quaternion to avoid the gimbal
/// discontinuities Euler angles would introduce in training data.
struct MotionSample: Codable {
    let t: Double
    /// User acceleration, in G, device frame.
    let ax: Double
    let ay: Double
    let az: Double
    /// Gravity vector, in G, device frame.
    let gx: Double
    let gy: Double
    let gz: Double
    /// Rotation rate, rad/s, bias-corrected.
    let rx: Double
    let ry: Double
    let rz: Double
    /// Attitude quaternion relative to the reference frame recorded in the manifest.
    let qx: Double
    let qy: Double
    let qz: Double
    let qw: Double
    /// Calibrated magnetic field, microtesla.
    let mx: Double
    let my: Double
    let mz: Double
    /// `CMMagneticFieldCalibrationAccuracy` raw value; -1 is uncalibrated.
    let magAcc: Int
}

// MARK: - ARKit

/// Camera pose for one `ARFrame`, in ARKit's gravity-aligned world frame.
///
/// The world origin is wherever the session started, so poses are only
/// meaningful *within* a session — they are a local odometry track, not a
/// geographic one. Fuse with `LocationSample` to georeference.
struct PoseSample: Codable {
    let t: Double
    /// Index into the video track. Matches the Nth appended frame in video.mov.
    let frame: Int
    /// Camera position in world coordinates, metres.
    let tx: Float
    let ty: Float
    let tz: Float
    /// Camera orientation, world-from-camera.
    let qx: Float
    let qy: Float
    let qz: Float
    let qw: Float
    /// Pinhole intrinsics for `capturedImage`, pixels.
    let fx: Float
    let fy: Float
    let cx: Float
    let cy: Float
    /// Raw value of `ARCamera.TrackingState`: "normal", "limited:<reason>", "notAvailable".
    let tracking: String
    /// Exposure duration in seconds, useful for rejecting motion-blurred frames.
    let exposure: Double
    /// Sensor gain at this frame, as ISO.
    ///
    /// **Why duration alone was not enough.** `docs/VLIO.md` tried to normalise
    /// frame brightness by dividing out the logged exposure and found that a
    /// full stop of logged change shows up in the image as 0.04 stops — the
    /// pipeline had already compensated with gain, which nothing recorded, so
    /// dividing by duration *injected* the factor of two it meant to remove and
    /// inflated the warped residual by up to 20x. That page asks for this field
    /// by name.
    ///
    /// `nil` where the camera device could not be reached — see
    /// `ARRecorder.configureCaptureDevice` — and on every session recorded
    /// before this build, where the key is simply absent.
    let iso: Float?
    /// Unit gravity vector expressed in **camera** coordinates (+X right in the
    /// image, +Y up, -Z forward).
    ///
    /// This is what tells a consumer how the phone was held. ARKit always hands
    /// back `capturedImage` in the camera's native landscape orientation, so a
    /// portrait-held recording produces files that are structurally identical
    /// to a landscape one — same 1920x1440 buffer — with the world rotated 90
    /// degrees inside it. Nothing in the image or its dimensions distinguishes
    /// the two.
    ///
    /// `atan2(gravX, -gravY)` gives the in-image roll needed to make the frame
    /// upright, and it handles the phone being held at any angle rather than
    /// snapping to four discrete orientations.
    let gravX: Float
    let gravY: Float
    let gravZ: Float
}

/// Index entry for one depth map inside `depth.bin`.
///
/// Depth is bulky enough that it gets a raw sidecar file rather than JSON: at
/// 256x192 float16 a frame is ~96 KB, so 30 Hz would be ~10 GB/hour. The
/// capture rate is throttled separately from video (see `CaptureConfig`).
struct DepthIndexEntry: Codable {
    let t: Double
    let frame: Int
    /// Byte offset into depth.bin.
    let offset: UInt64
    /// Byte length of this frame's depth payload.
    let length: Int
    let width: Int
    let height: Int
    /// Row-major float16 metres. Non-finite entries mean "no return".
    let format: String
    /// Byte offset into confidence.bin, or nil when confidence capture is off.
    let confidenceOffset: UInt64?
    /// One byte per pixel, `ARConfidenceLevel` raw value (0 low … 2 high).
    let confidenceLength: Int?
    /// Smallest/largest eigenvalue ratio of the current frame-only Hessian.
    /// Unlike ICP's conditioning, this uses no frame correspondences.
    let cond: Double?
    /// Weakest constrained 6-DoF axis in the depth frame (+Z forward, +Y down),
    /// not the ARKit camera frame (−Z forward, +Y up).
    let weakAxis: [Double]?
    /// Number of 4x4 depth-grid samples included in the frame-only Hessian.
    let conditioningSamples: Int
}

/// One observation of an ARKit plane anchor — floor, wall, ceiling, table.
///
/// Indoors these are the structural skeleton of the scene, and ARKit derives
/// them from the LiDAR return rather than guessing from imagery. Planes grow
/// and merge as a room is explored, so a single plane produces many rows over a
/// session; take the last `updated` row for a given `id` to get its final
/// extent, or replay them in order to see what was known at any moment.
struct PlaneSample: Codable {
    let t: Double
    /// Stable across a session. Two planes merging shows up as one `removed`.
    let id: String
    /// "added", "updated" or "removed".
    let event: String
    /// "horizontal" or "vertical".
    let alignment: String
    /// ARKit's semantic guess: floor, wall, ceiling, table, seat, door, window,
    /// or "none:<reason>" when it declines to classify.
    let classification: String
    /// Anchor pose in the session world frame.
    let tx: Float
    let ty: Float
    let tz: Float
    let qx: Float
    let qy: Float
    let qz: Float
    let qw: Float
    /// Plane centre, in the anchor's local frame.
    let cx: Float
    let cy: Float
    let cz: Float
    /// Extent in metres, and the plane's rotation about its own Y axis.
    let width: Float
    let height: Float
    let rotationOnYAxis: Float
}

/// Index row for one written JPEG in stills mode.
///
/// `frame` matches `PoseSample.frame`, so an image and its pose are joined by
/// that key alone — no timestamp search, no interpolation.
struct FrameIndexEntry: Codable {
    let t: Double
    let frame: Int
    /// Path relative to the session directory, e.g. `frames/000123.jpg`.
    let file: String
    let width: Int
    let height: Int
    let bytes: Int
}

// MARK: - Events

/// Anything that changes how the rest of the stream should be interpreted:
/// thermal throttling, backgrounding, tracking loss, permission changes.
struct SessionEvent: Codable {
    let t: Double
    let kind: String
    let detail: String
}

// MARK: - simd helpers

extension simd_float4x4 {
    var translation: SIMD3<Float> {
        SIMD3(columns.3.x, columns.3.y, columns.3.z)
    }

    /// Gravity as a unit vector in the camera's own frame, for a
    /// `world_from_camera` transform in ARKit's gravity-aligned world (+Y up).
    ///
    /// Expressing the world's down vector (0, -1, 0) in camera coordinates is
    /// `Rᵀ · v`, and for a rotation matrix that is just the dot product with
    /// each column — so this reduces to negating the Y component of the three
    /// basis columns. No matrix inverse required.
    var gravityInCameraFrame: SIMD3<Float> {
        SIMD3(-columns.0.y, -columns.1.y, -columns.2.y)
    }

    /// Assumes a rigid transform (no scale), which holds for `ARCamera.transform`.
    var rotationQuaternion: simd_quatf {
        let m = simd_float3x3(
            SIMD3(columns.0.x, columns.0.y, columns.0.z),
            SIMD3(columns.1.x, columns.1.y, columns.1.z),
            SIMD3(columns.2.x, columns.2.y, columns.2.z)
        )
        return simd_quatf(m)
    }
}
