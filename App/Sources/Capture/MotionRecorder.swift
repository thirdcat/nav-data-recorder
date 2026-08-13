import CoreMotion
import Foundation

/// IMU capture via `CMDeviceMotion`.
///
/// Device motion rather than raw accelerometer/gyro: CoreMotion has already
/// separated gravity from user acceleration and removed gyro bias, which is
/// work a training pipeline would otherwise have to redo — badly, without
/// access to the same sensor fusion.
final class MotionRecorder {

    private let manager = CMMotionManager()
    private let opQueue = OperationQueue()
    private let queue: DispatchQueue

    /// Called on `queue`.
    var onMotion: ((MotionSample) -> Void)?
    var onEvent: ((String, String) -> Void)?

    private(set) var isRunning = false
    private(set) var referenceFrame: CMAttitudeReferenceFrame = .xArbitraryZVertical

    /// Both frames are gravity-aligned with an arbitrary yaw origin. The
    /// corrected variant additionally pulls yaw towards magnetic north, which
    /// is an improvement outdoors and actively harmful inside a building — see
    /// `CaptureConfig.useMagnetometerCorrection`.
    static func referenceFrame(magnetometerCorrected: Bool) -> CMAttitudeReferenceFrame {
        magnetometerCorrected ? .xArbitraryCorrectedZVertical : .xArbitraryZVertical
    }

    static func describe(_ frame: CMAttitudeReferenceFrame) -> String {
        frame == .xArbitraryCorrectedZVertical
            ? "xArbitraryCorrectedZVertical"
            : "xArbitraryZVertical"
    }

    init(queue: DispatchQueue) {
        self.queue = queue
        // Dedicated serial queue: CoreMotion delivers at 100 Hz and must not
        // land on main.
        opQueue.maxConcurrentOperationCount = 1
        opQueue.qualityOfService = .userInitiated
    }

    var isAvailable: Bool { manager.isDeviceMotionAvailable }

    func start(hz: Double, magnetometerCorrected: Bool) {
        guard !isRunning else { return }
        guard manager.isDeviceMotionAvailable else {
            queue.async { [weak self] in
                self?.onEvent?("motion.unavailable", "device motion not available")
            }
            return
        }
        isRunning = true
        referenceFrame = Self.referenceFrame(magnetometerCorrected: magnetometerCorrected)
        manager.deviceMotionUpdateInterval = 1.0 / max(1.0, hz)
        manager.startDeviceMotionUpdates(using: referenceFrame, to: opQueue) { [weak self] motion, error in
            guard let self = self else { return }
            if let error = error {
                self.queue.async { self.onEvent?("motion.error", error.localizedDescription) }
                return
            }
            guard let m = motion else { return }

            let sample = MotionSample(
                t: m.timestamp,
                ax: m.userAcceleration.x, ay: m.userAcceleration.y, az: m.userAcceleration.z,
                gx: m.gravity.x, gy: m.gravity.y, gz: m.gravity.z,
                rx: m.rotationRate.x, ry: m.rotationRate.y, rz: m.rotationRate.z,
                qx: m.attitude.quaternion.x, qy: m.attitude.quaternion.y,
                qz: m.attitude.quaternion.z, qw: m.attitude.quaternion.w,
                mx: m.magneticField.field.x, my: m.magneticField.field.y, mz: m.magneticField.field.z,
                // `CMMagneticFieldCalibrationAccuracy` is Int32-backed.
                magAcc: Int(m.magneticField.accuracy.rawValue))

            self.queue.async { self.onMotion?(sample) }
        }
    }

    func stop() {
        guard isRunning else { return }
        isRunning = false
        manager.stopDeviceMotionUpdates()
    }
}
