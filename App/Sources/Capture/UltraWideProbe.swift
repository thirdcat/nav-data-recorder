import AVFoundation
import Foundation
import UIKit

/// Captures ultra-wide stills together with the lens calibration needed to
/// rectify them, so the ultra-wide question can be answered with images rather
/// than argued from spec sheets.
///
/// This is deliberately *not* part of the recording path. An `AVCaptureSession`
/// cannot coexist with the `ARSession` the recorder is built on, so this runs
/// alone, produces no poses, and exists only to establish whether a rectified
/// ultra-wide frame is good enough to be worth the rest of the work.
///
/// The calibration is the whole point. A 106° lens is not a pinhole, and
/// `AVCameraCalibrationData` is where Apple puts the correction — as a radial
/// magnification lookup table, which `tools/rectify_ultrawide.py` consumes.
/// Getting it is the awkward part: it is delivered only alongside constituent
/// photos from a virtual device (or with depth), never from the bare ultra-wide
/// on its own. So the probe drives the triple/dual-wide camera and asks for its
/// ultra-wide and wide constituents, which has the side benefit of capturing
/// exactly the synchronised pair a dual-lens capture app would.
final class UltraWideProbe: NSObject {

    enum ProbeError: LocalizedError {
        case noCamera
        case cannotConfigure(String)

        var errorDescription: String? {
            switch self {
            case .noCamera:
                return "No back camera with an ultra-wide constituent."
            case .cannotConfigure(let why):
                return why
            }
        }
    }

    /// Progress and results, always delivered on the main queue.
    var onStatus: ((String) -> Void)?
    var onFinished: ((Result<URL, Error>) -> Void)?

    private let session = AVCaptureSession()
    private let output = AVCapturePhotoOutput()
    private let queue = DispatchQueue(label: "uwprobe.session")

    private var device: AVCaptureDevice?
    private var constituents: [AVCaptureDevice] = []
    private var calibrationAvailable = false

    private var directory: URL?
    private var shotIndex = 0
    /// One entry per lens, keyed by device type; written out at the end.
    ///
    /// Everything in this group is owned by `queue`. Photo delegate callbacks
    /// arrive on AVFoundation's own serial queue, which is serial with respect
    /// to itself but not to `finish()` — so they hop onto `queue` before
    /// touching any of it.
    private var calibrations: [String: [String: Any]] = [:]
    private var files: [String: [String]] = [:]
    private var notes: [String] = []

    // MARK: - Lifecycle

    /// Picks the device and wires up the session. Safe to call once.
    func start(completion: @escaping (Result<Void, Error>) -> Void) {
        queue.async { [weak self] in
            guard let self = self else { return }
            do {
                try self.configure()
                self.session.startRunning()
                // The capability lines are the first thing worth knowing — if
                // calibration delivery is unsupported the rest of the run
                // cannot answer anything — so put them on screen rather than
                // only in the file that delivery would have produced.
                let notes = self.notes
                DispatchQueue.main.async {
                    notes.forEach { self.onStatus?($0) }
                    completion(.success(()))
                }
            } catch {
                DispatchQueue.main.async { completion(.failure(error)) }
            }
        }
    }

    // MARK: - Configuration

    private func configure() throws {
        // A virtual device is required: calibration data rides along with
        // constituent photo delivery, and the standalone ultra-wide has no
        // constituents to deliver.
        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInTripleCamera, .builtInDualWideCamera, .builtInUltraWideCamera],
            mediaType: .video,
            position: .back)

        guard let chosen = discovery.devices.first(where: {
            $0.constituentDevices.contains { $0.deviceType == .builtInUltraWideCamera }
        }) ?? discovery.devices.first(where: { $0.deviceType == .builtInUltraWideCamera }) else {
            throw ProbeError.noCamera
        }
        device = chosen
        notes.append("device \(Self.shortName(chosen.deviceType))")

        session.beginConfiguration()
        // .photo, not a video preset: the calibration is quoted against the
        // photo dimensions, and stills are what a rectification check wants.
        session.sessionPreset = .photo

        let input = try AVCaptureDeviceInput(device: chosen)
        guard session.canAddInput(input) else {
            session.commitConfiguration()
            throw ProbeError.cannotConfigure("Cannot add the camera input.")
        }
        session.addInput(input)

        guard session.canAddOutput(output) else {
            session.commitConfiguration()
            throw ProbeError.cannotConfigure("Cannot add the photo output.")
        }
        session.addOutput(output)

        // Order matters: constituent delivery has to be enabled before
        // calibration delivery will report itself as supported.
        let wanted = chosen.constituentDevices.filter {
            $0.deviceType == .builtInUltraWideCamera || $0.deviceType == .builtInWideAngleCamera
        }
        if output.isVirtualDeviceConstituentPhotoDeliverySupported, wanted.count >= 2 {
            output.isVirtualDeviceConstituentPhotoDeliveryEnabled = true
            constituents = wanted
            notes.append("constituents \(wanted.map { Self.shortName($0.deviceType) }.joined(separator: " + "))")
        } else {
            notes.append("constituent delivery unsupported — single lens only")
        }

        calibrationAvailable = output.isCameraCalibrationDataDeliverySupported
        notes.append("calibration delivery \(calibrationAvailable ? "supported" : "UNSUPPORTED")")

        session.commitConfiguration()

        // Fixed focus, so every shot in the probe shares one focal length. A
        // rectification checked against a lens that refocused between frames
        // would be checking two lenses.
        if chosen.isFocusModeSupported(.locked) {
            try? chosen.lockForConfiguration()
            chosen.focusMode = .locked
            chosen.unlockForConfiguration()
            notes.append("focus locked")
        }

        let dir = SessionStore.documents
            .appendingPathComponent("uw_probe", isDirectory: true)
            .appendingPathComponent(SessionStore.makeSessionID(), isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        directory = dir
    }

    // MARK: - Capture

    /// Takes one shot. Each shot yields one photo per enabled constituent.
    func capture() {
        queue.async { [weak self] in
            guard let self = self, self.session.isRunning else { return }
            let settings = AVCapturePhotoSettings(format: [
                AVVideoCodecKey: AVVideoCodecType.jpeg
            ])
            if self.output.isVirtualDeviceConstituentPhotoDeliveryEnabled {
                settings.virtualDeviceConstituentPhotoDeliveryEnabledDevices = self.constituents
            }
            if self.calibrationAvailable {
                settings.isCameraCalibrationDataDeliveryEnabled = true
            }
            self.output.capturePhoto(with: settings, delegate: self)
        }
    }

    /// Writes `calibration.json` for every lens seen, and finishes.
    func finish() {
        queue.async { [weak self] in
            guard let self = self, let dir = self.directory else { return }
            do {
                try self.writeMetadata()
                self.session.stopRunning()
                DispatchQueue.main.async { self.onFinished?(.success(dir)) }
            } catch {
                DispatchQueue.main.async { self.onFinished?(.failure(error)) }
            }
        }
    }

    /// Rewrites `calibration.json` for every lens seen so far.
    ///
    /// Called after every photo, not only at the end. The first version of this
    /// only wrote at `finish()`, so leaving the screen — or a crash, or a
    /// force-quit — left a directory full of JPEGs with no calibration and
    /// nothing to say why. A handful of shots makes this a few hundred bytes of
    /// rewriting, which is not worth being clever about.
    ///
    /// Must be called on `queue`.
    private func writeMetadata() throws {
        guard let dir = directory else { return }
        for (lens, names) in files {
            let out = dir.appendingPathComponent(lens, isDirectory: true)
                .appendingPathComponent("calibration.json")
            guard var payload = calibrations[lens] else {
                // No calibration for this lens: still say so on disk, so the
                // directory is not silently un-rectifiable.
                try write(["lens": lens,
                           "images": names.sorted(),
                           "notes": notes,
                           "calibration": "unavailable"], to: out)
                continue
            }
            payload["images"] = names.sorted()
            payload["notes"] = notes
            try write(payload, to: out)
        }
    }

    private func write(_ object: [String: Any], to url: URL) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        let data = try JSONSerialization.data(withJSONObject: object,
                                              options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url, options: .atomic)
    }

    // MARK: - Calibration serialisation

    /// The exact payload `tools/rectify_ultrawide.py` reads.
    ///
    /// Named scalars rather than raw matrices: `matrix_float3x3` is
    /// column-major, and a JSON array of arrays gives a reader no way to tell
    /// which convention was meant. Being explicit here removes a whole class of
    /// silent transposition bug on the other side.
    private static func encode(_ c: AVCameraCalibrationData) -> [String: Any] {
        let k = c.intrinsicMatrix
        let ref = c.intrinsicMatrixReferenceDimensions
        var payload: [String: Any] = [
            "reference_dimensions": [Int(ref.width), Int(ref.height)],
            "intrinsics": [
                "fx": Double(k.columns.0.x),
                "fy": Double(k.columns.1.y),
                "cx": Double(k.columns.2.x),
                "cy": Double(k.columns.2.y),
            ],
            "lens_distortion_center": [Double(c.lensDistortionCenter.x),
                                       Double(c.lensDistortionCenter.y)],
            "pixel_size_mm": Double(c.pixelSize),
        ]
        if let table = c.lensDistortionLookupTable {
            payload["lens_distortion_lookup_table"] = floats(table)
        }
        if let table = c.inverseLensDistortionLookupTable {
            payload["inverse_lens_distortion_lookup_table"] = floats(table)
        }
        // Rotation and translation of this lens relative to the reference
        // camera. Not needed to rectify one frame, but it is what would carry a
        // pose from one lens to the other, so it is worth having on disk.
        let e = c.extrinsicMatrix
        payload["extrinsic_matrix_columns"] = [
            [Double(e.columns.0.x), Double(e.columns.0.y), Double(e.columns.0.z)],
            [Double(e.columns.1.x), Double(e.columns.1.y), Double(e.columns.1.z)],
            [Double(e.columns.2.x), Double(e.columns.2.y), Double(e.columns.2.z)],
            [Double(e.columns.3.x), Double(e.columns.3.y), Double(e.columns.3.z)],
        ]
        return payload
    }

    private static func floats(_ data: Data) -> [Double] {
        data.withUnsafeBytes { raw in
            raw.bindMemory(to: Float32.self).map { Double($0) }
        }
    }

    static func shortName(_ type: AVCaptureDevice.DeviceType) -> String {
        type.rawValue.replacingOccurrences(of: "AVCaptureDeviceTypeBuiltIn", with: "")
    }
}

// MARK: - AVCapturePhotoCaptureDelegate

extension UltraWideProbe: AVCapturePhotoCaptureDelegate {

    func photoOutput(_ output: AVCapturePhotoOutput,
                     didFinishProcessingPhoto photo: AVCapturePhoto,
                     error: Error?) {
        if let error = error {
            DispatchQueue.main.async { self.onStatus?("shot failed: \(error.localizedDescription)") }
            return
        }
        guard let data = photo.fileDataRepresentation() else { return }
        // Which lens this constituent came from. Without virtual-device
        // delivery there is one photo per shot and no source type, so fall back
        // to the device that was opened.
        let sourceType = photo.sourceDeviceType
        let calibration = photo.cameraCalibrationData

        queue.async { [weak self] in
            guard let self = self, let dir = self.directory else { return }
            let lens = sourceType.map(Self.shortName)
                ?? self.device.map { Self.shortName($0.deviceType) }
                ?? "unknown"

            let name = String(format: "%04d.jpg", self.shotIndex)
            let lensDir = dir.appendingPathComponent(lens, isDirectory: true)
            do {
                try FileManager.default.createDirectory(at: lensDir,
                                                        withIntermediateDirectories: true)
                try data.write(to: lensDir.appendingPathComponent(name), options: .atomic)
                self.files[lens, default: []].append(name)
            } catch {
                DispatchQueue.main.async {
                    self.onStatus?("write failed: \(error.localizedDescription)")
                }
                return
            }

            // Calibration is per lens, not per shot — with focus locked it does
            // not change between shots, so the first one that arrives is kept
            // and the rest are redundant.
            if self.calibrations[lens] == nil, let calibration = calibration {
                self.calibrations[lens] = Self.encode(calibration)
            }

            let haveCalibration = self.calibrations[lens] != nil
            // Persist now rather than at finish, so every shot leaves the
            // directory complete on its own.
            do {
                try self.writeMetadata()
            } catch {
                DispatchQueue.main.async {
                    self.onStatus?("calibration.json failed: \(error.localizedDescription)")
                }
            }
            DispatchQueue.main.async {
                self.onStatus?("\(lens) \(name)\(haveCalibration ? "" : " — NO CALIBRATION")")
            }
        }
    }

    func photoOutput(_ output: AVCapturePhotoOutput,
                     didFinishCaptureFor resolvedSettings: AVCaptureResolvedPhotoSettings,
                     error: Error?) {
        // Fires once per shot, after every constituent for that shot has been
        // delivered — so both lenses share one index and the pair stays
        // matched by filename.
        queue.async { [weak self] in self?.shotIndex += 1 }
    }
}
