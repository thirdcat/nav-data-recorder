import AVFoundation
import CoreMotion
import Foundation
import UIKit

/// Reports what the camera and motion hardware actually offer, as opposed to
/// what a spec sheet or an API header implies.
///
/// This exists because the interesting questions about an alternative capture
/// path — can the LiDAR depth camera and the ultra-wide run in the same
/// multi-cam session? at what resolution? — are answered by device-specific
/// tables that Apple does not publish and that vary by chip. Guessing is how
/// you find out three weeks in that the combination you designed around is not
/// in `supportedMultiCamDeviceSets`.
enum CaptureCapabilities {

    static func report() -> String {
        var lines: [String] = []

        lines.append("# Device")
        lines.append("model            \(UIDevice.current.modelIdentifier)")
        lines.append("iOS              \(UIDevice.current.systemVersion)")
        lines.append("")

        lines.append("# ARKit")
        lines.append("world tracking   \(ARRecorderSupport.isSupported)")
        lines.append("scene depth      \(ARRecorderSupport.hasLiDAR)")
        lines.append("")

        lines.append("# Multi-camera")
        lines.append("supported        \(AVCaptureMultiCamSession.isMultiCamSupported)")
        lines.append("")

        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: backCameraTypes,
            mediaType: .video,
            position: .back)

        lines.append("# Back cameras")
        if discovery.devices.isEmpty {
            lines.append("(none discovered)")
        }
        for device in discovery.devices {
            lines.append(contentsOf: describe(device))
        }
        lines.append("")

        lines.append("# Supported multi-cam device sets")
        let sets = discovery.supportedMultiCamDeviceSets
        if sets.isEmpty {
            lines.append("(none — simultaneous capture is not available)")
        }
        for (index, set) in sets.enumerated() {
            let names = set.map { shortName($0.deviceType) }.sorted().joined(separator: " + ")
            lines.append("\(index).  \(names)")
        }
        let lidarAndUltraWide = sets.contains { set in
            let types = set.map { $0.deviceType }
            return types.contains(.builtInLiDARDepthCamera)
                && types.contains(.builtInUltraWideCamera)
        }
        lines.append("LiDAR + ultra-wide \(lidarAndUltraWide ? "supported" : "not supported")")
        lines.append("")

        lines.append(contentsOf: depthPairings(discovery.devices))

        lines.append("# Motion")
        let motion = CMMotionManager()
        lines.append("device motion    \(motion.isDeviceMotionAvailable)")
        lines.append("accelerometer    \(motion.isAccelerometerAvailable)")
        lines.append("gyroscope        \(motion.isGyroAvailable)")
        lines.append("magnetometer     \(motion.isMagnetometerAvailable)")
        lines.append("barometer        \(CMAltimeter.isRelativeAltitudeAvailable())")
        lines.append("")
        lines.append("confidence map   no per-pixel map on the AVFoundation path. With "
                     + "AVCaptureDepthDataOutput.isFilteringEnabled = false the LiDAR "
                     + "camera drops low-confidence points instead, so absence of a "
                     + "value is the confidence signal. Filtering defaults to ON and "
                     + "fills holes with invented depth — see MultiCamDepthProbe.")

        return lines.joined(separator: "\n")
    }

    /// Which video format carries which depth, rather than the two lists apart.
    ///
    /// The existing per-device summary says "33 video formats carry depth" and
    /// separately that depth comes at 320x240. It does not say whether those are
    /// the *same* formats — and the question that decides the multi-camera plan
    /// is exactly that: can a usable colour resolution and the largest depth be
    /// active at once, or does depth only ride along with formats nobody wants?
    ///
    /// The field of view is printed per format because it is not a property of
    /// the lens. A format that crops the sensor sees less than the lens does:
    /// this device reports 74.6° for the wide camera while ARKit's 1920x1440
    /// format actually delivers 70.6°, and the second number is the one that
    /// bounds what a capture can cover.
    private static func depthPairings(_ devices: [AVCaptureDevice]) -> [String] {
        var lines: [String] = ["# Multi-cam video formats that carry depth"]
        var any = false

        for device in devices {
            let formats = device.formats.filter {
                $0.isMultiCamSupported && !$0.supportedDepthDataFormats.isEmpty
            }
            if formats.isEmpty { continue }
            any = true
            lines.append("## \(shortName(device.deviceType))")

            // Several raw formats differ only in pixel encoding or binning and
            // are identical for this purpose, so collapse them and say how many
            // collapsed. Sorted by pixel count so the largest is first, which is
            // the one a capture would want.
            var groups: [String: (count: Int, depths: Set<String>, pixels: Int)] = [:]
            for format in formats {
                let d = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
                let fps = format.videoSupportedFrameRateRanges
                    .map { Int($0.maxFrameRate) }.max() ?? 0
                let ratio = aspect(width: Int(d.width), height: Int(d.height))
                let size = pad("\(d.width)x\(d.height)@\(fps)", to: 16)
                let key = "\(size)\(pad(ratio, to: 6))fov "
                    + String(format: "%.1f", format.videoFieldOfView) + "°"
                let sizes = Set(format.supportedDepthDataFormats.map { depthSize($0) })
                var entry = groups[key] ?? (0, [], Int(d.width) * Int(d.height))
                entry.count += 1
                entry.depths.formUnion(sizes)
                groups[key] = entry
            }

            for key in groups.keys.sorted(by: { (groups[$0]?.pixels ?? 0) > (groups[$1]?.pixels ?? 0) }) {
                guard let entry = groups[key] else { continue }
                let depths = entry.depths.sorted().joined(separator: ", ")
                let dupes = entry.count > 1 ? "  (\(entry.count) encodings)" : ""
                lines.append("   \(key)  depth \(depths)\(dupes)")
            }
        }

        if !any {
            lines.append("(no multi-cam format on any back camera carries depth)")
        }
        lines.append("")
        return lines
    }

    private static func pad(_ text: String, to width: Int) -> String {
        text.count >= width ? text + " "
            : text + String(repeating: " ", count: width - text.count)
    }

    private static func aspect(width: Int, height: Int) -> String {
        guard height > 0 else { return "?" }
        let r = Double(width) / Double(height)
        if abs(r - 4.0 / 3.0) < 0.02 { return "4:3" }
        if abs(r - 16.0 / 9.0) < 0.02 { return "16:9" }
        return String(format: "%.2f", r)
    }

    /// Depth dimensions only, without the frame rate the video format already
    /// states — the pairing is what this line is for.
    private static func depthSize(_ format: AVCaptureDevice.Format) -> String {
        let d = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
        return "\(d.width)x\(d.height) \(mediaSubtype(format))"
    }

    private static var backCameraTypes: [AVCaptureDevice.DeviceType] {
        var types: [AVCaptureDevice.DeviceType] = [
            .builtInWideAngleCamera,
            .builtInUltraWideCamera,
            .builtInTelephotoCamera,
            .builtInDualCamera,
            .builtInDualWideCamera,
            .builtInTripleCamera
        ]
        // The LiDAR depth camera is a virtual device pairing the wide camera
        // with the scanner. Its presence is what decides whether depth survives
        // a switch away from ARKit.
        types.append(.builtInLiDARDepthCamera)
        return types
    }

    private static func describe(_ device: AVCaptureDevice) -> [String] {
        var lines: [String] = []
        lines.append("## \(shortName(device.deviceType))")

        // Focal length in 35mm terms is not exposed, so report the field of
        // view directly — it is the number that actually matters here.
        let fovs = device.formats.map { $0.videoFieldOfView }
        if let maxFOV = fovs.max() {
            lines.append("   fov (h)       \(String(format: "%.1f", maxFOV))°")
        }

        let multiCamFormats = device.formats.filter { $0.isMultiCamSupported }
        lines.append("   formats       \(device.formats.count) total, \(multiCamFormats.count) multi-cam capable")

        if let best = largest(device.formats) {
            lines.append("   max single    \(dimensions(best))")
        }
        if let best = largest(multiCamFormats) {
            lines.append("   max multi-cam \(dimensions(best))")
        } else if !device.formats.isEmpty {
            lines.append("   max multi-cam (none — cannot participate in multi-cam)")
        }

        // Report both sets. The second one is the question this probe exists to
        // answer: what depth survives when this video device is used in a
        // simultaneous multi-camera session?
        let withDepth = device.formats.filter { !$0.supportedDepthDataFormats.isEmpty }
        lines.append(contentsOf: depthReport("depth", videoFormats: withDepth))
        let multiCamWithDepth = multiCamFormats.filter {
            !$0.supportedDepthDataFormats.isEmpty
        }
        lines.append(contentsOf: depthReport("depth multi-cam",
                                              videoFormats: multiCamWithDepth))

        lines.append("   focus lock    \(device.isFocusModeSupported(.locked))")
        lines.append("   exposure lock \(device.isExposureModeSupported(.locked))")
        return lines
    }

    private static func largest(_ formats: [AVCaptureDevice.Format]) -> AVCaptureDevice.Format? {
        formats.max { a, b in
            let da = CMVideoFormatDescriptionGetDimensions(a.formatDescription)
            let db = CMVideoFormatDescriptionGetDimensions(b.formatDescription)
            return Int(da.width) * Int(da.height) < Int(db.width) * Int(db.height)
        }
    }

    private static func dimensions(_ format: AVCaptureDevice.Format) -> String {
        let d = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
        let fps = format.videoSupportedFrameRateRanges.map { Int($0.maxFrameRate) }.max() ?? 0
        return "\(d.width)x\(d.height)@\(fps)"
    }

    private static func depthDimensions(_ format: AVCaptureDevice.Format) -> String {
        let d = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
        let fps = format.videoSupportedFrameRateRanges
            .map { Int($0.maxFrameRate) }.max() ?? 0
        return "\(d.width)x\(d.height)@\(fps) \(mediaSubtype(format))"
    }

    private static func depthReport(_ label: String,
                                    videoFormats: [AVCaptureDevice.Format]) -> [String] {
        let depthFormats = videoFormats.flatMap { $0.supportedDepthDataFormats }
        let sizes = Set(depthFormats.map { depthDimensions($0) })
        return [
            "   \(label) formats \(videoFormats.count) video formats carry depth",
            "   \(label) sizes   \(sizes.isEmpty ? "(none)" : sizes.sorted().joined(separator: ", "))"
        ]
    }

    private static func mediaSubtype(_ format: AVCaptureDevice.Format) -> String {
        let subtype = CMFormatDescriptionGetMediaSubType(format.formatDescription)
        let bytes: [UInt8] = [
            UInt8((subtype >> 24) & 0xff),
            UInt8((subtype >> 16) & 0xff),
            UInt8((subtype >> 8) & 0xff),
            UInt8(subtype & 0xff)
        ]
        let code = String(bytes: bytes, encoding: .ascii)
            ?? String(format: "0x%08x", subtype)
        switch code {
        case "hdep": return "hdep(float16-depth)"
        case "fdep": return "fdep(float32-depth)"
        case "hdis": return "hdis(float16-disparity)"
        case "fdis": return "fdis(float32-disparity)"
        default: return code
        }
    }

    private static func shortName(_ type: AVCaptureDevice.DeviceType) -> String {
        type.rawValue.replacingOccurrences(of: "AVCaptureDeviceTypeBuiltIn", with: "")
    }
}

/// Indirection so this file does not have to import ARKit just to ask two
/// questions, and so the report still builds if the AR path is ever removed.
enum ARRecorderSupport {
    static var isSupported: Bool { ARRecorder.isSupported }
    static var hasLiDAR: Bool { ARRecorder.hasLiDAR }
}
