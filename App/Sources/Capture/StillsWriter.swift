import CoreImage
import CoreVideo
import Foundation

/// Writes individual JPEGs, one per captured `ARFrame`.
///
/// The alternative — encode HEVC, extract frames later — costs a lossy
/// generation and a seek-and-decode step between the recorder and the
/// dataloader. For posed-image training data that trade is the wrong way round,
/// so stills are the default and video is the option.
///
/// Encoding runs on its own queue rather than ARKit's. `ARFrame` pixel buffers
/// come from a small pool and holding one starves capture, so the buffer is
/// deep-copied on the AR queue — about a millisecond — and the copy is what
/// gets encoded. Doing the ~20 ms JPEG encode inline would stall ARKit's
/// delegate and cost pose samples, which are the expensive thing to lose.
final class StillsWriter {

    /// One written image, reported back for the frames index.
    struct Entry {
        let t: Double
        let frame: Int
        let file: String
        let width: Int
        let height: Int
        let bytes: Int
    }

    private let directory: URL
    private let relativePrefix: String
    private let quality: Double
    private let encodeQueue = DispatchQueue(label: "nav.stills", qos: .utility)
    private let context = CIContext(options: [.useSoftwareRenderer: false])

    /// Called on the encode queue for each image successfully written.
    var onWrite: ((Entry) -> Void)?
    var onError: ((String) -> Void)?

    private let lock = NSLock()
    private var _written = 0
    private var _failed = 0
    /// Frames handed to the encoder but not yet finished. Bounded so a slow
    /// disk cannot let the backlog grow without limit.
    private var _pending = 0
    private static let maxPending = 8

    init(directory: URL, relativePrefix: String, quality: Double) {
        self.directory = directory
        self.relativePrefix = relativePrefix
        self.quality = min(max(quality, 0.1), 1.0)
    }

    var written: Int {
        lock.lock(); defer { lock.unlock() }
        return _written
    }

    var failed: Int {
        lock.lock(); defer { lock.unlock() }
        return _failed
    }

    func prepare() throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    /// Call on the AR queue. Copies the buffer and returns immediately.
    /// Returns false if the frame was dropped because the encoder is behind.
    @discardableResult
    func capture(_ pixelBuffer: CVPixelBuffer, t: Double, frame: Int) -> Bool {
        lock.lock()
        if _pending >= Self.maxPending {
            _failed += 1
            lock.unlock()
            return false
        }
        _pending += 1
        lock.unlock()

        guard let copy = Self.copy(pixelBuffer) else {
            lock.lock(); _pending -= 1; _failed += 1; lock.unlock()
            onError?("could not copy pixel buffer for frame \(frame)")
            return false
        }

        encodeQueue.async { [weak self] in
            guard let self = self else { return }
            defer {
                self.lock.lock()
                self._pending -= 1
                self.lock.unlock()
            }
            self.encode(copy, t: t, frame: frame)
        }
        return true
    }

    /// Blocks until every queued frame has been written. Called on stop so the
    /// session is not marked complete with encodes still in flight.
    func finish() {
        encodeQueue.sync {}
    }

    private func encode(_ pixelBuffer: CVPixelBuffer, t: Double, frame: Int) {
        let image = CIImage(cvPixelBuffer: pixelBuffer)
        let name = String(format: "%06d.jpg", frame)
        let url = directory.appendingPathComponent(name)

        // sRGB explicitly: ARKit's buffers carry a video colour space, and
        // letting it default means images that look subtly different depending
        // on the device that recorded them.
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
            recordFailure("no sRGB colour space")
            return
        }

        let options: [CIImageRepresentationOption: Any] = [
            CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String): quality
        ]

        guard let data = context.jpegRepresentation(of: image,
                                                    colorSpace: colorSpace,
                                                    options: options) else {
            recordFailure("JPEG encode failed for frame \(frame)")
            return
        }

        do {
            try data.write(to: url, options: .atomic)
        } catch {
            recordFailure("could not write frame \(frame): \(error.localizedDescription)")
            return
        }

        lock.lock()
        _written += 1
        lock.unlock()

        onWrite?(Entry(
            t: t,
            frame: frame,
            file: "\(relativePrefix)/\(name)",
            width: CVPixelBufferGetWidth(pixelBuffer),
            height: CVPixelBufferGetHeight(pixelBuffer),
            bytes: data.count))
    }

    private func recordFailure(_ message: String) {
        lock.lock()
        _failed += 1
        lock.unlock()
        onError?(message)
    }

    /// Deep-copies a pixel buffer, plane by plane, respecting each plane's own
    /// row stride — the source and destination are not guaranteed to pad rows
    /// the same way, so a single flat memcpy would shear the image.
    static func copy(_ src: CVPixelBuffer) -> CVPixelBuffer? {
        let width = CVPixelBufferGetWidth(src)
        let height = CVPixelBufferGetHeight(src)
        let format = CVPixelBufferGetPixelFormatType(src)

        var dst: CVPixelBuffer?
        let attributes: [CFString: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary
        ]
        guard CVPixelBufferCreate(kCFAllocatorDefault, width, height, format,
                                  attributes as CFDictionary, &dst) == kCVReturnSuccess,
              let destination = dst else {
            return nil
        }

        CVPixelBufferLockBaseAddress(src, .readOnly)
        CVPixelBufferLockBaseAddress(destination, [])
        defer {
            CVPixelBufferUnlockBaseAddress(destination, [])
            CVPixelBufferUnlockBaseAddress(src, .readOnly)
        }

        let planeCount = CVPixelBufferGetPlaneCount(src)
        if planeCount == 0 {
            guard let s = CVPixelBufferGetBaseAddress(src),
                  let d = CVPixelBufferGetBaseAddress(destination) else { return nil }
            let srcStride = CVPixelBufferGetBytesPerRow(src)
            let dstStride = CVPixelBufferGetBytesPerRow(destination)
            let rowBytes = min(srcStride, dstStride)
            for y in 0..<height {
                memcpy(d.advanced(by: y * dstStride), s.advanced(by: y * srcStride), rowBytes)
            }
        } else {
            for plane in 0..<planeCount {
                guard let s = CVPixelBufferGetBaseAddressOfPlane(src, plane),
                      let d = CVPixelBufferGetBaseAddressOfPlane(destination, plane) else { return nil }
                let planeHeight = CVPixelBufferGetHeightOfPlane(src, plane)
                let srcStride = CVPixelBufferGetBytesPerRowOfPlane(src, plane)
                let dstStride = CVPixelBufferGetBytesPerRowOfPlane(destination, plane)
                let rowBytes = min(srcStride, dstStride)
                for y in 0..<planeHeight {
                    memcpy(d.advanced(by: y * dstStride), s.advanced(by: y * srcStride), rowBytes)
                }
            }
        }
        return destination
    }
}
