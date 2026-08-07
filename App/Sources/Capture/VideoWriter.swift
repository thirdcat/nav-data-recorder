import AVFoundation
import CoreVideo
import Foundation

/// Encodes `ARFrame.capturedImage` buffers into an HEVC .mov.
///
/// Presentation timestamps are written in the master monotonic clock domain
/// (`ARFrame.timestamp`) rather than starting at zero. That means a frame's PTS
/// is literally the same number as the matching `PoseSample.t`, so aligning
/// video against the sensor streams offline is a lookup, not a computation.
final class VideoWriter {

    private var writer: AVAssetWriter?
    private var input: AVAssetWriterInput?
    private var adaptor: AVAssetWriterInputPixelBufferAdaptor?
    private var started = false

    private(set) var frameCount = 0
    private(set) var droppedCount = 0
    private(set) var info: SessionManifest.VideoInfo?

    let url: URL
    private let bitrate: Int
    private let fps: Int

    /// Frames arriving faster than `fps` are dropped here rather than being
    /// handed to the encoder — ARKit runs at 60 Hz on a 16 Pro and we usually
    /// want 30. Poses for dropped frames are still recorded upstream.
    private var lastAcceptedPTS: Double = -.infinity

    init(url: URL, bitrate: Int, fps: Int) {
        self.url = url
        self.bitrate = bitrate
        self.fps = max(1, fps)
    }

    /// Configures the writer from the first buffer, since ARKit's capture
    /// resolution depends on the device and the chosen video format.
    private func start(with pixelBuffer: CVPixelBuffer, pts: Double) throws {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)

        let writer = try AVAssetWriter(outputURL: url, fileType: .mov)

        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: bitrate,
                AVVideoExpectedSourceFrameRateKey: fps,
                // Keyframe at least once a second: long GOPs would make seeking
                // during dataset preparation painful.
                AVVideoMaxKeyFrameIntervalDurationKey: 1.0
            ]
        ]

        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = true

        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: CVPixelBufferGetPixelFormatType(pixelBuffer),
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height
            ])

        guard writer.canAdd(input) else { throw VideoWriterError.cannotAddInput }
        writer.add(input)

        guard writer.startWriting() else {
            throw writer.error ?? VideoWriterError.cannotStartWriting
        }
        writer.startSession(atSourceTime: Self.time(pts))

        self.writer = writer
        self.input = input
        self.adaptor = adaptor
        self.started = true
        self.info = SessionManifest.VideoInfo(
            file: url.lastPathComponent,
            width: width,
            height: height,
            codec: "hevc",
            nominalFPS: fps,
            bitrate: bitrate,
            firstFramePTS: pts)
    }

    /// Returns true if the frame was encoded, false if it was skipped.
    @discardableResult
    func append(_ pixelBuffer: CVPixelBuffer, pts: Double) -> Bool {
        // Rate limit before doing any work. The 0.5 factor accepts a frame that
        // lands slightly early rather than dropping it and stuttering to half
        // the requested rate.
        let minInterval = 1.0 / Double(fps)
        if pts - lastAcceptedPTS < minInterval * 0.5 {
            return false
        }

        if !started {
            do {
                try start(with: pixelBuffer, pts: pts)
            } catch {
                droppedCount += 1
                return false
            }
        }

        guard let input = input, let adaptor = adaptor, let writer = writer else { return false }
        guard writer.status == .writing else {
            droppedCount += 1
            return false
        }
        // The encoder is allowed to fall behind; dropping is far better than
        // blocking ARKit's delegate queue and stalling every other stream.
        guard input.isReadyForMoreMediaData else {
            droppedCount += 1
            return false
        }

        if adaptor.append(pixelBuffer, withPresentationTime: Self.time(pts)) {
            frameCount += 1
            lastAcceptedPTS = pts
            return true
        } else {
            droppedCount += 1
            return false
        }
    }

    /// Finalises the .mov. Blocks until the file is playable, which matters —
    /// an un-finalised AVAssetWriter output is an unreadable file.
    func finish(completion: @escaping () -> Void) {
        guard started, let writer = writer, let input = input else {
            completion()
            return
        }
        input.markAsFinished()
        writer.finishWriting {
            completion()
        }
        self.started = false
    }

    /// Microsecond timescale — fine enough that rounding never merges two
    /// frames, coarse enough to stay well inside Int64.
    private static func time(_ seconds: Double) -> CMTime {
        CMTime(seconds: seconds, preferredTimescale: 1_000_000)
    }

    enum VideoWriterError: Error {
        case cannotAddInput
        case cannotStartWriting
    }
}
