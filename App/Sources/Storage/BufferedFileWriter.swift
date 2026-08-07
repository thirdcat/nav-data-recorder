import Foundation

/// Append-only file writer with an in-memory buffer.
///
/// At 100 Hz motion plus 30 Hz poses a naive write-per-sample would hit the
/// filesystem ~130 times a second for the whole drive. Buffering to 64 KB cuts
/// that to a handful of writes per second; `flush()` on backgrounding and stop
/// bounds what a crash can cost to roughly one buffer.
///
/// Not thread-safe on its own — callers serialise access on their own queue.
final class BufferedFileWriter {
    let url: URL
    private var handle: FileHandle
    private var buffer = Data()
    private let threshold: Int
    private(set) var bytesWritten: UInt64 = 0
    private var closed = false

    init(url: URL, bufferSize: Int = 64 * 1024) throws {
        self.url = url
        self.threshold = bufferSize
        let fm = FileManager.default
        if !fm.fileExists(atPath: url.path) {
            guard fm.createFile(atPath: url.path, contents: nil) else {
                throw CocoaError(.fileWriteUnknown)
            }
        }
        self.handle = try FileHandle(forWritingTo: url)
        // Seed the byte counter from the file's existing length rather than
        // zero, so `nextOffset` stays truthful if a file is ever reopened —
        // depth index entries are built from it and would otherwise point into
        // the wrong place.
        self.bytesWritten = try self.handle.seekToEnd()
        self.buffer.reserveCapacity(bufferSize + 4096)
    }

    /// Byte offset the next appended payload will land at, counting buffered
    /// but not yet flushed bytes. Depth index entries are built from this, so
    /// it has to account for the buffer.
    var nextOffset: UInt64 { bytesWritten + UInt64(buffer.count) }

    func append(_ data: Data) throws {
        guard !closed else { return }
        buffer.append(data)
        if buffer.count >= threshold {
            try flush()
        }
    }

    func flush() throws {
        guard !closed, !buffer.isEmpty else { return }
        let payload = buffer
        buffer.removeAll(keepingCapacity: true)
        try handle.write(contentsOf: payload)
        bytesWritten += UInt64(payload.count)
    }

    func close() {
        guard !closed else { return }
        try? flush()
        try? handle.close()
        closed = true
    }

    deinit {
        close()
    }
}

/// Writes `Codable` rows as newline-delimited JSON.
///
/// JSONL rather than one big JSON array: a session that dies mid-drive — crash,
/// battery, force-quit — still leaves every completed line readable, and the
/// file can be streamed by a reader instead of parsed whole.
final class JSONLWriter {
    private let writer: BufferedFileWriter
    private let encoder: JSONEncoder
    private(set) var count: Int = 0

    init(url: URL) throws {
        self.writer = try BufferedFileWriter(url: url)
        self.encoder = JSONEncoder()
        // Compact keys, stable ordering, and full precision on doubles — the
        // default strategy would round-trip lat/lon lossily in some cases.
        self.encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    }

    func write<T: Encodable>(_ row: T) throws {
        var data = try encoder.encode(row)
        data.append(0x0A)
        try writer.append(data)
        count += 1
    }

    func flush() throws { try writer.flush() }
    func close() { writer.close() }
}
