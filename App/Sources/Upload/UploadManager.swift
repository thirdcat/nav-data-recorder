import Foundation
import UIKit

/// Ships completed sessions to a server over a background `URLSession`.
///
/// Background rather than foreground because the transfers are enormous — a
/// full-rate hour is tens of gigabytes — and the user is not going to sit
/// watching a progress bar. iOS keeps these going after the app is suspended
/// and relaunches it when they finish.
///
/// Uploading never deletes anything by default. The Files-app copy stays put
/// unless the user explicitly opts into deletion, so a server-side mistake
/// cannot silently destroy a drive.
final class UploadManager: NSObject, ObservableObject, URLSessionDelegate, URLSessionTaskDelegate {

    static let sessionIdentifier = "com.thirdcat.navdatarecorder.upload"

    /// A background `URLSession` may only have one owner per identifier, and
    /// iOS can relaunch the app straight into the delegate before any SwiftUI
    /// view exists — so this has to be reachable without a view hierarchy.
    static let shared = UploadManager()

    struct SessionProgress: Equatable, Identifiable {
        let sessionID: String
        var bytesSent: UInt64
        var totalBytes: UInt64
        var completedFiles: Int
        var totalFiles: Int
        var currentFile: String?
        var inFlightFiles: Int

        var id: String { sessionID }
    }

    struct Progress: Equatable {
        var pendingFiles = 0
        var inFlightFiles = 0
        var failedFiles = 0
        var sessions: [SessionProgress] = []
        var totalBytesSent: UInt64 = 0
        var totalBytes: UInt64 = 0
        var completedFiles = 0
        var totalFiles = 0
        var activeSessionCount = 0
        var lastError: String?
        var lastCompleted: String?
        var lastCompletedResult: String?
    }

    @Published private(set) var progress = Progress()
    @Published var settings: UploadSettings {
        didSet {
            AppSettings.shared.uploadSettings = settings
            let updated = settings
            stateQueue.async { [weak self] in self?.currentSettings = updated }
            // Drop anything aimed somewhere else before queueing more. The base
            // URL is edited a character at a time and this fires on every one,
            // so `http://1`, `http://192.16` and every other prefix briefly
            // parses as a usable URL and gets a full session queued against it.
            // Those tasks then retry for `timeoutIntervalForResource` — seven
            // days — holding every file "in flight", which is exactly the state
            // `enqueue` skips. The real address never gets a task, the queue
            // never drains, and the server never sees a request.
            discardMisdirectedTasks()
        }
    }

    /// `settings` is bound to the UI and mutated on main, but the upload path
    /// runs on `stateQueue` and on URLSession's delegate queue. This is the copy
    /// those paths read, so no background code touches the published property.
    private var currentSettings: UploadSettings

    private struct ActiveUpload {
        let relativePath: String
        let fileSize: UInt64
        var bytesSent: UInt64
    }

    /// Cached while a session is queued. All fields are touched on
    /// `stateQueue`; progress-byte callbacks never inspect the disk.
    private struct SessionState {
        let sessionID: String
        let filePaths: [String]
        let fileSizes: [String: UInt64]
        let totalBytes: UInt64
        var completedFiles: Set<String>
        var completedBytes: UInt64
        var activeUploads: [Int: ActiveUpload]
    }

    /// Set by the app delegate when iOS relaunches us to report finished
    /// background transfers; called once the session has drained its events.
    var backgroundCompletionHandler: (() -> Void)?

    private var urlSession: URLSession!
    private let stateQueue = DispatchQueue(label: "nav.upload.state")
    private var sessionStates: [String: SessionState] = [:]
    private var failureCount = 0
    private var lastError: String?
    private var lastCompletedSession: String?
    private var lastCompletedResult: String?

    override init() {
        let stored = AppSettings.shared.uploadSettings
        self.settings = stored
        self.currentSettings = stored
        super.init()

        let config = URLSessionConfiguration.background(withIdentifier: Self.sessionIdentifier)
        config.allowsCellularAccess = !stored.wifiOnly
        // This used to be true so iOS could defer a 40 GB transfer until power
        // and network conditions were ideal. It also left completed recordings
        // sitting in the queue indefinitely, so Wi-Fi-only now provides the
        // data-cost guard while uploads are allowed to start promptly.
        config.isDiscretionary = false
        config.sessionSendsLaunchEvents = true
        config.timeoutIntervalForResource = 7 * 24 * 60 * 60
        self.urlSession = URLSession(configuration: config, delegate: self, delegateQueue: nil)
    }

    // MARK: - Public API

    /// Refreshes the published snapshot from the upload state cache. The
    /// settings screen calls this from a low-frequency UI timer so byte
    /// callbacks do not publish once per network packet.
    func refreshProgress() {
        stateQueue.async { [weak self] in
            self?.publishProgress()
        }
    }

    /// Result of the last `testConnection()`, for the settings screen.
    @Published private(set) var testResult: String?
    @Published private(set) var testing = false

    /// Reaches the upload server once from *this* process, and says what
    /// happened.
    ///
    /// Two jobs, and the second is the reason it exists at all.
    ///
    /// The obvious one: uploads fail silently in a dozen ways — wrong address,
    /// wrong port, laptop asleep, phone on a different network, bad token —
    /// and a queue that never drains looks identical for all of them.
    ///
    /// The one that is not obvious: since iOS 14 the first connection to a
    /// local-network address needs the user's permission, and **a background
    /// `URLSession` cannot obtain it.** Those transfers run in `nsurlsessiond`,
    /// not in the app, and a daemon cannot present a prompt — so the request is
    /// refused, no alert appears, and the app never even shows up under
    /// Settings → Privacy → Local Network, because nothing ever triggered the
    /// check. A foreground request from the app process does trigger it. So
    /// this button is also how the permission gets granted in the first place.
    func testConnection() {
        guard let base = settings.resolvedBaseURL else {
            testResult = "Enter a base URL first, including http://"
            return
        }
        testing = true
        testResult = nil

        var request = URLRequest(url: base)
        request.httpMethod = "GET"
        request.timeoutInterval = 10
        let token = settings.authToken.trimmingCharacters(in: .whitespacesAndNewlines)
        if !token.isEmpty {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        // Deliberately the shared session rather than the background one: the
        // point is to run in this process, where a prompt can be shown.
        URLSession.shared.dataTask(with: request) { [weak self] _, response, error in
            let message: String
            if let error = error as NSError? {
                message = Self.describe(error)
            } else if let http = response as? HTTPURLResponse {
                switch http.statusCode {
                case 200...299:
                    message = "Reached the server."
                case 401, 403:
                    message = "Reached the server, but it rejected the token."
                default:
                    message = "Reached the server, but it answered \(http.statusCode)."
                }
            } else {
                message = "No response."
            }
            DispatchQueue.main.async {
                self?.testing = false
                self?.testResult = message
            }
        }.resume()
    }

    private static func describe(_ error: NSError) -> String {
        // The local-network refusal arrives as a generic connection failure, so
        // name it: it is the one a user cannot diagnose from the message alone.
        switch error.code {
        case NSURLErrorCannotConnectToHost:
            return "Could not connect. Server not running, or wrong port?"
        case NSURLErrorTimedOut:
            return "Timed out. Same Wi-Fi network as the server?"
        case NSURLErrorNotConnectedToInternet, NSURLErrorNetworkConnectionLost:
            return "No network."
        case NSURLErrorAppTransportSecurityRequiresSecureConnection:
            return "Blocked as cleartext HTTP — the address must be a local one."
        default:
            return "\(error.localizedDescription) (\(error.code))"
        }
    }

    /// Queues every not-yet-uploaded file in a session. Safe to call repeatedly;
    /// files already in flight or already acknowledged are skipped.
    func enqueue(sessionID: String) {
        guard settings.isUsable else { return }
        guard SessionStore.isComplete(id: sessionID) else { return }

        urlSession.getAllTasks { [weak self] tasks in
            guard let self = self else { return }
            self.stateQueue.async {
                var state = self.sessionStates[sessionID]
                    ?? self.makeSessionState(sessionID: sessionID)
                var inFlightPaths = Set<String>()

                for task in tasks {
                    guard let key = task.taskDescription else { continue }
                    let parts = key.split(separator: "/", maxSplits: 1).map(String.init)
                    guard parts.count == 2, parts[0] == sessionID,
                          let fileSize = state.fileSizes[parts[1]] else { continue }
                    inFlightPaths.insert(parts[1])
                    if state.activeUploads[task.taskIdentifier] == nil {
                        state.activeUploads[task.taskIdentifier] = ActiveUpload(
                            relativePath: parts[1],
                            fileSize: fileSize,
                            bytesSent: min(UInt64(clamping: task.countOfBytesSent), fileSize))
                    }
                }

                for active in state.activeUploads.values {
                    inFlightPaths.insert(active.relativePath)
                }

                for relative in state.filePaths {
                    let key = "\(sessionID)/\(relative)"
                    guard !state.completedFiles.contains(relative),
                          !inFlightPaths.contains(relative) else { continue }
                    let url = SessionStore.url(for: sessionID, relativePath: relative)
                    if let task = self.startUpload(sessionID: sessionID,
                                                   relativePath: relative,
                                                   file: url,
                                                   key: key) {
                        let fileSize = state.fileSizes[relative] ?? 0
                        state.activeUploads[task.taskIdentifier] = ActiveUpload(
                            relativePath: relative,
                            fileSize: fileSize,
                            bytesSent: 0)
                        inFlightPaths.insert(relative)
                    }
                }

                self.sessionStates[sessionID] = state
                self.publishProgress()
            }
        }
    }

    /// Re-queues every complete session that still has unsent files. Called at
    /// launch and whenever the upload settings change, so a session recorded
    /// before a server was configured is not stranded.
    func resumePending() {
        guard settings.isUsable else { return }
        for id in SessionStore.listSessionIDs() where SessionStore.isComplete(id: id) {
            enqueue(sessionID: id)
        }
    }

    /// Cancels tasks that are not aimed at the current base URL, then re-queues.
    ///
    /// Called whenever the settings change and once at launch. The launch case
    /// matters as much as the edit case: a background session outlives the app,
    /// so tasks queued against a half-typed address in a previous run are still
    /// there, still retrying, still blocking their files.
    func discardMisdirectedTasks() {
        let wanted = settings.resolvedBaseURL?.absoluteString
        urlSession.getAllTasks { [weak self] tasks in
            guard let self = self else { return }
            var cancelled = 0
            for task in tasks {
                let url = task.originalRequest?.url?.absoluteString
                if let wanted, let url, url.hasPrefix(wanted) { continue }
                task.cancel()
                cancelled += 1
            }
            if cancelled > 0 {
                self.stateQueue.async {
                    // Their bookkeeping goes with them, or the files stay
                    // marked in flight against tasks that no longer exist.
                    for id in self.sessionStates.keys {
                        self.sessionStates[id]?.activeUploads.removeAll()
                    }
                    self.publishProgress()
                }
            }
            DispatchQueue.main.async { self.resumePending() }
        }
    }

    /// Files in this session that the server has acknowledged.
    static func uploadedFiles(sessionID: String) -> Set<String> {
        let url = SessionStore.directory(for: sessionID)
            .appendingPathComponent(SessionStore.Filename.uploadState)
        guard let data = try? Data(contentsOf: url),
              let names = try? JSONDecoder().decode([String].self, from: data) else {
            return []
        }
        return Set(names)
    }

    /// Whether every payload file in a session has been acknowledged.
    static func isFullyUploaded(sessionID: String) -> Bool {
        let payload = Set(SessionStore.payloadFiles(id: sessionID))
        guard !payload.isEmpty else { return false }
        return payload.isSubset(of: uploadedFiles(sessionID: sessionID))
    }

    // MARK: - Internals

    /// Builds the one disk-backed snapshot needed when a session first enters
    /// the queue. The delegate callbacks use this cache instead of enumerating
    /// the session repeatedly.
    private func makeSessionState(sessionID: String) -> SessionState {
        let filePaths = SessionStore.payloadFiles(id: sessionID)
        let fileSizes = Dictionary(uniqueKeysWithValues: filePaths.map { relative in
            let values = try? SessionStore.url(for: sessionID, relativePath: relative)
                .resourceValues(forKeys: [.fileSizeKey])
            return (relative, UInt64(max(0, values?.fileSize ?? 0)))
        })
        let uploaded = Self.uploadedFiles(sessionID: sessionID)
        let completedFiles = Set(filePaths.filter { uploaded.contains($0) })
        let completedBytes = completedFiles.reduce(UInt64(0)) { sum, relative in
            sum + (fileSizes[relative] ?? 0)
        }

        return SessionState(
            sessionID: sessionID,
            filePaths: filePaths,
            fileSizes: fileSizes,
            totalBytes: SessionStore.totalBytes(id: sessionID),
            completedFiles: completedFiles,
            completedBytes: completedBytes,
            activeUploads: [:])
    }

    /// Makes a value snapshot from stateQueue-owned data and publishes only on
    /// the main thread. This is deliberately not called by didSendBodyData.
    private func publishProgress() {
        let sessions = sessionStates.values.compactMap { state -> SessionProgress? in
            let active = Array(state.activeUploads.values)
            guard state.completedFiles.count < state.filePaths.count || !active.isEmpty else {
                return nil
            }

            let activeBytes = active.reduce(UInt64(0)) { sum, upload in
                sum + min(upload.bytesSent, upload.fileSize)
            }
            return SessionProgress(
                sessionID: state.sessionID,
                bytesSent: min(state.totalBytes, state.completedBytes + activeBytes),
                totalBytes: state.totalBytes,
                completedFiles: state.completedFiles.count,
                totalFiles: state.filePaths.count,
                currentFile: active.first?.relativePath,
                inFlightFiles: active.count)
        }
        .sorted { $0.sessionID > $1.sessionID }

        let snapshot = Progress(
            pendingFiles: sessions.reduce(0) { sum, session in
                sum + max(0, session.totalFiles - session.completedFiles - session.inFlightFiles)
            },
            inFlightFiles: sessions.reduce(0) { $0 + $1.inFlightFiles },
            failedFiles: failureCount,
            sessions: sessions,
            totalBytesSent: sessions.reduce(UInt64(0)) { $0 + $1.bytesSent },
            totalBytes: sessions.reduce(UInt64(0)) { $0 + $1.totalBytes },
            completedFiles: sessions.reduce(0) { $0 + $1.completedFiles },
            totalFiles: sessions.reduce(0) { $0 + $1.totalFiles },
            activeSessionCount: sessions.count,
            lastError: lastError,
            lastCompleted: lastCompletedSession,
            lastCompletedResult: lastCompletedResult)

        DispatchQueue.main.async { [weak self] in
            self?.progress = snapshot
        }
    }

    /// Returns a task if it was created. Runs on `stateQueue`.
    private func startUpload(sessionID: String, relativePath: String, file: URL, key: String) -> URLSessionUploadTask? {
        let settings = currentSettings
        guard let base = settings.resolvedBaseURL else { return nil }

        // `relativePath` can contain a directory component (`frames/000123.jpg`),
        // which is preserved on the server so the uploaded tree mirrors the
        // on-device one.
        var target = base.appendingPathComponent(sessionID, isDirectory: true)
        for component in relativePath.split(separator: "/") {
            target.appendPathComponent(String(component))
        }

        var request = URLRequest(url: target)
        request.httpMethod = "PUT"
        request.setValue(Self.contentType(for: file), forHTTPHeaderField: "Content-Type")
        let token = settings.authToken.trimmingCharacters(in: .whitespacesAndNewlines)
        if !token.isEmpty {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        // A background session can only upload from a file, never from Data.
        let task = urlSession.uploadTask(with: request, fromFile: file)
        task.taskDescription = key
        task.resume()
        return task
    }

    private static func contentType(for url: URL) -> String {
        switch url.pathExtension.lowercased() {
        case "json": return "application/json"
        case "jsonl": return "application/x-ndjson"
        case "mov": return "video/quicktime"
        case "jpg", "jpeg": return "image/jpeg"
        case "bin": return "application/octet-stream"
        default: return "application/octet-stream"
        }
    }

    /// Records that the server has taken this file. Runs on `stateQueue`.
    private func markUploaded(sessionID: String, relativePath: String) {
        var names = Self.uploadedFiles(sessionID: sessionID)
        names.insert(relativePath)
        let url = SessionStore.directory(for: sessionID)
            .appendingPathComponent(SessionStore.Filename.uploadState)
        if let data = try? JSONEncoder().encode(Array(names).sorted()) {
            try? data.write(to: url, options: .atomic)
        }
    }

    // MARK: - URLSessionTaskDelegate

    func urlSession(_ session: URLSession,
                    task: URLSessionTask,
                    didSendBodyData _: Int64,
                    totalBytesSent: Int64,
                    totalBytesExpectedToSend _: Int64) {
        let key = task.taskDescription ?? ""
        let sent = UInt64(clamping: totalBytesSent)

        // URLSession supplies this callback on its delegate queue, which is
        // not the main thread. Accumulate into the ordinary state cache only;
        // the UI timer later asks for a main-thread-published snapshot.
        stateQueue.async { [weak self] in
            guard let self = self else { return }
            let parts = key.split(separator: "/", maxSplits: 1).map(String.init)
            guard parts.count == 2,
                  var state = self.sessionStates[parts[0]],
                  var active = state.activeUploads[task.taskIdentifier],
                  active.relativePath == parts[1] else { return }
            active.bytesSent = min(sent, active.fileSize)
            state.activeUploads[task.taskIdentifier] = active
            self.sessionStates[parts[0]] = state
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        let key = task.taskDescription ?? ""
        // Session id first, everything after it is the relative path — which
        // may itself contain slashes, hence maxSplits.
        let parts = key.split(separator: "/", maxSplits: 1).map(String.init)
        let status = (task.response as? HTTPURLResponse)?.statusCode ?? 0
        let success = error == nil && (200...299).contains(status)

        stateQueue.async { [weak self] in
            guard let self = self else { return }
            var fullyUploaded = false

            if parts.count == 2,
               var state = self.sessionStates[parts[0]] {
                state.activeUploads.removeValue(forKey: task.taskIdentifier)
                if success {
                    if state.completedFiles.insert(parts[1]).inserted {
                        state.completedBytes += state.fileSizes[parts[1]] ?? 0
                    }
                    fullyUploaded = state.completedFiles.count == state.filePaths.count
                }
                self.sessionStates[parts[0]] = state
            }

            if success, parts.count == 2 {
                self.markUploaded(sessionID: parts[0], relativePath: parts[1])
                if fullyUploaded && currentSettings.deleteAfterUpload {
                    try? SessionStore.delete(id: parts[0])
                }
            }
            let message: String?
            if let error = error {
                message = "Upload failed for \(key): \(error.localizedDescription)"
            } else if !success {
                message = "Upload failed for \(key): HTTP \(status)"
            } else {
                message = nil
            }

            if success && fullyUploaded {
                self.lastCompletedSession = parts[0]
                self.lastCompletedResult = "All files uploaded"
            } else if !success {
                self.failureCount += 1
                self.lastError = message
            }
            self.publishProgress()
        }
    }

    func urlSessionDidFinishEvents(forBackgroundURLSession session: URLSession) {
        DispatchQueue.main.async { [weak self] in
            self?.backgroundCompletionHandler?()
            self?.backgroundCompletionHandler = nil
        }
    }
}
