import Combine
import SwiftUI

struct SettingsView: View {
    @EnvironmentObject private var coordinator: RecordingCoordinator
    @EnvironmentObject private var uploadManager: UploadManager

    var body: some View {
        NavigationStack {
            Form {
                presetSection
                captureSection
                rateSection
                uploadSection
                aboutSection
            }
            .navigationTitle("Settings")
            // Changing capture settings mid-drive would make the manifest lie
            // about how the session was recorded.
            .disabled(coordinator.isRecording)
            .overlay {
                if coordinator.isRecording {
                    Text("Settings are locked while recording.")
                        .font(.footnote)
                        .padding(8)
                        .background(.thinMaterial, in: Capsule())
                        .frame(maxHeight: .infinity, alignment: .bottom)
                        .padding(.bottom, 24)
                }
            }
            .onReceive(Timer.publish(every: 0.25, on: .main, in: .common).autoconnect()) { _ in
                uploadManager.refreshProgress()
            }
        }
    }

    /// Buttons rather than a `Picker`, deliberately.
    ///
    /// A picker's selection has to be one of its tags, so a config that has
    /// been hand-edited away from every preset would have to be drawn as one of
    /// them anyway. These show a tick only when the settings really match, and
    /// the row underneath says what the manifest is going to record — including
    /// `custom`, which is a legitimate answer and not an error.
    private var presetSection: some View {
        Section {
            ForEach(CaptureConfig.Preset.allCases, id: \.self) { preset in
                Button {
                    coordinator.config = coordinator.config.applying(preset)
                } label: {
                    HStack(alignment: .firstTextBaseline) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(preset.title)
                            Text(preset.summary)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        if coordinator.config.activePreset == preset {
                            Image(systemName: "checkmark")
                                .foregroundStyle(Color.accentColor)
                        }
                    }
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
            LabeledContent("manifest.json will say",
                           value: coordinator.config.presetName)
        } header: {
            Text("Preset")
        } footer: {
            Text("""
                 One recorder, two datasets. 5 Hz is what VLN episodes are consumed at; a Gaussian splat is not consumed at any rate, so the scan preset takes every ARKit frame that carries depth.

                 A preset moves the whole bundle at once and the session records which one it was, so a rate raised for a scan cannot follow you into the next episode without saying so.

                 The name is read back off the settings rather than remembered, so editing a rate below re-derives it: land on a preset's exact values and it says so, land anywhere else and it says "custom" — which is recorded, and which leaves the duration cap off.
                 """)
        }
    }

    private var captureSection: some View {
        Section {
            Picker("RGB format", selection: binding(\.captureMode)) {
                Text("Stills (JPEG)").tag(CaptureConfig.CaptureMode.stills)
                Text("Video (HEVC)").tag(CaptureConfig.CaptureMode.video)
            }
            Picker("Frame shape", selection: binding(\.formatPreference)) {
                Text("4:3 — taller").tag(CaptureConfig.FormatPreference.tallest)
                Text("16:9 — more pixels").tag(CaptureConfig.FormatPreference.highestResolution)
            }
            Picker("Sensor rate", selection: binding(\.frameRatePreference)) {
                Text("Fastest").tag(CaptureConfig.FrameRatePreference.highest)
                Text("30 fps").tag(CaptureConfig.FrameRatePreference.thirty)
                Text("60 fps").tag(CaptureConfig.FrameRatePreference.sixty)
            }
            Toggle("LiDAR depth", isOn: binding(\.recordDepth))
                .disabled(!coordinator.hasLiDAR)
            Toggle("Depth confidence map", isOn: binding(\.recordConfidence))
                .disabled(!coordinator.config.recordDepth || !coordinator.hasLiDAR)
            Toggle("Detect planes", isOn: binding(\.detectPlanes))
            Toggle("Magnetometer heading correction", isOn: binding(\.useMagnetometerCorrection))
            Toggle("Pause camera when hot", isOn: binding(\.degradeOnThermalPressure))
        } header: {
            Text("Capture")
        } footer: {
            Text("""
                 GPS and IMU are always recorded, though indoors GPS is context rather than a pose source — ARKit does the localising.

                 Leave magnetometer correction off indoors: it pulls device heading towards magnetic north, and wiring, appliances and steel in a building all lie about where that is.

                 Sensor rate is the camera's own rate, not the stills rate below. It was never selectable before and every recording so far came out at 60 because two formats tied on pixel count; "Fastest" is that same choice made on purpose. 30 halves the read noise and doubles the motion blur at a walking pace — a trade to measure on a recorded pair, not to reason out. Changing it does not make the preset read as custom.

                 """)
        }
    }

    private var rateSection: some View {
        Section {
            if coordinator.config.captureMode == .stills {
                Stepper("Stills \(Int(coordinator.config.stillsHz)) Hz",
                        value: Binding(
                            get: { Int(coordinator.config.stillsHz) },
                            set: { coordinator.config.stillsHz = Double($0) }),
                        in: 1...30, step: 1)
                Picker("JPEG quality", selection: binding(\.stillQuality)) {
                    Text("0.70").tag(0.70)
                    Text("0.85").tag(0.85)
                    Text("0.95").tag(0.95)
                }
            } else {
                Stepper("Video \(coordinator.config.videoFPS) fps",
                        value: binding(\.videoFPS), in: 5...60, step: 5)
                Picker("Video bitrate", selection: binding(\.videoBitrate)) {
                    Text("6 Mbps").tag(6_000_000)
                    Text("12 Mbps").tag(12_000_000)
                    Text("24 Mbps").tag(24_000_000)
                    Text("40 Mbps").tag(40_000_000)
                }
            }
            Stepper("Depth \(Int(coordinator.config.depthHz)) Hz",
                    value: Binding(
                        get: { Int(coordinator.config.depthHz) },
                        set: { coordinator.config.depthHz = Double($0) }),
                    in: 1...60, step: 1)
                .disabled(!coordinator.config.recordDepth)
            Stepper("IMU \(Int(coordinator.config.motionHz)) Hz",
                    value: Binding(
                        get: { Int(coordinator.config.motionHz) },
                        set: { coordinator.config.motionHz = Double($0) }),
                    in: 10...200, step: 10)
        } header: {
            Text("Rates")
        } footer: {
            Text(estimate)
        }
    }

    /// Rough per-hour footprint. Storage is the binding constraint on how long a
    /// drive can be, so it belongs next to the controls that determine it.
    private var estimate: String {
        let config = coordinator.config
        var bytesPerSecond = 0.0
        switch config.captureMode {
        case .video:
            bytesPerSecond += Double(config.videoBitrate) / 8.0
        case .stills:
            // Measured, not assumed: 391 KB is the mean over the 5 065
            // 1920x1440 frames in `~/nav_data`, against the 500 KB this line
            // used to guess. The median is 343 KB and the 90th percentile 649,
            // so a bright, detailed room costs nearly twice a dim one — the
            // mean is the right one for a footprint. Scales roughly linearly
            // with quality over the range offered.
            let perImage = 391_000.0 * (config.stillQuality / 0.85)
            bytesPerSecond += perImage * config.stillsHz
        }
        if config.recordDepth && coordinator.hasLiDAR {
            // 256x192 float16, plus a byte per pixel when confidence is on.
            let perFrame = 256.0 * 192.0 * (config.recordConfidence ? 3.0 : 2.0)
            bytesPerSecond += perFrame * config.depthHz
        }
        // JSONL rows: ~200 B per IMU sample, ~250 B per pose.
        bytesPerSecond += config.motionHz * 200
        // One pose row per ARKit frame regardless of capture mode.
        bytesPerSecond += 60 * 250

        let perMinute = bytesPerSecond * 60
        let minutesOfSpace = Double(coordinator.freeBytes) / max(perMinute, 1)
        // Per minute rather than per hour: a room scan is a few minutes, and an
        // hourly figure makes the number feel further away than it is.
        return String(format: "About %@ per minute — roughly %.0f minutes in the free space left.",
                      Format.bytes(UInt64(perMinute)), minutesOfSpace)
    }

    /// Split out of `uploadSection` rather than inlined. That section already
    /// carries five `Binding(get:set:)` closures and several conditional
    /// branches, and `ViewBuilder` takes at most ten children — adding two more
    /// put it on the limit and gave the type checker a body it would not
    /// finish. Sub-views cost nothing and both problems go away.
    @ViewBuilder
    private var connectionTest: some View {
        // Also how the local-network permission gets granted: a background
        // URLSession runs outside the app and cannot raise the prompt, so
        // without a foreground request the app never appears under
        // Settings → Privacy → Local Network and every upload is refused with
        // nothing on screen to say why.
        Button {
            uploadManager.testConnection()
        } label: {
            HStack {
                Text("Test connection")
                if uploadManager.testing {
                    Spacer()
                    ProgressView()
                }
            }
        }
        .disabled(uploadManager.testing)

        if let result = uploadManager.testResult {
            let reached = result.hasPrefix("Reached the server.")
            Text(result)
                .font(.footnote)
                .foregroundStyle(reached ? Color.secondary : Color.orange)
        }
    }

    private var uploadSection: some View {
        Section {
            Toggle("Upload sessions", isOn: Binding(
                get: { uploadManager.settings.enabled },
                set: { uploadManager.settings.enabled = $0 }))

            TextField("https://example.com/uploads", text: Binding(
                get: { uploadManager.settings.baseURL },
                set: { uploadManager.settings.baseURL = $0 }))
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .keyboardType(.URL)

            SecureField("Bearer token (optional)", text: Binding(
                get: { uploadManager.settings.authToken },
                set: { uploadManager.settings.authToken = $0 }))

            Toggle("Wi-Fi only", isOn: Binding(
                get: { uploadManager.settings.wifiOnly },
                set: { uploadManager.settings.wifiOnly = $0 }))

            Toggle("Delete after upload", isOn: Binding(
                get: { uploadManager.settings.deleteAfterUpload },
                set: { uploadManager.settings.deleteAfterUpload = $0 }))

            connectionTest

            if !uploadManager.progress.sessions.isEmpty {
                Text("\(uploadManager.progress.activeSessionCount) active session\(uploadManager.progress.activeSessionCount == 1 ? "" : "s")")
                    .font(.footnote)

                ForEach(uploadManager.progress.sessions) { session in
                    VStack(alignment: .leading, spacing: 6) {
                        Text(session.sessionID)
                            .font(.footnote)
                            .monospaced()
                            .lineLimit(1)
                        Text("\(session.completedFiles) / \(session.totalFiles) files   \(Format.bytes(session.bytesSent)) / \(Format.bytes(session.totalBytes))")
                            .font(.footnote)
                        ProgressView(
                            value: Double(session.bytesSent),
                            total: max(Double(session.totalBytes), 1))
                        HStack {
                            if let currentFile = session.currentFile {
                                Text(currentFile)
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                                    .lineLimit(1)
                            }
                            Spacer()
                            Text("\(uploadPercent(session.bytesSent, total: session.totalBytes))%")
                                .font(.caption)
                                .monospacedDigit()
                        }
                    }
                    .padding(.vertical, 3)
                }

                Text("All active sessions: \(uploadManager.progress.completedFiles) / \(uploadManager.progress.totalFiles) files   \(Format.bytes(uploadManager.progress.totalBytesSent)) / \(Format.bytes(uploadManager.progress.totalBytes))")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
            if let completed = uploadManager.progress.lastCompleted {
                Label("Finished \(completed) — \(uploadManager.progress.lastCompletedResult ?? "uploaded")",
                      systemImage: "checkmark.circle")
                    .font(.footnote)
                    .foregroundStyle(.green)
            }
            if let error = uploadManager.progress.lastError {
                Label(error, systemImage: "exclamationmark.triangle")
                    .font(.footnote)
                    .foregroundStyle(.orange)
            }
        } header: {
            Text("Upload")
        } footer: {
            Text("Each file is PUT to <base URL>/<session id>/<filename>. Wi-Fi only is on by default because a full session runs to tens of gigabytes. Changing the Wi-Fi-only setting takes effect on the next app launch.")
        }
    }

    private func uploadPercent(_ bytesSent: UInt64, total: UInt64) -> Int {
        guard total > 0 else { return 0 }
        return min(100, Int((Double(bytesSent) / Double(total) * 100).rounded()))
    }

    private var aboutSection: some View {
        Section("Device") {
            LabeledContent("LiDAR", value: coordinator.hasLiDAR ? "available" : "not available")
            LabeledContent("ARKit", value: coordinator.canRecord ? "supported" : "unsupported")
            LabeledContent("Location", value: LocationRecorder.describe(coordinator.locationAuthorization))
            LabeledContent("Free space", value: Format.bytes(coordinator.freeBytes))
            LabeledContent("Version", value: Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "?")
            NavigationLink("Hardware capabilities") {
                DiagnosticsView()
            }
            NavigationLink("Ultra-wide probe") {
                UltraWideProbeView()
            }
            NavigationLink("Multi-cam depth probe") {
                MultiCamDepthProbeView()
            }
            NavigationLink("Ultra-wide walk (no poses)") {
                MultiCamRecordView()
            }
        }
    }

    /// Writes straight through to `coordinator.config`, whose `didSet` persists
    /// it — so a setting survives a force-quit between drives.
    private func binding<T>(_ keyPath: WritableKeyPath<CaptureConfig, T>) -> Binding<T> {
        Binding(
            get: { coordinator.config[keyPath: keyPath] },
            set: { coordinator.config[keyPath: keyPath] = $0 })
    }
}
