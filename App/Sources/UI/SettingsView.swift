import SwiftUI

struct SettingsView: View {
    @EnvironmentObject private var coordinator: RecordingCoordinator
    @EnvironmentObject private var uploadManager: UploadManager

    var body: some View {
        NavigationStack {
            Form {
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
        }
    }

    private var captureSection: some View {
        Section("Capture") {
            Toggle("Video", isOn: binding(\.recordVideo))
            Toggle("LiDAR depth", isOn: binding(\.recordDepth))
                .disabled(!coordinator.hasLiDAR)
            Toggle("Depth confidence map", isOn: binding(\.recordConfidence))
                .disabled(!coordinator.config.recordDepth || !coordinator.hasLiDAR)
            Toggle("Pause camera when hot", isOn: binding(\.degradeOnThermalPressure))
        } footer: {
            Text("GPS and IMU are always recorded. When the device overheats, pausing the camera keeps the GPS track alive instead of losing the whole drive.")
        }
    }

    private var rateSection: some View {
        Section("Rates") {
            Stepper("Video \(coordinator.config.videoFPS) fps",
                    value: binding(\.videoFPS), in: 5...60, step: 5)
            Stepper("Depth \(Int(coordinator.config.depthHz)) Hz",
                    value: Binding(
                        get: { Int(coordinator.config.depthHz) },
                        set: { coordinator.config.depthHz = Double($0) }),
                    in: 1...30, step: 1)
            Stepper("IMU \(Int(coordinator.config.motionHz)) Hz",
                    value: Binding(
                        get: { Int(coordinator.config.motionHz) },
                        set: { coordinator.config.motionHz = Double($0) }),
                    in: 10...200, step: 10)
            Picker("Video bitrate", selection: binding(\.videoBitrate)) {
                Text("6 Mbps").tag(6_000_000)
                Text("12 Mbps").tag(12_000_000)
                Text("24 Mbps").tag(24_000_000)
                Text("40 Mbps").tag(40_000_000)
            }
        } footer: {
            Text(estimate)
        }
    }

    /// Rough per-hour footprint. Storage is the binding constraint on how long a
    /// drive can be, so it belongs next to the controls that determine it.
    private var estimate: String {
        let config = coordinator.config
        var bytesPerSecond = 0.0
        if config.recordVideo {
            bytesPerSecond += Double(config.videoBitrate) / 8.0
        }
        if config.recordDepth && coordinator.hasLiDAR {
            // 256x192 float16, plus a byte per pixel when confidence is on.
            let perFrame = 256.0 * 192.0 * (config.recordConfidence ? 3.0 : 2.0)
            bytesPerSecond += perFrame * config.depthHz
        }
        // JSONL rows: ~200 B per IMU sample, ~250 B per pose.
        bytesPerSecond += config.motionHz * 200
        bytesPerSecond += Double(config.videoFPS) * 250

        let perHour = bytesPerSecond * 3600
        let hoursOfSpace = Double(coordinator.freeBytes) / max(perHour, 1)
        return String(format: "About %@ per hour — roughly %.1f hours in the free space left.",
                      Format.bytes(UInt64(perHour)), hoursOfSpace)
    }

    private var uploadSection: some View {
        Section("Upload") {
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

            if uploadManager.progress.inFlightFiles > 0 {
                Label("\(uploadManager.progress.inFlightFiles) files in flight", systemImage: "arrow.up.circle")
                    .font(.footnote)
            }
            if let error = uploadManager.progress.lastError {
                Label(error, systemImage: "exclamationmark.triangle")
                    .font(.footnote)
                    .foregroundStyle(.orange)
            }
        } footer: {
            Text("Each file is PUT to <base URL>/<session id>/<filename>. Wi-Fi only is on by default because a full session runs to tens of gigabytes. Changing the Wi-Fi-only setting takes effect on the next app launch.")
        }
    }

    private var aboutSection: some View {
        Section("Device") {
            LabeledContent("LiDAR", value: coordinator.hasLiDAR ? "available" : "not available")
            LabeledContent("ARKit", value: coordinator.canRecord ? "supported" : "unsupported")
            LabeledContent("Location", value: LocationRecorder.describe(coordinator.locationAuthorization))
            LabeledContent("Free space", value: Format.bytes(coordinator.freeBytes))
            LabeledContent("Version", value: Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "?")
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
