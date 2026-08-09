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
        Section {
            Picker("RGB format", selection: binding(\.captureMode)) {
                Text("Stills (JPEG)").tag(CaptureConfig.CaptureMode.stills)
                Text("Video (HEVC)").tag(CaptureConfig.CaptureMode.video)
            }
            Picker("Frame shape", selection: binding(\.formatPreference)) {
                Text("4:3 — taller").tag(CaptureConfig.FormatPreference.tallest)
                Text("16:9 — more pixels").tag(CaptureConfig.FormatPreference.highestResolution)
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
            // ~500 KB for a 1920x1440 JPEG at q0.85; scales roughly linearly
            // with quality over the range offered.
            let perImage = 500_000.0 * (config.stillQuality / 0.85)
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

            if uploadManager.progress.inFlightFiles > 0 {
                Label("\(uploadManager.progress.inFlightFiles) files in flight", systemImage: "arrow.up.circle")
                    .font(.footnote)
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
