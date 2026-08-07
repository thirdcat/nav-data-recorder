import CoreLocation
import SwiftUI

struct RecordView: View {
    @EnvironmentObject private var coordinator: RecordingCoordinator
    @Environment(\.verticalSizeClass) private var verticalSizeClass

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    preview
                    warnings
                    if coordinator.state == .recording || coordinator.state == .stopping {
                        liveStats
                    } else {
                        readiness
                    }
                    recordButton
                    if let message = coordinator.statusMessage {
                        Text(message)
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)
                            .frame(maxWidth: .infinity)
                    }
                }
                .padding()
            }
            .navigationTitle("Record")
        }
    }

    // MARK: - Preview

    private var preview: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.black)
            if let image = coordinator.previewImage {
                Image(decorative: image, scale: 1, orientation: previewOrientation)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
            } else {
                VStack(spacing: 6) {
                    Image(systemName: "camera.metering.none")
                        .font(.largeTitle)
                    Text(coordinator.isRecording ? "Waiting for frames…" : "Camera preview appears while recording")
                        .font(.caption)
                        .multilineTextAlignment(.center)
                }
                .foregroundStyle(.white.opacity(0.6))
                .padding()
            }
        }
        .frame(height: 220)
    }

    /// ARKit hands back frames in the camera's native landscape orientation, so
    /// they need rotating when the phone is held upright.
    private var previewOrientation: Image.Orientation {
        verticalSizeClass == .compact ? .up : .right
    }

    // MARK: - Warnings

    @ViewBuilder
    private var warnings: some View {
        VStack(spacing: 8) {
            if !coordinator.canRecord {
                banner("This device does not support ARKit world tracking.", .red)
            }
            if !coordinator.hasLiDAR {
                banner("No LiDAR on this device — depth capture will be skipped.", .orange)
            }
            switch coordinator.locationAuthorization {
            case .denied, .restricted:
                banner("Location access is denied. Enable it in Settings or the GPS track will be empty.", .red)
            case .authorizedWhenInUse:
                banner("Location is “While Using”. Set it to “Always” so the track survives the screen locking.", .orange)
            default:
                EmptyView()
            }
            // Driven by the thermal state rather than by `isThrottled`, so the
            // warning is visible *before* a recording starts. Beginning a
            // session on a hot device is the case worth catching — it produces
            // a session with no images in it.
            switch coordinator.thermalState {
            case .serious, .critical:
                banner(coordinator.isThrottled
                       ? "Device is hot — video and depth are paused. GPS and IMU are still recording."
                       : "Device is hot (\(RecordingCoordinator.describe(coordinator.thermalState))). Let it cool before recording, or images and depth will be skipped.",
                       .orange)
            default:
                EmptyView()
            }
            if coordinator.freeBytes < 8 * 1024 * 1024 * 1024 {
                banner("Only \(Format.bytes(coordinator.freeBytes)) free. Recording stops automatically at 2 GB.", .orange)
            }
        }
    }

    private func banner(_ text: String, _ color: Color) -> some View {
        Text(text)
            .font(.footnote)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(10)
            .background(color.opacity(0.15), in: RoundedRectangle(cornerRadius: 8))
            .foregroundStyle(color)
    }

    // MARK: - Stats

    private var liveStats: some View {
        VStack(spacing: 12) {
            Text(Format.duration(coordinator.elapsed))
                .font(.system(size: 44, weight: .semibold, design: .monospaced))

            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 10) {
                stat("GPS fixes", "\(coordinator.stats.locations)")
                stat("IMU samples", "\(coordinator.stats.motion)")
                stat("Poses", "\(coordinator.stats.poses)")
                stat(coordinator.config.captureMode == .stills ? "Images" : "Video frames",
                     "\(coordinator.stats.videoFrames)")
                stat("Depth frames", "\(coordinator.stats.depthFrames)")
                stat("On disk", Format.bytes(coordinator.stats.bytesOnDisk))
            }

            if coordinator.stats.droppedVideoFrames > 0 {
                Text("\(coordinator.stats.droppedVideoFrames) frames dropped by the encoder")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if let loc = coordinator.lastLocation {
                VStack(spacing: 2) {
                    Text("\(Format.coordinate(loc.coordinate.latitude)), \(Format.coordinate(loc.coordinate.longitude))")
                        .font(.caption.monospaced())
                    Text("\(Format.speedKPH(loc.speed)) · ±\(String(format: "%.0f", loc.horizontalAccuracy)) m")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private var readiness: some View {
        VStack(alignment: .leading, spacing: 8) {
            row("RGB", coordinator.config.captureMode == .stills
                ? "\(Int(coordinator.config.stillsHz)) Hz JPEG"
                : "\(coordinator.config.videoFPS) fps HEVC")
            row("Depth", coordinator.config.recordDepth && coordinator.hasLiDAR
                ? "\(Int(coordinator.config.depthHz)) Hz LiDAR" : "off")
            row("IMU", "\(Int(coordinator.config.motionHz)) Hz")
            row("Free space", Format.bytes(coordinator.freeBytes))
            row("Thermal", RecordingCoordinator.describe(coordinator.thermalState))
        }
        .padding()
        .background(Color.secondary.opacity(0.1), in: RoundedRectangle(cornerRadius: 10))
    }

    private func row(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label).foregroundStyle(.secondary)
            Spacer()
            Text(value)
        }
        .font(.subheadline)
    }

    private func stat(_ label: String, _ value: String) -> some View {
        VStack(spacing: 2) {
            Text(value)
                .font(.title3.monospacedDigit())
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 8)
        .background(Color.secondary.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
    }

    // MARK: - Button

    private var recordButton: some View {
        Button {
            if coordinator.isRecording {
                coordinator.stop()
            } else {
                coordinator.start()
            }
        } label: {
            Label(buttonTitle, systemImage: coordinator.isRecording ? "stop.fill" : "record.circle")
                .font(.title3.weight(.semibold))
                .frame(maxWidth: .infinity)
                .padding(.vertical, 14)
        }
        .buttonStyle(.borderedProminent)
        .tint(coordinator.isRecording ? .red : .accentColor)
        .disabled(!coordinator.canRecord || coordinator.state == .stopping)
    }

    private var buttonTitle: String {
        switch coordinator.state {
        case .idle: return "Start recording"
        case .starting: return "Starting…"
        case .recording: return "Stop"
        case .stopping: return "Finalising…"
        }
    }
}
