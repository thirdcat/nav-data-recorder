import CoreLocation
import simd
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

    /// The walk seen from above, in ARKit's gravity-aligned world, with the
    /// starting point ringed. A loop that has not closed is the one defect that
    /// cannot be recovered afterwards — the score for every session here is the
    /// distance between where it started and where it ended, and a session that
    /// did not come back is unscoreable no matter how good the depth was.
    /// One walk was found 0.535 m open, hours later, at a desk.
    private var trackMap: some View {
        let track = coordinator.stats.track
        return ZStack {
            RoundedRectangle(cornerRadius: 10)
                .fill(Color.primary.opacity(0.05))
            if track.count > 2 {
                GeometryReader { geo in
                    // Horizontal plane only: ARKit's world is gravity-aligned,
                    // so height is the y axis and the map is x against z.
                    let xs = track.map { Double($0.x) }
                    let zs = track.map { Double($0.z) }
                    let cx = (xs.min()! + xs.max()!) / 2
                    let cz = (zs.min()! + zs.max()!) / 2
                    let span = max(xs.max()! - xs.min()!,
                                   zs.max()! - zs.min()!, 0.5) * 1.15
                    let side = min(geo.size.width, geo.size.height)
                    let ox = (geo.size.width - side) / 2
                    let oy = (geo.size.height - side) / 2
                    let place = { (x: Double, z: Double) -> CGPoint in
                        CGPoint(x: ox + side * ((x - cx) / span + 0.5),
                                y: oy + side * (0.5 - (z - cz) / span))
                    }
                    Path { path in
                        path.move(to: place(xs[0], zs[0]))
                        for i in 1..<track.count { path.addLine(to: place(xs[i], zs[i])) }
                    }
                    .stroke(Color.accentColor, style: StrokeStyle(lineWidth: 2,
                                                                  lineCap: .round,
                                                                  lineJoin: .round))
                    Circle()
                        .stroke(Color.secondary, lineWidth: 2)
                        .frame(width: 11, height: 11)
                        .position(place(xs[0], zs[0]))
                    Circle()
                        .fill(Color.accentColor)
                        .frame(width: 9, height: 9)
                        .position(place(xs[track.count - 1], zs[track.count - 1]))
                }
                .padding(8)
            } else {
                Text("Map appears once you have moved")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .frame(height: 150)
        .overlay(alignment: .topLeading) {
            if track.count > 2 {
                Text(Self.loopLabel(track))
                    .font(.caption2.monospaced())
                    .foregroundStyle(.secondary)
                    .padding(7)
            }
        }
        .accessibilityLabel("Walk seen from above. \(Self.loopLabel(track))")
    }

    /// How far the walk currently is from where it began, which is the number
    /// the session will be scored on.
    private static func loopLabel(_ track: [SIMD3<Float>]) -> String {
        guard let first = track.first, let last = track.last else { return "" }
        let gap = Double(simd_distance(
            SIMD2(first.x, first.z), SIMD2(last.x, last.z)))
        return String(format: "%.2f m from start", gap)
    }

    /// Conditioning is a ratio spanning decades, so a number on screen is
    /// noise to read while walking. The band is the part that changes a
    /// decision.
    private static func conditioningLabel(_ cond: Double) -> String {
        if cond >= 0.02 { return "well constrained" }
        if cond >= ARRecorder.conditioningFloor { return "adequate" }
        return "weak — depth alone may not hold this"
    }

    /// Naming the axis the geometry cannot see turns a warning into an
    /// instruction. The vector is in the depth frame: +X right, +Y down,
    /// +Z forward.
    private static func weakAxisAdvice(_ axis: [Double]?) -> String {
        guard let axis = axis, axis.count == 3 else {
            return "Point at something with more shape."
        }
        let (x, y, z) = (abs(axis[0]), abs(axis[1]), abs(axis[2]))
        if z >= x && z >= y { return "Depth cannot see forward motion — turn to face along your path." }
        if x >= y { return "Depth cannot see sideways motion — turn towards a corner or doorway." }
        return "Depth cannot see height — tilt down to bring the floor in."
    }

    private var liveStats: some View {
        VStack(spacing: 12) {
            Text(Format.duration(coordinator.elapsed))
                .font(.system(size: 44, weight: .semibold, design: .monospaced))

            trackMap

            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 10) {
                stat("GPS fixes", "\(coordinator.stats.locations)")
                stat("IMU samples", "\(coordinator.stats.motion)")
                stat("Poses", "\(coordinator.stats.poses)")
                stat(coordinator.config.captureMode == .stills ? "Images" : "Video frames",
                     "\(coordinator.stats.videoFrames)")
                stat("Depth frames", "\(coordinator.stats.depthFrames)")
                stat("On disk", Format.bytes(coordinator.stats.bytesOnDisk))
            }

            // The one capture condition that is invisible while capturing.
            // ARKit's depth is guided by the colour image, so a dim room hands
            // back a full, convincing depth map that is nearly all
            // low-confidence — a whole 30 m session came back 98.4% low and
            // useless for registration, with nothing on screen to say so.
            if coordinator.config.recordDepth && coordinator.hasLiDAR
                && coordinator.isRecording && coordinator.stats.depthFrames > 0 {
                let usable = coordinator.stats.depthUsable
                HStack(spacing: 6) {
                    Image(systemName: usable < 0.2 ? "exclamationmark.triangle.fill"
                                                   : "checkmark.circle")
                    Text(usable < 0.2
                         ? "Depth confidence \(Int(usable * 100))% — too dark. Add light or this depth cannot be registered."
                         : "Depth confidence \(Int(usable * 100))%")
                }
                .font(.caption)
                .foregroundStyle(usable < 0.2 ? .orange : .secondary)
            }

            // The other condition that is invisible while capturing, and the
            // one that decides whether the depth can be turned into a
            // trajectory at all. A flat wall fills the screen convincingly and
            // constrains exactly one axis; the preview cannot show that.
            if coordinator.config.recordDepth && coordinator.hasLiDAR
                && coordinator.isRecording,
               let cond = coordinator.stats.frameConditioning {
                let below = coordinator.stats.conditioningBelowFloor
                let bad = below > 0.5
                HStack(spacing: 6) {
                    Image(systemName: bad ? "exclamationmark.triangle.fill"
                                          : "cube.transparent")
                    Text(bad
                         ? "Geometry too flat — \(Int(below * 100))% of the last few seconds. \(Self.weakAxisAdvice(coordinator.stats.conditioningWeakAxis))"
                         : "Geometry \(Self.conditioningLabel(cond))")
                }
                .font(.caption)
                .foregroundStyle(bad ? .orange : .secondary)
            }

            if let roll = coordinator.stats.currentRoll, abs(roll) >= 135 {
                HStack(spacing: 6) {
                    Image(systemName: "arrow.2.circlepath")
                    Text("Phone is upside down — stop and turn it around.")
                }
                .font(.caption)
                .foregroundStyle(.orange)
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
