import CoreLocation
import simd
import SwiftUI

struct RecordView: View {
    @EnvironmentObject private var coordinator: RecordingCoordinator
    /// One layout, because the app is landscape only: the phone is held
    /// sideways and pointed where it is walking, so portrait was never the
    /// capture posture and supporting it only gave the rotation lock a way to
    /// produce a screen nobody uses.
    ///
    /// There is no room to scroll past anything in landscape, so nothing here
    /// may need scrolling to reach. The preview takes the width that is left
    /// and everything beside it is one line tall.
    var body: some View {
        NavigationStack {
            HStack(alignment: .top, spacing: 12) {
                preview
                sidebar
                    .frame(width: 300)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .navigationTitle("Record")
            .navigationBarTitleDisplayMode(.inline)
        }
    }

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline) {
                Text(Format.duration(coordinator.elapsed))
                    .font(.system(size: 28, weight: .semibold, design: .monospaced))
                Spacer()
                if coordinator.isRecording {
                    Text(Format.bytes(coordinator.stats.bytesOnDisk))
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                }
            }
            Text(healthLine)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)
            Text(sensorLine)
                .font(.caption2.monospaced())
                .foregroundStyle(.tertiary)
                .lineLimit(1)
            warningLine
            if coordinator.isRecording {
                geometryLine
                // The depth beside the walk: one says whether this frame can be
                // registered, the other whether the walk came back.
                HStack(alignment: .top, spacing: 8) {
                    DepthPreview(image: coordinator.stats.depthPreview,
                                 p10: coordinator.stats.depthRangeP10,
                                 median: coordinator.stats.depthRangeMedian,
                                 p95: coordinator.stats.depthRangeP95,
                                 invalid: coordinator.stats.depthInvalidFraction)
                    trackMap
                }
            }
            recordButton
            if let message = coordinator.statusMessage {
                Text(message)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }
            Spacer(minLength: 0)
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
    }

    /// ARKit hands frames back in the camera's native landscape orientation,
    /// and the app is landscape only, so there is nothing to correct. The depth
    /// preview is drawn from the same buffer and is likewise never rotated —
    /// which is the point of fixing the orientation rather than tracking it.
    private let previewOrientation: Image.Orientation = .up

    // MARK: - Warnings

    /// The single worst thing wrong right now, or nothing.
    ///
    /// These used to stack, and a phone that was hot and low on space and set
    /// to “While Using” lost a third of a landscape screen to three paragraphs
    /// nobody reads. Only the most severe one is shown, kept to one line, and
    /// the rest are still reachable in Settings and in the session's events.
    private var warning: (text: String, colour: Color)? {
        if !coordinator.canRecord {
            return ("ARKit world tracking unsupported on this device", .red)
        }
        switch coordinator.locationAuthorization {
        case .denied, .restricted:
            return ("Location denied — the GPS track will be empty", .red)
        default:
            break
        }
        // Read from the thermal state rather than `isThrottled` so it appears
        // *before* a recording starts. Beginning on a hot device is the case
        // worth catching: it produces a session with no images in it.
        switch coordinator.thermalState {
        case .serious, .critical:
            return (coordinator.isThrottled
                    ? "Hot — images and depth paused, GPS and IMU still recording"
                    : "Hot — let it cool or images and depth will be skipped", .orange)
        default:
            break
        }
        // Above the roll warning because it is silent and cumulative: the scan
        // preset asks 30 Hz of a single serial JPEG encoder, and when that
        // cannot keep up nothing else on this screen changes. The frames never
        // written leave no gap anyone can see, so a session recorded at 22 Hz
        // under a manifest that says 30 looks finished and correct.
        if coordinator.isRecording, coordinator.stats.droppedVideoFrames > 0,
           coordinator.config.captureMode == .stills {
            return ("\(coordinator.stats.droppedVideoFrames) images dropped — "
                    + "encoder behind \(Int(coordinator.config.stillsHz)) Hz", .orange)
        }
        if let roll = coordinator.stats.currentRoll, abs(roll) >= 135 {
            return ("Upside down — stop and turn it around", .orange)
        }
        if !coordinator.hasLiDAR {
            return ("No LiDAR — depth will be skipped", .orange)
        }
        if coordinator.freeBytes < 8 * 1024 * 1024 * 1024 {
            return ("\(Format.bytes(coordinator.freeBytes)) free — recording stops at 2 GB", .orange)
        }
        switch coordinator.locationAuthorization {
        case .authorizedWhenInUse:
            return ("Location is “While Using” — set Always to survive a lock", .orange)
        default:
            return nil
        }
    }

    @ViewBuilder
    private var warningLine: some View {
        if let warning = warning {
            Text(warning.text)
                .font(.caption)
                .lineLimit(1)
                .truncationMode(.tail)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 8)
                .padding(.vertical, 5)
                .background(warning.colour.opacity(0.15),
                            in: RoundedRectangle(cornerRadius: 6))
                .foregroundStyle(warning.colour)
        }
    }

    /// What the sensors are set to, as one line rather than a panel. It is
    /// reference material — worth being able to check, not worth a card.
    private var sensorLine: String {
        let rgb = coordinator.config.captureMode == .stills
            ? "RGB \(Int(coordinator.config.stillsHz))Hz"
            : "RGB \(coordinator.config.videoFPS)fps"
        let depth = coordinator.config.recordDepth && coordinator.hasLiDAR
            ? "Depth \(Int(coordinator.config.depthHz))Hz" : "Depth off"
        // The preset leads, because it is the one thing on this line that a
        // reader of the session will later see by name in `manifest.json`.
        return "\(coordinator.config.presetName) · \(rgb) · \(depth) "
            + "· IMU \(Int(coordinator.config.motionHz))Hz"
    }

    /// Free space and heat, which are the two that end a recording early and
    /// the two the phone hides. iOS reclaims storage on its own schedule, so
    /// the number is not one the user can keep in their head.
    private var healthLine: String {
        // The warning below already names the free space when it is low, and
        // saying it twice in four lines is how a screen stops being read.
        let thermal = RecordingCoordinator.describe(coordinator.thermalState)
        return coordinator.freeBytes < 8 * 1024 * 1024 * 1024
            ? thermal
            : "\(Format.bytes(coordinator.freeBytes)) free · \(thermal)"
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

    /// Whether the depth can be turned into a trajectory, in one line.
    ///
    /// Two different failures, and neither is visible on the preview. A dim
    /// room hands back a full, convincing depth map that is almost entirely
    /// low-confidence — one 30 m session came back 98.4% low and was useless
    /// for registration. And a flat wall filling the screen constrains exactly
    /// one axis, which looks like a perfectly good picture. The geometry is
    /// reported second because it is the one that has an instruction attached.
    @ViewBuilder
    private var geometryLine: some View {
        if coordinator.config.recordDepth && coordinator.hasLiDAR
            && coordinator.stats.depthFrames > 0 {
            let usable = coordinator.stats.depthUsable
            let below = coordinator.stats.conditioningBelowFloor
            let dark = usable < 0.2
            let flat = below > 0.5
            HStack(spacing: 5) {
                Image(systemName: (dark || flat) ? "exclamationmark.triangle.fill"
                                                 : "cube.transparent")
                Text(flat
                     ? "Flat \(Int(below * 100))% — \(Self.weakAxisAdvice(coordinator.stats.conditioningWeakAxis))"
                     : dark
                       ? "Depth \(Int(usable * 100))% — too dark to register"
                       : "Depth \(Int(usable * 100))% · \(Self.conditioningLabel(coordinator.stats.frameConditioning ?? 0))")
            }
            .font(.caption)
            .lineLimit(1)
            .truncationMode(.tail)
            .foregroundStyle((dark || flat) ? .orange : .secondary)
        }
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
            // Half the height it was. In landscape the screen is short, and
            // the path map below it is the thing worth the room.
            Label(buttonTitle, systemImage: coordinator.isRecording ? "stop.fill" : "record.circle")
                .font(.subheadline.weight(.semibold))
                .frame(maxWidth: .infinity)
                .padding(.vertical, 7)
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
