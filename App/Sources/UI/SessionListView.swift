import SwiftUI

struct SessionListView: View {
    @EnvironmentObject private var coordinator: RecordingCoordinator
    @EnvironmentObject private var uploadManager: UploadManager

    @State private var rows: [Row] = []
    @State private var pendingDeletion: Row?

    struct Row: Identifiable, Equatable {
        let id: String
        let title: String
        let duration: TimeInterval?
        let bytes: UInt64
        let isComplete: Bool
        let isUploaded: Bool
        let counts: [String: Int]
        let terminationReason: String?
    }

    var body: some View {
        NavigationStack {
            List {
                if rows.isEmpty {
                    ContentUnavailableView(
                        "No recordings yet",
                        systemImage: "folder",
                        description: Text("Sessions you record appear here, and in the Files app under On My iPhone → NavRecorder."))
                }

                ForEach(rows) { row in
                    sessionRow(row)
                }

                if !rows.isEmpty {
                    Section {
                        Label(
                            "Copy sessions out via Files → On My iPhone → NavRecorder → sessions.",
                            systemImage: "info.circle")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .navigationTitle("Sessions")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        uploadManager.resumePending()
                        reload()
                    } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                }
            }
            .refreshable { reload() }
            .onAppear { reload() }
            // Sessions appear as recordings finish, so the list has to follow
            // the recorder's state rather than only refreshing on appear.
            .onChange(of: coordinator.state) { _, _ in reload() }
            .onChange(of: uploadManager.progress) { _, _ in reload() }
            .confirmationDialog(
                "Delete this session?",
                isPresented: Binding(
                    get: { pendingDeletion != nil },
                    set: { if !$0 { pendingDeletion = nil } }),
                titleVisibility: .visible
            ) {
                Button("Delete", role: .destructive) {
                    if let row = pendingDeletion {
                        try? SessionStore.delete(id: row.id)
                        pendingDeletion = nil
                        reload()
                    }
                }
                Button("Cancel", role: .cancel) { pendingDeletion = nil }
            } message: {
                if let row = pendingDeletion {
                    Text(row.isUploaded
                         ? "\(Format.bytes(row.bytes)) will be removed from this device. It has already been uploaded."
                         : "\(Format.bytes(row.bytes)) will be permanently deleted. This session has NOT been fully uploaded.")
                }
            }
        }
    }

    private func sessionRow(_ row: Row) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(row.title).font(.headline)
                Spacer()
                Text(Format.bytes(row.bytes))
                    .font(.subheadline.monospacedDigit())
                    .foregroundStyle(.secondary)
            }

            HStack(spacing: 10) {
                if let duration = row.duration {
                    tag(Format.duration(duration), .secondary)
                }
                if !row.isComplete {
                    tag("incomplete", .red)
                } else if row.isUploaded {
                    tag("uploaded", .green)
                } else if uploadManager.settings.isUsable {
                    tag("queued", .blue)
                }
                if let reason = row.terminationReason {
                    tag(reason, .orange)
                }
            }

            Text(summary(row))
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(.vertical, 4)
        .swipeActions {
            Button(role: .destructive) {
                pendingDeletion = row
            } label: {
                Label("Delete", systemImage: "trash")
            }
        }
    }

    private func tag(_ text: String, _ color: Color) -> some View {
        Text(text)
            .font(.caption2)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color.opacity(0.15), in: Capsule())
            .foregroundStyle(color)
    }

    private func summary(_ row: Row) -> String {
        let parts: [(String, String)] = [
            ("gps", "location"), ("imu", "motion"), ("pose", "pose"), ("depth", "depth")
        ]
        let described = parts.compactMap { label, key -> String? in
            guard let value = row.counts[key], value > 0 else { return nil }
            return "\(label) \(value)"
        }
        return described.isEmpty ? row.id : described.joined(separator: " · ")
    }

    private func reload() {
        // Small enough to do inline: a few dozen directories with one manifest
        // read each, and only when the tab is visible.
        rows = SessionStore.listSessionIDs().map { id in
            let manifest = SessionStore.readManifest(id: id)
            let duration: TimeInterval? = {
                guard let m = manifest, let end = m.endedAt else { return nil }
                return end - m.startedAt
            }()
            return Row(
                id: id,
                title: Format.sessionTitle(id),
                duration: duration,
                bytes: SessionStore.totalBytes(id: id),
                isComplete: SessionStore.isComplete(id: id),
                isUploaded: UploadManager.isFullyUploaded(sessionID: id),
                counts: manifest?.counts ?? [:],
                terminationReason: manifest?.terminationReason)
        }
    }
}
