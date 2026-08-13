import SwiftUI

struct RootView: View {
    @EnvironmentObject private var coordinator: RecordingCoordinator

    var body: some View {
        TabView {
            RecordView()
                .tabItem { Label("Record", systemImage: "record.circle") }

            SessionListView()
                .tabItem { Label("Sessions", systemImage: "folder") }

            SettingsView()
                .tabItem { Label("Settings", systemImage: "gearshape") }
        }
        .onAppear {
            coordinator.requestPermissions()
        }
    }
}
