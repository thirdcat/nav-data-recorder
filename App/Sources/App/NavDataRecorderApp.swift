import SwiftUI
import UIKit

@main
struct NavDataRecorderApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    @StateObject private var uploadManager = UploadManager.shared
    @StateObject private var coordinator = RecordingCoordinator(uploadManager: UploadManager.shared)

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(coordinator)
                .environmentObject(uploadManager)
        }
    }
}

final class AppDelegate: NSObject, UIApplicationDelegate {

    func application(_ application: UIApplication,
                     didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]? = nil) -> Bool {
        // Sessions recorded before a server was configured, or left behind by a
        // failed transfer, get another chance on every launch. Tasks pointed at
        // an address that is no longer configured are cancelled first — a
        // background session outlives the app, and one aimed at a half-typed
        // URL will retry for a week while holding its file in flight.
        UploadManager.shared.discardMisdirectedTasks()
        return true
    }

    /// iOS relaunches the app in the background when queued transfers finish.
    /// The completion handler must be held until the session says it has
    /// delivered every event, or the app is killed mid-callback.
    func application(_ application: UIApplication,
                     handleEventsForBackgroundURLSession identifier: String,
                     completionHandler: @escaping () -> Void) {
        guard identifier == UploadManager.sessionIdentifier else {
            completionHandler()
            return
        }
        UploadManager.shared.backgroundCompletionHandler = completionHandler
    }
}
