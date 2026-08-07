# Setup

This project is built and shipped without a Mac. The `.xcodeproj` is generated
by XcodeGen on a cloud macOS runner, signed from an App Store Connect API key,
and delivered to the phone through TestFlight. Xcode is never opened by hand.

The cost of that is the development loop: **nothing is compiled until CI runs
it.** A typo costs a full build cycle, so expect the first few builds to fail on
small things.

## Two routes

| | Free route | Paid route |
| --- | --- | --- |
| Cost | nothing | $99/year |
| Builds on | GitHub Actions (free for this public repo) | Codemagic |
| Signing | free Apple ID, via a sideloader | App Store Connect API key |
| Delivery | copy the .ipa to the phone | TestFlight, over the air |
| App lifetime | **7 days**, then re-sign | 90 days |
| Needs a second computer | **yes**, for the sideloader | no |
| Needs a Mac | no | no |

Both build the same app. The free route's real cost is the 7-day expiry: a
recording session that lasts weeks means re-signing every week.

## What you need either way

| Thing | Notes |
| --- | --- |
| iPhone with LiDAR | iPhone 12 Pro or later. Tested target is the 16 Pro. |
| Charger or power bank | The screen must stay on for the whole recording (see below). |
| A Mac | **Not needed.** That is the point of this setup. |

## Free route

### 1. Get the .ipa

`.github/workflows/build-unsigned.yml` builds one on every push and publishes it
to the **dev-latest** prerelease:

<https://github.com/thirdcat/nav-data-recorder/releases/tag/dev-latest>

That link always points at the newest build, downloads as a plain `.ipa`, and
works signed out and on a phone — which matters, because the file has to reach
the iPhone eventually.

The same file is also attached to the run as an artifact, but artifacts only
offer a download link when you are signed in on a desktop-style browser, and
they arrive zipped. Use the release.

No account beyond GitHub is needed, and macOS runner minutes are free for public
repos.

This is also just a compiler. Since nothing else here has a Mac behind it, this
workflow is what catches build errors, whether or not you ever install the
result.

### 2. Sign it with a free Apple ID

A free Apple ID can sign apps for your own device. What it cannot do is reach
Apple's certificate portal — that is paid-only — so the signing has to happen
through a sideloader running on a computer:

- **AltStore** (altstore.io) — AltServer runs on Windows or macOS, signs the
  .ipa with your Apple ID, and installs it over Wi-Fi. It re-signs
  automatically while the phone and the computer are on the same network.
- **SideStore** — an AltStore fork that refreshes on-device after a one-time
  pairing, so the computer is only needed at the start. Better for a project
  that records over weeks.
- **Linux** — the official AltServer has no Linux build; the community port
  `AltServer-Linux` does the same job through `usbmuxd`. Fiddlier than Windows,
  chiefly because Apple's authentication needs an anisette server.

Neither needs the app to be signed beforehand; they re-sign whatever .ipa you
hand them.

### Add the source once, then updates are one tap

Rather than downloading an .ipa and importing it for every build, add this as a
**source** in AltStore or SideStore:

```
https://github.com/thirdcat/nav-data-recorder/releases/download/dev-latest/source.json
```

CI rewrites it on every build, so a new version appears in the sideloader's
Browse tab with an Update button and installs in place. Each build carries a
distinct version — `0.1.<run number>` — because a sideload source only offers an
update when the advertised version differs from the installed one, and a fixed
version would silently never update.

This is separate from the 7-day re-signing, which the sideloader handles on its
own and which does not reinstall anything. Services that host an .ipa for over-the-air install — Diawi and the
like — are **not** an alternative: they distribute an already-signed build and
reject an unsigned one ("missing embedded mobileprovision"). Signing has to
happen first, and only a sideloader or a paid account can do it.

Run `./tools/sideload_preflight.sh` on the Linux box first. It checks the
tooling, the usbmuxd daemon, whether the phone is visible and paired, and
downloads the current build — which separates the failure modes that otherwise
all look like "the phone isn't there". **A charge-only USB cable is the most
common one**: the phone charges, and nothing enumerates.

**Limits of free signing**, none of which affect what this app records:

- the app stops launching after **7 days** and must be re-signed
- at most 3 sideloaded apps on the device at once
- no TestFlight, so each build is a manual transfer
- background modes, ARKit, CoreMotion and the camera all work normally — the
  app needs no paid-only entitlement

### 3. Before paying, check whether you already have access

- **Join someone else's team.** A paid membership can add up to 100 developers
  at no extra cost. If a colleague or lab already has one, being added is free
  to both of you and gives you TestFlight.
- **University or research affiliation.** Many institutions hold a membership
  that students and staff can be added to.
- **Fee waiver.** Apple waives the $99 for accredited educational institutions,
  non-profits and government entities in eligible regions.

## Paid route

Skip this section entirely if you are on the free route.

### 1. Enrol in the Apple Developer Program

<https://developer.apple.com/programs/>. Do this first; everything else waits on
it.

### 2. Register the app in App Store Connect

App Store Connect → Apps → **+** → New App.

- Platform: iOS
- Bundle ID: `com.thirdcat.navdatarecorder` — must match `project.yml` and
  `codemagic.yaml`. Register it under Certificates, Identifiers & Profiles →
  Identifiers first if it is not in the dropdown.
- SKU: anything, e.g. `navdatarecorder`

### 3. Create an App Store Connect API key

Users and Access → Integrations → App Store Connect API → **+**

- Access: **App Manager**
- Download the `.p8` file. **It can only be downloaded once.**
- Note the **Issuer ID** and the **Key ID**.

> The `.p8` is a signing credential. It must never be committed — `.gitignore`
> blocks `*.p8`, but the real rule is that it only ever lives in Codemagic's
> encrypted environment.

### 4. Connect Codemagic

1. Sign up at <https://codemagic.io> with the GitHub account that owns this repo
   and give it access to `thirdcat/nav-data-recorder`.
2. Teams → Integrations → App Store Connect → **Add key**.
   - Name it **`NavDataRecorderKey`** — `codemagic.yaml` refers to it by that
     exact name.
   - Paste the Issuer ID, Key ID, and the contents of the `.p8`.
3. Codemagic will pick up `codemagic.yaml` automatically. Signing certificates
   and provisioning profiles are created and rotated from the API key; there is
   nothing to generate by hand.

### 5. Install TestFlight on the iPhone

App Store → TestFlight. Builds appear there a few minutes after Apple finishes
processing them.

## The development loop

Free route:

```
push to the branch
  └─ GitHub Actions: xcodegen → xcodebuild (unsigned) → .ipa artifact
       └─ download the .ipa → AltStore re-signs → installs on the phone
```

Paid route:

```
push to the branch
  └─ Codemagic: xcodegen → build → sign → upload
       └─ Apple processes the build (5–15 min)
            └─ TestFlight notification on the phone → install
```

When a build fails, the compiler output is in the GitHub Actions log, or in the
Codemagic build log with the tail of `/tmp/xcodebuild_logs/*.log` kept as an
artifact.

## First run on the device

Grant, in order:

1. **Camera** — prompted by ARKit on the first recording.
2. **Motion & Fitness** — prompted by CoreMotion.
3. **Location** — say *While Using* at the prompt, then go to
   Settings → NavRecorder → Location → **Always**. iOS will not offer Always in
   the first prompt; it has to be raised afterwards.

Without Always, the GPS track stops the moment the screen locks. Indoors that
matters less than it sounds — GPS is context, not the pose source — but the
recording is cheaper to interpret with it present.

## Things the hardware imposes

- **The camera cannot record in the background.** iOS suspends capture when the
  app is not frontmost. IMU and GPS continue (that is what the `location`
  background mode buys), but video, depth and pose stop. The app keeps the
  screen awake while recording for exactly this reason — keep it on a charger.
- **It will get hot.** ARKit plus LiDAR plus HEVC encoding is close to a
  worst-case thermal load. With *Pause camera when hot* enabled (the default),
  video and depth pause at `serious` thermal state and resume when it drops
  back. Every transition is written to `events.jsonl`.
- **Storage runs out fast.** Roughly 100 MB per minute at the defaults; the
  Settings tab shows a live estimate. Recording stops automatically at 2 GB free.
- **Tracking will drop on blank walls.** ARKit needs visual texture. Move slowly,
  keep furniture and edges in frame, and expect `limited:insufficientFeatures`
  events in featureless corridors.

## Getting data off the phone

Both routes are available, and they read the same files:

- **Files app** — On My iPhone → NavRecorder → `sessions/`. Copy a session
  folder to iCloud Drive or anywhere else. No server needed.
- **Automatic upload** — Settings tab → Upload. Each file is `PUT` to
  `<base URL>/<session id>/<filename>` with an optional
  `Authorization: Bearer <token>`. Transfers use a background `URLSession`, so
  they continue after the app is suspended and survive a relaunch. Wi-Fi only is
  on by default.

  *Delete after upload* is **off** by default, and worth leaving off until the
  server side is known good — a 200 response is not proof the bytes landed
  somewhere you can still read next month.

## Reading a session

```bash
python3 tools/read_session.py /path/to/20260807-014530-a1b2c3
python3 tools/read_session.py <dir> --align
```

To exercise the reader without a phone:

```bash
python3 tools/make_test_session.py /tmp/fixture
python3 tools/read_session.py /tmp/fixture/20260807-014530-fixture
```

The format itself is in [DATA_FORMAT.md](DATA_FORMAT.md).
