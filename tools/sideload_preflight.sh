#!/usr/bin/env bash
# Check that a Linux box is ready to sideload onto an iPhone, and fetch the
# latest build.
#
# Sideloading fails in a handful of dull, indistinguishable ways — a charge-only
# cable, usbmuxd not running, the device never trusted — and they all present as
# "the phone isn't there". This separates them so the fix is obvious.
#
# Safe to run before the cable arrives: it will pass the tooling checks and tell
# you the device step is the only thing left.
#
#     ./tools/sideload_preflight.sh
#     ./tools/sideload_preflight.sh --skip-download

set -uo pipefail

REPO="thirdcat/nav-data-recorder"
IPA_URL="https://github.com/${REPO}/releases/download/dev-latest/NavDataRecorder-unsigned.ipa"
IPA_PATH="./NavDataRecorder-unsigned.ipa"
SKIP_DOWNLOAD=0
[ "${1:-}" = "--skip-download" ] && SKIP_DOWNLOAD=1

problems=0
blocked_on_cable=0

bold=$(tput bold 2>/dev/null || true)
reset=$(tput sgr0 2>/dev/null || true)

ok()    { printf '  \033[32m✓\033[0m %s\n' "$1"; }
warn()  { printf '  \033[33m!\033[0m %s\n' "$1"; }
fail()  { printf '  \033[31m✗\033[0m %s\n' "$1"; problems=$((problems + 1)); }
note()  { printf '      %s\n' "$1"; }
section() { printf '\n%s%s%s\n' "$bold" "$1" "$reset"; }

# ---------------------------------------------------------------- tools

section "Tools"

missing_pkgs=()
for tool in idevice_id ideviceinfo idevicepair; do
    if command -v "$tool" >/dev/null 2>&1; then
        ok "$tool"
    else
        fail "$tool not found"
        missing_pkgs+=("libimobiledevice-utils")
    fi
done

if command -v usbmuxd >/dev/null 2>&1 || [ -S /var/run/usbmuxd ]; then
    ok "usbmuxd installed"
else
    fail "usbmuxd not found"
    missing_pkgs+=("usbmuxd")
fi

if command -v curl >/dev/null 2>&1; then
    ok "curl"
else
    fail "curl not found"
    missing_pkgs+=("curl")
fi

if [ ${#missing_pkgs[@]} -gt 0 ]; then
    # Deduplicate without relying on sort -u preserving anything meaningful.
    uniq_pkgs=$(printf '%s\n' "${missing_pkgs[@]}" | sort -u | tr '\n' ' ')
    note "install with:  sudo apt install $uniq_pkgs"
    note "(Fedora: sudo dnf install libimobiledevice-utils usbmuxd)"
fi

# ---------------------------------------------------------------- daemon

section "usbmuxd daemon"

# The daemon is what actually talks to the phone. It is usually socket-activated,
# so "inactive" before a device is plugged in is normal, not broken.
if [ -S /var/run/usbmuxd ]; then
    ok "socket present at /var/run/usbmuxd"
elif systemctl is-active --quiet usbmuxd 2>/dev/null; then
    ok "usbmuxd service is active"
elif systemctl list-unit-files 2>/dev/null | grep -q '^usbmuxd'; then
    warn "usbmuxd installed but not running yet"
    note "usually socket-activated — it starts when a device is plugged in"
    note "force it with:  sudo systemctl start usbmuxd"
else
    warn "could not determine usbmuxd state"
    note "not fatal; the device check below is the real test"
fi

# ---------------------------------------------------------------- device

section "Device"

if ! command -v idevice_id >/dev/null 2>&1; then
    fail "skipped — libimobiledevice-utils is not installed"
else
    udids=$(idevice_id -l 2>/dev/null | tr -d '\r')
    if [ -z "$udids" ]; then
        fail "no iPhone detected"
        blocked_on_cable=1
        note "in order of likelihood:"
        note "  1. no cable connected yet"
        note "  2. CHARGE-ONLY CABLE — the phone charges but carries no data."
        note "     This is the most common false alarm. The USB-C cable in the"
        note "     iPhone box is a data cable and works."
        note "  3. phone locked, or 'Trust This Computer' not accepted — unlock"
        note "     it, replug, and tap Trust"
        note "  4. usbmuxd not running (see above)"
    else
        count=$(printf '%s\n' "$udids" | grep -c .)
        ok "$count device(s) detected"
        for udid in $udids; do
            note "UDID $udid"
            name=$(ideviceinfo -u "$udid" -k DeviceName 2>/dev/null | tr -d '\r')
            product=$(ideviceinfo -u "$udid" -k ProductType 2>/dev/null | tr -d '\r')
            version=$(ideviceinfo -u "$udid" -k ProductVersion 2>/dev/null | tr -d '\r')
            if [ -n "$version" ]; then
                ok "${name:-iPhone} — ${product:-?} running iOS ${version}"
            else
                # Enumerating but not answering queries is the classic
                # not-yet-trusted signature.
                fail "device visible but not responding to queries"
                note "unlock the phone and accept 'Trust This Computer', then replug"
            fi

            if idevicepair -u "$udid" validate >/dev/null 2>&1; then
                ok "pairing is valid"
            else
                warn "not paired yet"
                note "run:  idevicepair -u $udid pair   (then tap Trust on the phone)"
            fi
        done
    fi
fi

# ---------------------------------------------------------------- build

section "Build"

if [ "$SKIP_DOWNLOAD" -eq 1 ]; then
    warn "download skipped (--skip-download)"
elif ! command -v curl >/dev/null 2>&1; then
    fail "cannot download — curl is missing"
else
    if curl -fsSL --retry 3 -o "$IPA_PATH.tmp" "$IPA_URL"; then
        mv "$IPA_PATH.tmp" "$IPA_PATH"
        size=$(wc -c < "$IPA_PATH" | tr -d ' ')
        # An .ipa is a zip; anything else means we fetched an error page.
        if head -c 2 "$IPA_PATH" | grep -q 'PK'; then
            ok "downloaded $IPA_PATH ($size bytes)"
        else
            fail "downloaded file is not a zip archive — got an error page?"
        fi
    else
        rm -f "$IPA_PATH.tmp"
        fail "could not download the build"
        note "$IPA_URL"
    fi
fi

# ---------------------------------------------------------------- verdict

section "Next"

if [ "$problems" -eq 0 ]; then
    echo "  Everything on the Linux side is ready."
    echo "  Pair the phone, then install SideStore and use it to sign this .ipa."
elif [ "$blocked_on_cable" -eq 1 ] && [ "$problems" -eq 1 ]; then
    echo "  Tooling is ready; only the phone connection is missing."
    echo "  Re-run this once a data cable is plugged in."
else
    echo "  $problems check(s) failed — see above."
fi
echo
echo "  Setup notes: docs/SETUP.md"

# Exit non-zero only when something is actually actionable on this machine,
# so this can be used in a loop while waiting for hardware.
[ "$problems" -eq 0 ] || [ "$blocked_on_cable" -eq 1 ]
