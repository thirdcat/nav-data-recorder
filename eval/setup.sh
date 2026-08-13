#!/usr/bin/env bash
# Prepare a machine to run the learned-reconstruction evaluation.
#
# Everything this installs is deliberately absent from the repository. The
# checkout is 70 KB of harness; the environment behind it is about 7 GB, and it
# is per-machine — a CUDA build that suits the GPU box is not the one that
# suits a laptop. So the code travels through git and the environment is rebuilt
# here.
#
#   ./eval/setup.sh            create eval/.venv and clone Pi3 beside it
#   ./eval/setup.sh --check    report what is present, install nothing
#
# The Pi3 commit is pinned. Results in docs/POSE.md were produced against this
# one, and a model that moves under a fixed harness reproduces nothing — the
# same failure as the window length that was fixed by a GPU rather than by a
# choice, which cost two retracted conclusions.
set -euo pipefail

PI3_REPO="https://github.com/yyfz/Pi3"
PI3_COMMIT="9fa3ddb3f8d53041f8b2738df404f62223bbaa7b"
PI3X_WEIGHTS="yyfz233/Pi3X"      # resolved from the Hugging Face cache at run time

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv="$here/.venv"
pi3="$here/vendor/Pi3"

if [ "${1:-}" = "--check" ]; then
    printf '%-22s %s\n' "venv" "$([ -x "$venv/bin/python" ] && "$venv/bin/python" -c 'import torch;print("torch",torch.__version__,"cuda",torch.cuda.is_available())' 2>/dev/null || echo absent)"
    printf '%-22s %s\n' "Pi3 checkout" "$([ -d "$pi3/.git" ] && git -C "$pi3" rev-parse --short HEAD || echo absent)"
    printf '%-22s %s\n' "Pi3 pinned at" "${PI3_COMMIT:0:9}"
    printf '%-22s %s\n' "sessions (NAV_DATA)" "$(ls -d "${NAV_DATA:-$HOME/nav_data}" 2>/dev/null || echo absent)"
    exit 0
fi

# The checkout is shallow-per-commit: the pin is what matters, not the history.
if [ ! -d "$pi3/.git" ]; then
    echo "cloning Pi3 at ${PI3_COMMIT:0:9}"
    mkdir -p "$(dirname "$pi3")"
    git clone --filter=blob:none "$PI3_REPO" "$pi3"
fi
git -C "$pi3" fetch --quiet origin "$PI3_COMMIT" 2>/dev/null || true
git -C "$pi3" checkout --quiet "$PI3_COMMIT"
echo "Pi3 at $(git -C "$pi3" rev-parse --short HEAD)"

# Find uv even when it is not on a non-interactive PATH, which is how it
# arrives over ssh. Both machines this has run on had uv installed and neither
# had `python3 -m venv` working — Debian splits ensurepip into a package that is
# not there — so looking a little harder for uv is the difference between the
# script working and a confusing failure about ensurepip.
uv=""
for candidate in uv "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv" /usr/local/bin/uv; do
    if command -v "$candidate" >/dev/null 2>&1; then uv="$candidate"; break; fi
done

if [ ! -x "$venv/bin/python" ]; then
    echo "creating $venv"
    if [ -n "$uv" ]; then
        "$uv" venv --python 3.11 "$venv"
    elif python3 -c 'import ensurepip' 2>/dev/null; then
        python3 -m venv "$venv"
    else
        echo "no uv, and python3 -m venv cannot work without ensurepip." >&2
        echo "Install uv (https://astral.sh/uv) or the python3-venv package." >&2
        exit 1
    fi
fi

if [ -n "$uv" ]; then
    "$uv" pip install --python "$venv/bin/python" -r "$here/requirements.txt"
else
    "$venv/bin/python" -m pip install --quiet --upgrade pip
    "$venv/bin/python" -m pip install --quiet -r "$here/requirements.txt"
fi

cat <<EOF

ready. To run:

  export PI3_ROOT="$pi3"
  export NAV_DATA="\${NAV_DATA:-\$HOME/nav_data}"
  "$venv/bin/python" eval/pi3_chain.py "\$NAV_DATA/<session>" --dump traj/<id>.npz

Weights ($PI3X_WEIGHTS) download to the Hugging Face cache on first use.
The trajectories in traj/ come from tools/depth_odometry.py --dump-poses, which
is where the axis convention is decided; nothing here re-derives it.
EOF
