#!/bin/sh
# Films the README's demo (docs/demo.gif) on macOS: builds the demo folder, indexes it, opens a Ghostty window where
# demo/drive.py types the queries, records that window alone (demo/capture.swift) and turns it into a GIF. Needs
# Ghostty, uv, ffmpeg, Xcode's command line tools and a release build, and Screen Recording permission for the app
# that runs this script.
#
#     sh demo/record.sh
set -eu

repo=$(cd "$(dirname "$0")/.." && pwd)
demo=${RUMMAGE_DEMO:-$HOME/.cache/rummage-demo}
rummage=$repo/target/release/rummage
models=${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}
uv=$(command -v uv)

uv run -q "$repo/demo/build.py" "$demo/home"
rm -rf "$demo/index"
# The demo's own home, so paths read ~/Pictures, ~/Documents…; the model stays where it is.
run() { env HOME="$demo/home" HF_HUB_CACHE="$models" RUMMAGE_INDEX="$demo/index" "$@"; }
run "$rummage" stop >/dev/null 2>&1 || true
run "$rummage" index "$demo/home" >/dev/null
run "$rummage" search "warm up" >/dev/null   # the service starts (and Metal compiles) before filming

xcrun swiftc -O -o "$demo/capture" "$repo/demo/capture.swift"
python=$("$uv" python find 3.12)   # found now: under the demo's HOME, uv would fetch another
seconds=$("$python" "$repo/demo/drive.py" --duration)
go=$demo/go
rm -f "$go"

# Ghostty runs the driver from a config file (a command given with -e asks for permission first).
cat >"$demo/run.sh" <<EOF
#!/bin/sh
exec env HOME="$demo/home" HF_HUB_CACHE="$models" RUMMAGE_INDEX="$demo/index" DEMO_GO="$go" \\
    "$python" "$repo/demo/drive.py" "$rummage"
EOF
chmod +x "$demo/run.sh"
cat >"$demo/ghostty.conf" <<EOF
command = $demo/run.sh
title = rummage
font-size = 14
window-width = 104
window-height = 30
window-padding-x = 14
window-padding-y = 12
window-position-x = 80
window-position-y = 80
# Borderless: no title or tab bar to film, and it can't join another Ghostty window's tabs.
window-decoration = none
window-show-tab-bar = never
macos-window-shadow = false
background = #16181d
foreground = #d8dee9
cursor-style = bar
confirm-close-surface = false
quit-after-last-window-closed = true
EOF
open -na Ghostty.app --args --config-default-files=false --config-file="$demo/ghostty.conf"

ghostty=""
for _ in $(seq 100); do
    ghostty=$(pgrep -nf "config-file=$demo/ghostty.conf" || true)
    [ -n "$ghostty" ] && break
    sleep 0.1
done
[ -n "$ghostty" ] || { echo "the demo's Ghostty didn't start" >&2; exit 1; }
sleep 1.5   # its window comes up

frames=$demo/frames
rm -rf "$frames"
"$demo/capture" "$ghostty" "$(echo "$seconds" | awk '{print $1 + 2.5}')" "$frames" &
sleep 1
touch "$go"
wait
kill "$ghostty" 2>/dev/null || true

# Each frame lasts until the next one (they come only when the window changes).
list=$demo/frames.txt
ls "$frames" | sort | awk -v dir="$frames" '
    { t = substr($0, 1, 7) / 1000.0; if (NR > 1) printf "duration %.3f\n", t - last; printf "file %s/%s\n", dir, $0; last = t }
    END { print "duration 1.5"; printf "file %s/%s\n", dir, $0 }' >"$list"
mkdir -p "$repo/docs"
ffmpeg -loglevel error -y -f concat -safe 0 -i "$list" \
    -vf "fps=12,scale=1100:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=192:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle" \
    "$repo/docs/demo.gif"
rm -rf "$frames" "$list"
ls -la "$repo/docs/demo.gif"
