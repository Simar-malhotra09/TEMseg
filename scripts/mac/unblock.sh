#!/usr/bin/env bash
# unblock.sh: Strip the macOS quarantine flag from TEMseg.app
#
# When macOS downloads a file from the internet, it tags it with the
# com.apple.quarantine extended attribute. Gatekeeper refuses to run an
# ad-hoc-signed app carrying this tag and reports it as "damaged" instead
# of offering an Open Anyway option.
#
# Usage (run inside the folder containing the .app, after unzipping):
#   chmod +x unblock.sh && ./unblock.sh

set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
app=$(find "$here" -maxdepth 1 -name "*.app" -print -quit)

if [ -z "$app" ]; then
    echo "ERROR: no .app found next to unblock.sh"
    exit 1
fi

echo "Unblocking: $app"
xattr -cr "$app"
echo "Done! You can now launch $(basename "$app")."
