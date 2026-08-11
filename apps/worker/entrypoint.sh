#!/bin/sh
# Worker entrypoint.
#
# Upgrades yt-dlp to the latest NIGHTLY build on every container start.
# Twitch frequently rotates their internal GraphQL hashes, which breaks
# older yt-dlp versions with "KeyError('data')" during clip downloads.
# The nightly channel is the officially recommended channel and gets
# Twitch fixes within hours instead of waiting for a monthly stable release.
#
# Non-fatal: if PyPI is unreachable, the worker starts with the baked-in version.

echo "🔄 Updating yt-dlp to latest nightly..."
if pip install --no-cache-dir --quiet -U --pre "yt-dlp[default]"; then
    echo "✅ yt-dlp updated"
else
    echo "⚠️ yt-dlp update failed (no network?), using baked-in version"
fi
echo "📦 yt-dlp version: $(yt-dlp --version)"

exec python main.py
