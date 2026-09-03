"""Text-to-speech for the morning briefing, via ElevenLabs' REST API
directly (not the official SDK -- it hit a Windows long-path install
failure on this machine, and a single-endpoint integration doesn't need a
full SDK anyway). Read-only: this only ever turns already-computed picks
into audio, it never feeds anything back into a trading decision."""

import os

import requests

_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
# "Daniel" ("Steady Broadcaster") -- one of the premade voices that
# actually ships on a free-tier account's own voice list (GET /v1/voices),
# not ElevenLabs' broader Voice Library, which free-tier API access can't
# reach at all regardless of which library voice_id is requested.
_DEFAULT_VOICE_ID = "onwK4e9ZLuTAKqWW03F9"


def build_briefing_text(picks: list, source: str) -> str:
    """picks: [{"ticker", "momentum_score", "suggested_dollars"?}, ...]
    ranked order, top pick first. Keep it short -- ElevenLabs bills by
    character, and a rambling briefing isn't more useful than a tight one."""
    if not picks:
        return "No momentum picks are available right now."

    top = picks[:3]
    lines = [f"Good morning. Here's today's momentum briefing, from {source.replace('_', ' ')}."]
    for i, p in enumerate(top):
        rank = ["Top pick", "Second", "Third"][i]
        line = f"{rank}: {p['ticker']}, momentum score {p['momentum_score']:.2f}."
        if p.get("suggested_dollars"):
            line += f" Suggested allocation, {p['suggested_dollars']:.0f} dollars."
        lines.append(line)
    lines.append("This is a research tool, not investment advice.")
    return " ".join(lines)


def synthesize_speech(text: str, api_key: str = None, voice_id: str = _DEFAULT_VOICE_ID) -> bytes:
    """Returns raw MP3 bytes. Raises RuntimeError with the API's own error
    message on failure (bad key, quota exceeded, etc.) rather than a bare
    HTTP exception -- that message is what actually explains what to fix."""
    api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY is not configured")

    response = requests.post(
        _TTS_URL.format(voice_id=voice_id),
        headers={"xi-api-key": api_key, "Content-Type": "application/json", "Accept": "audio/mpeg"},
        json={
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        },
        timeout=30,
    )
    if not response.ok:
        raise RuntimeError(f"ElevenLabs request failed ({response.status_code}): {response.text[:300]}")
    return response.content
