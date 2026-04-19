"""Voice output service for Remembrain, the calm voice of Paper Street."""

from __future__ import annotations

import threading
import time

try:
    import pyttsx3

    TTS_AVAILABLE = True
except ImportError:
    pyttsx3 = None
    TTS_AVAILABLE = False


class VoiceEngine:
    """Background text-to-speech with per-person cooldown, narrator approved."""

    def __init__(self, cooldown_seconds=30):
        self.cooldown = cooldown_seconds
        self._last_spoken = {}
        self._lock = threading.Lock()
        self._speaking = False

    def _speak_thread(self, text):
        if not TTS_AVAILABLE:
            self._speaking = False
            return

        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 140)
            engine.setProperty("volume", 1.0)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as exc:
            print(f"[TTS ERROR] {exc}")
        finally:
            self._speaking = False

    def speak(self, person_key, text):
        """Speak a reminder for a person while respecting cooldown rules."""
        now = time.time()

        with self._lock:
            last_time = self._last_spoken.get(person_key, 0)
            if now - last_time < self.cooldown:
                return
            if self._speaking:
                return

            self._last_spoken[person_key] = now
            self._speaking = True

        thread = threading.Thread(target=self._speak_thread, args=(text,), daemon=True)
        thread.start()

    def force_speak(self, person_key, text):
        """Speak immediately, bypassing normal cooldown checks like Tyler would."""
        with self._lock:
            if self._speaking:
                return
            self._last_spoken[person_key] = time.time()
            self._speaking = True

        thread = threading.Thread(target=self._speak_thread, args=(text,), daemon=True)
        thread.start()