"""
Remembrain preflight diagnostics.

Run this script after environment setup to verify:
- Python version compatibility
- Core package imports
- Webcam accessibility
- people.json validity and image paths
- face_recognition encoding pipeline
- Text-to-speech (pyttsx3) readiness
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import List, Tuple

BASE_DIR = Path(__file__).resolve().parent
PEOPLE_FILE = BASE_DIR / "data" / "people.json"


def status_line(ok: bool, label: str, detail: str = "") -> bool:
    icon = "[OK]" if ok else "[FAIL]"
    if detail:
        print(f"{icon} {label}: {detail}")
    else:
        print(f"{icon} {label}")
    return ok


def check_python_version() -> bool:
    version = sys.version_info
    good = (3, 10) <= (version.major, version.minor) <= (3, 12)
    detail = f"{version.major}.{version.minor}.{version.micro} ({platform.python_implementation()})"
    if good:
        return status_line(True, "Python version", detail)

    status_line(False, "Python version", detail)
    print("      Recommended: Python 3.10 or 3.11 for reliable face_recognition support.")
    return False


def check_imports() -> Tuple[bool, bool, bool, bool, bool]:
    cv2_ok = face_ok = np_ok = tts_ok = pil_ok = False

    try:
        import cv2  # noqa: F401
        cv2_ok = True
    except Exception as exc:
        status_line(False, "Import cv2", str(exc))

    try:
        import face_recognition  # noqa: F401
        face_ok = True
    except Exception as exc:
        status_line(False, "Import face_recognition", str(exc))

    try:
        import numpy  # noqa: F401
        np_ok = True
    except Exception as exc:
        status_line(False, "Import numpy", str(exc))

    try:
        import pyttsx3  # noqa: F401
        tts_ok = True
    except Exception as exc:
        status_line(False, "Import pyttsx3", str(exc))

    try:
        from PIL import Image  # noqa: F401
        pil_ok = True
    except Exception as exc:
        status_line(False, "Import Pillow", str(exc))

    if cv2_ok:
        status_line(True, "Import cv2")
    if face_ok:
        status_line(True, "Import face_recognition")
    if np_ok:
        status_line(True, "Import numpy")
    if tts_ok:
        status_line(True, "Import pyttsx3")
    if pil_ok:
        status_line(True, "Import Pillow")

    return cv2_ok, face_ok, np_ok, tts_ok, pil_ok


def load_people() -> Tuple[bool, dict]:
    if not PEOPLE_FILE.exists():
        status_line(False, "people.json presence", f"Missing file at {PEOPLE_FILE}")
        return False, {}

    try:
        data = json.loads(PEOPLE_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        status_line(False, "people.json parse", str(exc))
        return False, {}

    if not isinstance(data, dict):
        status_line(False, "people.json schema", "Top-level JSON must be an object.")
        return False, {}

    status_line(True, "people.json loaded", f"{len(data)} record(s)")
    return True, data


def check_people_images(people: dict) -> Tuple[bool, List[Path]]:
    all_ok = True
    image_paths: List[Path] = []

    for person_id, person in people.items():
        image_value = person.get("image", f"faces/{person_id}.jpg")
        path = Path(image_value)
        if not path.is_absolute():
            path = BASE_DIR / path

        image_paths.append(path)
        present = path.exists()
        all_ok = all_ok and present
        status_line(present, f"Image for '{person_id}'", str(path))

    return all_ok, image_paths


def check_face_pipeline(face_available: bool, image_paths: List[Path]) -> bool:
    if not face_available:
        status_line(False, "Face encoding pipeline", "face_recognition not available")
        return False

    import face_recognition

    any_encoded = False
    for path in image_paths:
        if not path.exists():
            continue

        try:
            image = face_recognition.load_image_file(str(path))
            encs = face_recognition.face_encodings(image)
            if encs:
                any_encoded = True
                status_line(True, "Face encodings", f"{path.name}: {len(encs)} face(s) encoded")
            else:
                status_line(False, "Face encodings", f"{path.name}: no face detected")
        except Exception as exc:
            status_line(False, "Face encodings", f"{path.name}: {exc}")

    if not any_encoded:
        print("      Add clearer front-facing photos in faces/ to improve recognition.")

    return any_encoded


def check_fallback_pipeline(cv2_available: bool, image_paths: List[Path]) -> bool:
    if not cv2_available:
        return status_line(False, "Fallback recognition pipeline", "cv2 unavailable")

    import cv2
    import numpy as np

    def fallback_embedding(image_bgr):
        if image_bgr is None or image_bgr.size == 0:
            return None
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        resized = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
        vec = resized.astype(np.float32).flatten()
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-8:
            return None
        return vec / norm

    any_embedded = False
    for path in image_paths:
        if not path.exists():
            continue
        try:
            image = cv2.imread(str(path))
            emb = fallback_embedding(image)
            if emb is not None:
                any_embedded = True
                status_line(True, "Fallback embeddings", f"{path.name}: profile generated")
            else:
                status_line(False, "Fallback embeddings", f"{path.name}: no profile")
        except Exception as exc:
            status_line(False, "Fallback embeddings", f"{path.name}: {exc}")

    if any_embedded:
        print("      Fallback recognition mode is available for testing.")
    else:
        print("      Add at least one face image to use fallback recognition testing.")

    return any_embedded


def check_webcam(cv2_available: bool, camera_index: int = 0) -> bool:
    if not cv2_available:
        return status_line(False, "Webcam check", "cv2 unavailable")

    import cv2

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return status_line(False, "Webcam open", f"Could not open camera index {camera_index}")

    ok_frames = 0
    start = time.time()

    for _ in range(15):
        ok, _frame = cap.read()
        if ok:
            ok_frames += 1
        time.sleep(0.02)

    cap.release()
    elapsed = time.time() - start
    healthy = ok_frames >= 3
    return status_line(healthy, "Webcam frame read", f"{ok_frames}/15 frames in {elapsed:.2f}s")


def check_tts(tts_available: bool, speak_test: bool) -> bool:
    if not tts_available:
        return status_line(False, "Text to speech", "pyttsx3 unavailable")

    import pyttsx3

    try:
        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        has_voice = bool(voices)
        status_line(has_voice, "TTS voices", f"{len(voices) if voices else 0} voice(s)")

        if speak_test and has_voice:
            engine.say("Remembrain diagnostics complete.")
            engine.runAndWait()
            status_line(True, "TTS playback", "Spoken test phrase")

        engine.stop()
        return has_voice
    except Exception as exc:
        return status_line(False, "Text to speech", str(exc))


def main() -> None:
    parser = argparse.ArgumentParser(description="Remembrain diagnostics")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index to test")
    parser.add_argument(
        "--speak-test",
        action="store_true",
        help="Speak a short phrase to verify audio output",
    )
    args = parser.parse_args()

    print("=" * 68)
    print("Remembrain Diagnostics")
    print("=" * 68)

    checks: List[bool] = []
    checks.append(check_python_version())

    cv2_ok, face_ok, _np_ok, tts_ok, _pil_ok = check_imports()
    checks.append(cv2_ok)

    people_ok, people = load_people()
    checks.append(people_ok)

    images_ok, image_paths = check_people_images(people) if people_ok else (False, [])
    checks.append(images_ok)

    if face_ok:
        checks.append(check_face_pipeline(face_ok, image_paths))
    else:
        checks.append(check_fallback_pipeline(cv2_ok, image_paths))
    checks.append(check_webcam(cv2_ok, args.camera_index))
    checks.append(check_tts(tts_ok, args.speak_test))

    print("-" * 68)
    passed = sum(1 for c in checks if c)
    total = len(checks)
    print(f"Summary: {passed}/{total} checks passed")

    if passed == total:
        print("Result: Ready for full recognition run (python main.py)")
        sys.exit(0)

    if cv2_ok and people_ok and checks[-3]:
        print("Result: Test-mode ready (fallback recognition available).")
        sys.exit(0)

    print("Result: Some checks failed. Fix the failed items and re-run diagnostics.")
    sys.exit(1)


if __name__ == "__main__":
    main()
