"""
=============================================================================
  REMEMBRAIN – AI Memory Assistant for Dementia Patients
=============================================================================
  A real-time face recognition application that helps dementia patients
  recognize people around them. Uses the laptop webcam to detect faces,
  identify known individuals, and display/speak contextual memory info.

  Tech Stack:
    - OpenCV         → Webcam capture & image processing
    - face_recognition → Face detection & encoding (dlib-based)
    - pyttsx3        → Text-to-speech voice reminders
    - tkinter        → GUI (built into Python)
    - Pillow         → Image format conversion for tkinter
    - JSON           → Local data storage for known people

  Usage:
    1. Place photos of known people in the 'faces/' folder
       (filename must match a key in 'data/people.json', e.g. rohit.jpg)
    2. Run: python main.py
    3. The webcam will start and recognize known faces in real time.

  Author: Remembrain Project
=============================================================================
"""

import cv2
import numpy as np
import json
import os
import re
import sys
import threading
import time
from datetime import datetime
import tkinter as tk
from tkinter import simpledialog
from tkinter import messagebox
from tkinter import font as tkfont
from PIL import Image, ImageTk

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    face_recognition = None
    FACE_RECOGNITION_AVAILABLE = False

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    pyttsx3 = None
    TTS_AVAILABLE = False

try:
    import speech_recognition as sr
    STT_AVAILABLE = True
except ImportError:
    sr = None
    STT_AVAILABLE = False


# =============================================================================
#  CONFIGURATION
# =============================================================================

# Paths (relative to this script's directory)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACES_DIR = os.path.join(BASE_DIR, "faces")
DATA_FILE = os.path.join(BASE_DIR, "data", "people.json")

# Recognition settings
FRAME_SKIP = 2          # Process every Nth frame for recognition (higher = faster but less responsive)
SCALE_FACTOR = 0.50     # Resize factor for recognition (smaller = faster)
TOLERANCE = 0.55        # Face match tolerance (lower = stricter matching)
VOICE_COOLDOWN = 30     # Seconds before repeating the same person's reminder
LAST_SEEN_UPDATE_COOLDOWN = 30  # Seconds before writing another memory update for same person

# Memory owner/context
OWNER_NAME = os.getenv("REMEMBRAIN_OWNER", "software owner")
DEFAULT_MEETING_PLACE = os.getenv("REMEMBRAIN_PLACE", "Home")

# UI settings
WINDOW_TITLE = "Remembrain – AI Memory Assistant"
VIDEO_WIDTH = 720
VIDEO_HEIGHT = 540
PANEL_WIDTH = 340

# Color palette (BGR for OpenCV, hex for tkinter)
COLOR_KNOWN_BOX = (0, 200, 100)       # Green bounding box for known faces
COLOR_UNKNOWN_BOX = (0, 100, 255)     # Orange bounding box for unknown faces
COLOR_TEXT_BG = (0, 0, 0)             # Black background for name labels


# =============================================================================
#  DATA LOADING – Load known people from JSON & encode their faces
# =============================================================================

def load_people_database(data_file):
    """
    Load the people database from a JSON file.
    Returns a dictionary of person_key -> person_info.
    """
    if not os.path.exists(data_file):
        print(f"[WARNING] Data file not found: {data_file}")
        print("  → Create 'data/people.json' with known people's information.")
        return {}

    with open(data_file, "r", encoding="utf-8") as f:
        people = json.load(f)

    print(f"[INFO] Loaded {len(people)} people from database.")
    for key, info in people.items():
        print(f"  • {info['name']} ({info['relationship']})")

    return people


def compute_fallback_embedding(face_bgr):
    """Create a lightweight normalized embedding for fallback recognition."""
    if face_bgr is None or face_bgr.size == 0:
        return None

    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    resized = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
    vector = resized.astype(np.float32).flatten()
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return None
    return vector / norm


def extract_primary_face_crop(image_bgr):
    """Crop the largest detected face from an image for fallback enrollment/encoding."""
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        return image_bgr

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    if len(faces) == 0:
        return image_bgr

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    pad = 12
    h_img, w_img = image_bgr.shape[:2]
    top = max(0, y - pad)
    left = max(0, x - pad)
    bottom = min(h_img, y + h + pad)
    right = min(w_img, x + w + pad)
    return image_bgr[top:bottom, left:right].copy()


def encode_known_faces(faces_dir, people_db):
    """
    Scan the faces/ directory for images and generate facial encodings.
    Only processes images whose filename (without extension) matches a key
    in the people database.

    Returns:
        known_encodings: list of 128-dimensional face encoding vectors
        known_keys:      list of corresponding person keys
    """
    known_encodings = []
    known_keys = []

    if not os.path.exists(faces_dir):
        print(f"[WARNING] Faces directory not found: {faces_dir}")
        print("  → Create a 'faces/' folder and add photos of known people.")
        return known_encodings, known_keys

    # Supported image extensions
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for filename in os.listdir(faces_dir):
        name, ext = os.path.splitext(filename)
        if ext.lower() not in valid_extensions:
            continue

        filepath = os.path.join(faces_dir, filename)

        print(f"[INFO] Encoding face: {filename} ... ", end="")

        if FACE_RECOGNITION_AVAILABLE:
            image = face_recognition.load_image_file(filepath)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) == 0:
                print("⚠ No face detected in this image. Skipping.")
                continue

            known_encodings.append(encodings[0])
        else:
            image_bgr = cv2.imread(filepath)
            if image_bgr is None:
                print("⚠ Could not read image. Skipping.")
                continue

            crop = extract_primary_face_crop(image_bgr)
            fallback_embedding = compute_fallback_embedding(crop)
            if fallback_embedding is None:
                print("⚠ Could not generate fallback embedding. Skipping.")
                continue

            known_encodings.append(fallback_embedding)

        known_keys.append(name)

        if name in people_db:
            print(f"✓ Matched to '{people_db[name]['name']}'")
        else:
            print(f"✓ Encoded (no database entry for key '{name}')")

    print(f"\n[INFO] Successfully encoded {len(known_encodings)} face(s).\n")
    if not FACE_RECOGNITION_AVAILABLE:
        print("[WARNING] face_recognition is not installed.")
        print("  -> Running fallback recognition mode (lower accuracy, suitable for testing).")
    return known_encodings, known_keys


# =============================================================================
#  TEXT-TO-SPEECH ENGINE – Speaks memory reminders aloud
# =============================================================================

class VoiceEngine:
    """
    Handles text-to-speech in a background thread so the UI stays responsive.
    Includes a cooldown mechanism to avoid repeating the same message.
    """

    def __init__(self, cooldown_seconds=VOICE_COOLDOWN):
        self.cooldown = cooldown_seconds
        self._last_spoken = {}  # person_key -> timestamp of last speech
        self._lock = threading.Lock()
        self._speaking = False

    def _speak_thread(self, text):
        """Runs TTS in a separate thread to avoid blocking the GUI."""
        if not TTS_AVAILABLE:
            self._speaking = False
            return

        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 140)   # Slower speech for elderly users
            engine.setProperty("volume", 1.0)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print(f"[TTS ERROR] {e}")
        finally:
            self._speaking = False

    def speak(self, person_key, text):
        """
        Speak a reminder for a person, respecting the cooldown timer.
        Won't repeat the same person's message within the cooldown period.
        """
        now = time.time()

        with self._lock:
            # Check cooldown
            last_time = self._last_spoken.get(person_key, 0)
            if now - last_time < self.cooldown:
                return  # Still in cooldown, skip

            # Don't overlap speech
            if self._speaking:
                return

            self._last_spoken[person_key] = now
            self._speaking = True

        # Launch speech in background thread
        thread = threading.Thread(target=self._speak_thread, args=(text,), daemon=True)
        thread.start()

    def force_speak(self, person_key, text):
        """Force speak regardless of cooldown (used by the Speak button)."""
        with self._lock:
            if self._speaking:
                return
            self._last_spoken[person_key] = time.time()
            self._speaking = True

        thread = threading.Thread(target=self._speak_thread, args=(text,), daemon=True)
        thread.start()


# =============================================================================
#  FACE PROCESSOR – Handles detection, encoding, and matching per frame
# =============================================================================

class FaceProcessor:
    """
    Processes webcam frames to detect and recognize faces.
    Uses frame skipping and scaling for performance optimization.
    """

    def __init__(self, known_encodings, known_keys, people_db,
                 tolerance=TOLERANCE, scale=SCALE_FACTOR):
        self.known_encodings = known_encodings
        self.known_keys = known_keys
        self.people_db = people_db
        self.tolerance = tolerance
        self.scale = scale
        self.fallback_similarity_threshold = 0.78

        # Cache the latest results so we can draw them every frame
        self.face_locations = []
        self.face_names = []
        self.face_keys = []
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if self.face_cascade.empty():
            print("[WARNING] Failed to load OpenCV Haar cascade for fallback detection.")

    def process_frame(self, frame):
        """
        Detect and recognize faces in a video frame.

        Steps:
          1. Resize the frame for faster processing
          2. Convert BGR → RGB (face_recognition uses RGB)
          3. Detect face locations
          4. Compute face encodings
          5. Compare each encoding against known faces
          6. Store results for rendering

        Returns:
            List of tuples: (person_key, person_name, face_location_in_original_scale)
        """
        results = []
        names = []
        keys = []

        if FACE_RECOGNITION_AVAILABLE:
            # Step 1 & 2: Resize and convert color space
            small_frame = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Step 3: Detect face locations (top, right, bottom, left)
            face_locations = face_recognition.face_locations(rgb_small, model="hog")

            # Step 4: Compute 128-d face encodings
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

            for encoding, location in zip(face_encodings, face_locations):
                name = "Unknown Person"
                person_key = None

                if len(self.known_encodings) > 0:
                    # Step 5: Compare against all known faces
                    distances = face_recognition.face_distance(self.known_encodings, encoding)
                    best_match_idx = np.argmin(distances)

                    if distances[best_match_idx] <= self.tolerance:
                        person_key = self.known_keys[best_match_idx]
                        person_info = self.people_db.get(person_key, {})
                        name = person_info.get("name", person_key)

                # Scale location back to original frame size
                inv_scale = 1.0 / self.scale
                top, right, bottom, left = [int(coord * inv_scale) for coord in location]

                results.append(
                    {
                        "person_key": person_key,
                        "name": name,
                        "location": (top, right, bottom, left),
                        "encoding": encoding,
                    }
                )
                names.append(name)
                keys.append(person_key)
        else:
            # Fallback mode: OpenCV cascade detection + lightweight matching.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(30, 30),
            )

            for (x, y, w, h) in faces:
                top, right, bottom, left = y, x + w, y + h, x
                name = "Unknown Person"
                person_key = None
                face_crop = frame[top:bottom, left:right]
                embedding = compute_fallback_embedding(face_crop)

                if embedding is not None and len(self.known_encodings) > 0:
                    similarities = [float(np.dot(embedding, known)) for known in self.known_encodings]
                    best_idx = int(np.argmax(similarities))
                    if similarities[best_idx] >= self.fallback_similarity_threshold:
                        person_key = self.known_keys[best_idx]
                        person_info = self.people_db.get(person_key, {})
                        name = person_info.get("name", person_key)

                results.append(
                    {
                        "person_key": person_key,
                        "name": name,
                        "location": (top, right, bottom, left),
                        "encoding": embedding,
                    }
                )
                names.append(name)
                keys.append(person_key)

        # Cache for drawing
        self.face_locations = [r["location"] for r in results]
        self.face_names = names
        self.face_keys = keys

        return results

    def draw_annotations(self, frame):
        """
        Draw bounding boxes and name labels on the frame.
        Green boxes for known faces, orange for unknown.
        """
        for location, name, key in zip(self.face_locations, self.face_names, self.face_keys):
            top, right, bottom, left = location

            # Choose color based on whether the person is known
            color = COLOR_KNOWN_BOX if key is not None else COLOR_UNKNOWN_BOX
            thickness = 2

            # Draw bounding box
            cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)

            # Draw name label background
            label_h = 30
            cv2.rectangle(frame, (left, bottom), (right, bottom + label_h), color, cv2.FILLED)

            # Draw name text
            cv2.putText(frame, name, (left + 6, bottom + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return frame


# =============================================================================
#  MAIN APPLICATION – Tkinter GUI with live webcam feed
# =============================================================================

class RemembrainApp:
    """
    Main application window combining:
      - Live webcam video feed with face annotations
      - Side panel showing recognized person's memory card
      - Voice reminder controls
    """

    def __init__(self):
        # ── Load data ───────────────────────────────────────────────────
        print("=" * 60)
        print("  REMEMBRAIN – AI Memory Assistant for Dementia Patients")
        print("=" * 60)
        print()

        self.people_db = load_people_database(DATA_FILE)
        self.known_encodings, self.known_keys = encode_known_faces(FACES_DIR, self.people_db)

        # ── Initialize components ───────────────────────────────────────
        self.face_processor = FaceProcessor(
            self.known_encodings, self.known_keys, self.people_db
        )
        self.voice_engine = VoiceEngine()
        self.current_place = DEFAULT_MEETING_PLACE
        self.last_seen_write_times = {}
        self.unknown_candidate = None

        # ── Frame counter for frame skipping ────────────────────────────
        self.frame_count = 0
        self.current_person_key = None  # Currently displayed person

        # ── Open webcam ─────────────────────────────────────────────────
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("[ERROR] Could not open webcam. Please check your camera.")
            sys.exit(1)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)

        print("[INFO] Webcam opened successfully.")
        print("[INFO] Starting Remembrain interface...\n")

        # ── Build the GUI ───────────────────────────────────────────────
        self._build_gui()

    def _build_gui(self):
        """Construct the tkinter window with video canvas and info panel."""
        self.root = tk.Tk()
        self.root.title(WINDOW_TITLE)
        self.root.configure(bg="#1a1a2e")
        self.root.resizable(False, False)

        # ── Custom fonts ────────────────────────────────────────────────
        self.font_title = tkfont.Font(family="Segoe UI", size=20, weight="bold")
        self.font_subtitle = tkfont.Font(family="Segoe UI", size=11)
        self.font_label = tkfont.Font(family="Segoe UI", size=10, weight="bold")
        self.font_value = tkfont.Font(family="Segoe UI", size=12)
        self.font_status = tkfont.Font(family="Segoe UI", size=9)
        self.font_button = tkfont.Font(family="Segoe UI", size=11, weight="bold")

        # ── Main container ──────────────────────────────────────────────
        main_frame = tk.Frame(self.root, bg="#1a1a2e")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        # ── Left: Video feed ────────────────────────────────────────────
        video_frame = tk.Frame(main_frame, bg="#0f0f23", highlightthickness=0)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(10, 5), pady=10)

        # Header for video
        video_header = tk.Frame(video_frame, bg="#16213e")
        video_header.pack(fill=tk.X)
        tk.Label(video_header, text="📷  Live Camera Feed", font=self.font_subtitle,
                 bg="#16213e", fg="#e94560", pady=8, padx=10).pack(anchor=tk.W)

        self.video_canvas = tk.Canvas(
            video_frame, width=VIDEO_WIDTH, height=VIDEO_HEIGHT,
            bg="#0f0f23", highlightthickness=2, highlightbackground="#e94560"
        )
        self.video_canvas.pack()

        # Status bar under video
        self.status_label = tk.Label(
            video_frame, text="● Scanning for faces...",
            font=self.font_status, bg="#0f0f23", fg="#53bf9d", pady=5
        )
        self.status_label.pack(fill=tk.X)

        # ── Right: Info panel ───────────────────────────────────────────
        self.panel_frame = tk.Frame(
            main_frame, bg="#16213e", width=PANEL_WIDTH,
            highlightthickness=2, highlightbackground="#e94560"
        )
        self.panel_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 10), pady=10)
        self.panel_frame.pack_propagate(False)

        # ── App title in panel ──────────────────────────────────────────
        title_frame = tk.Frame(self.panel_frame, bg="#e94560", pady=2)
        title_frame.pack(fill=tk.X)

        tk.Label(title_frame, text="🧠 REMEMBRAIN",
                 font=self.font_title, bg="#e94560", fg="white",
                 pady=12).pack()
        tk.Label(title_frame, text="AI Memory Assistant",
                 font=self.font_subtitle, bg="#e94560", fg="#ffd5de",
                 pady=(0)).pack()

        # Spacer
        tk.Frame(self.panel_frame, bg="#16213e", height=15).pack(fill=tk.X)

        # ── Person info card ────────────────────────────────────────────
        self.card_frame = tk.Frame(self.panel_frame, bg="#1a1a40", padx=20, pady=15,
                                   highlightthickness=1, highlightbackground="#533483")
        self.card_frame.pack(fill=tk.X, padx=15, pady=(5, 10))

        # Card header
        self.card_header = tk.Label(
            self.card_frame, text="Waiting for face...",
            font=tkfont.Font(family="Segoe UI", size=14, weight="bold"),
            bg="#1a1a40", fg="#53bf9d", wraplength=280, justify=tk.LEFT
        )
        self.card_header.pack(anchor=tk.W, pady=(0, 10))

        # Separator line
        tk.Frame(self.card_frame, bg="#533483", height=1).pack(fill=tk.X, pady=5)

        # Info fields
        self.info_labels = {}
        fields = [
            ("name", "👤  Name"),
            ("relationship", "💝  Relationship"),
            ("last_seen", "🕐  Last Seen"),
            ("notes", "📝  Notes"),
        ]

        for key, label_text in fields:
            field_frame = tk.Frame(self.card_frame, bg="#1a1a40")
            field_frame.pack(fill=tk.X, pady=4)

            tk.Label(field_frame, text=label_text, font=self.font_label,
                     bg="#1a1a40", fg="#8d8daa").pack(anchor=tk.W)

            value_label = tk.Label(
                field_frame, text="—", font=self.font_value,
                bg="#1a1a40", fg="white", wraplength=260, justify=tk.LEFT
            )
            value_label.pack(anchor=tk.W, padx=(15, 0))
            self.info_labels[key] = value_label

        # ── Spoken reminder text ────────────────────────────────────────
        tk.Frame(self.panel_frame, bg="#16213e", height=10).pack(fill=tk.X)

        self.reminder_frame = tk.Frame(self.panel_frame, bg="#0f3460", padx=15, pady=12)
        self.reminder_frame.pack(fill=tk.X, padx=15)

        tk.Label(self.reminder_frame, text="🔊 Voice Reminder:", font=self.font_label,
                 bg="#0f3460", fg="#53bf9d").pack(anchor=tk.W)

        self.reminder_text = tk.Label(
            self.reminder_frame, text="No reminder yet.",
            font=self.font_status, bg="#0f3460", fg="#b0b0d0",
            wraplength=280, justify=tk.LEFT
        )
        self.reminder_text.pack(anchor=tk.W, pady=(5, 0))

        # ── Speak button ────────────────────────────────────────────────
        tk.Frame(self.panel_frame, bg="#16213e", height=10).pack(fill=tk.X)

        self.speak_btn = tk.Button(
            self.panel_frame, text="🔊  Speak Reminder",
            font=self.font_button, bg="#e94560", fg="white",
            activebackground="#c73e54", activeforeground="white",
            relief=tk.FLAT, padx=20, pady=10, cursor="hand2",
            command=self._on_speak_click
        )
        self.speak_btn.pack(padx=15, fill=tk.X)

        tk.Frame(self.panel_frame, bg="#16213e", height=8).pack(fill=tk.X)

        self.save_unknown_btn = tk.Button(
            self.panel_frame,
            text="Save Unknown Face",
            font=self.font_status,
            bg="#2f8f6f",
            fg="white",
            activebackground="#276f57",
            activeforeground="white",
            relief=tk.FLAT,
            padx=10,
            pady=8,
            cursor="hand2",
            command=self._on_save_unknown_click,
        )
        self.save_unknown_btn.pack(padx=15, fill=tk.X)

        tk.Frame(self.panel_frame, bg="#16213e", height=6).pack(fill=tk.X)

        self.save_unknown_voice_btn = tk.Button(
            self.panel_frame,
            text="Save by Voice",
            font=self.font_status,
            bg="#1f6aa5",
            fg="white",
            activebackground="#195582",
            activeforeground="white",
            relief=tk.FLAT,
            padx=10,
            pady=8,
            cursor="hand2",
            command=self._on_save_unknown_voice_click,
        )
        self.save_unknown_voice_btn.pack(padx=15, fill=tk.X)

        tk.Frame(self.panel_frame, bg="#16213e", height=8).pack(fill=tk.X)

        self.place_btn = tk.Button(
            self.panel_frame, text="Update Current Place",
            font=self.font_status, bg="#533483", fg="white",
            activebackground="#4a2f77", activeforeground="white",
            relief=tk.FLAT, padx=10, pady=8, cursor="hand2",
            command=self._set_current_place
        )
        self.place_btn.pack(padx=15, fill=tk.X)

        self.place_label = tk.Label(
            self.panel_frame,
            text=f"Current Place: {self.current_place}",
            font=self.font_status,
            bg="#16213e",
            fg="#b0b0d0",
            pady=6,
        )
        self.place_label.pack(fill=tk.X)

        # ── Footer ─────────────────────────────────────────────────────
        footer = tk.Frame(self.panel_frame, bg="#16213e")
        footer.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        tk.Label(footer, text="Press Q or close window to exit",
                 font=self.font_status, bg="#16213e", fg="#555577").pack()

        # ── Keyboard binding ───────────────────────────────────────────
        self.root.bind("<q>", lambda e: self._quit())
        self.root.bind("<Q>", lambda e: self._quit())
        self.root.bind("<n>", lambda e: self._on_save_unknown_click())
        self.root.bind("<N>", lambda e: self._on_save_unknown_click())
        self.root.bind("<v>", lambda e: self._on_save_unknown_voice_click())
        self.root.bind("<V>", lambda e: self._on_save_unknown_voice_click())
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

    # ─────────────────────────────────────────────────────────────────────
    #  UI UPDATE METHODS
    # ─────────────────────────────────────────────────────────────────────

    def _update_card(self, person_key, person_info):
        """Update the side panel card with a recognized person's info."""
        last_seen_text = self._update_last_seen_record(person_key)
        person_info = self.people_db.get(person_key, person_info)

        self.card_header.config(text=f"✅ Person Recognized!", fg="#53bf9d")
        self.info_labels["name"].config(text=person_info.get("name", "Unknown"))
        self.info_labels["relationship"].config(text=person_info.get("relationship", "Unknown"))
        if not last_seen_text or last_seen_text == "Unknown":
            last_seen_text = "No earlier meeting recorded"
        self.info_labels["last_seen"].config(text=last_seen_text)
        self.info_labels["notes"].config(text=person_info.get("notes", "No notes available"))

        # Build reminder sentence
        name = person_info.get("name", "this person")
        relationship = person_info.get("relationship", "someone you know")
        last_seen = last_seen_text

        reminder = f"This is {name}, your {relationship}. You last met {last_seen.lower()}."
        self.reminder_text.config(text=f'"{reminder}"')

        # Store for the speak button
        self._current_reminder = reminder
        self.current_person_key = person_key

        # Auto-speak on first recognition
        self.voice_engine.speak(person_key, reminder)

    def _update_card_unknown(self):
        """Update the side panel for an unknown person."""
        self.card_header.config(text="❓ Unknown Person", fg="#e94560")
        self.info_labels["name"].config(text="Not recognized")
        self.info_labels["relationship"].config(text="—")
        self.info_labels["last_seen"].config(text="—")
        self.info_labels["notes"].config(text="This person is not in the memory database.")
        self.reminder_text.config(text="No reminder for unknown person.")
        self._current_reminder = None
        self.current_person_key = None

    def _clear_card(self):
        """Reset the card to its default waiting state."""
        self.card_header.config(text="Waiting for face...", fg="#8d8daa")
        for label in self.info_labels.values():
            label.config(text="—")
        self.reminder_text.config(text="No reminder yet.")
        self._current_reminder = None
        self.current_person_key = None

    def _on_speak_click(self):
        """Handle the Speak button click — replay the current reminder."""
        if self._current_reminder and self.current_person_key:
            self.voice_engine.force_speak(self.current_person_key, self._current_reminder)

    def _sanitize_person_key(self, name):
        """Convert display name into a safe JSON key."""
        key = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        return key or "person"

    def _capture_unknown_candidate(self, result, frame):
        """Store latest unknown face candidate for manual enrollment."""
        top, right, bottom, left = result["location"]
        h, w = frame.shape[:2]
        pad = 16
        top = max(0, top - pad)
        left = max(0, left - pad)
        bottom = min(h, bottom + pad)
        right = min(w, right + pad)

        face_crop = frame[top:bottom, left:right].copy()
        self.unknown_candidate = {
            "location": (top, right, bottom, left),
            "encoding": result.get("encoding"),
            "face_crop": face_crop,
        }

    def _validate_enrollment_ready(self):
        """Check prerequisites for saving a new person from unknown face."""
        if not self.unknown_candidate:
            self.status_label.config(
                text="● No unknown face available to save right now",
                fg="#f0a500",
            )
            return False

        return True

    def _listen_for_text(self, field_name, timeout=6, phrase_time_limit=4):
        """Capture one short spoken value (name/relationship) from microphone."""
        if not STT_AVAILABLE:
            return None

        recognizer = sr.Recognizer()

        try:
            with sr.Microphone() as source:
                self.status_label.config(
                    text=f"● Listening for {field_name}... speak now",
                    fg="#53bf9d",
                )
                self.root.update_idletasks()
                recognizer.adjust_for_ambient_noise(source, duration=0.6)
                audio = recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit,
                )

            text = recognizer.recognize_google(audio)
            return text.strip()
        except Exception as exc:
            self.status_label.config(
                text=f"● Voice capture failed: {exc}",
                fg="#e94560",
            )
            return None

    def _on_save_unknown_click(self):
        """Enroll currently visible unknown face into local database."""
        if not self._validate_enrollment_ready():
            return

        name = simpledialog.askstring(
            "Save New Person",
            "Enter person's name:",
            parent=self.root,
        )
        if not name:
            return

        relationship = simpledialog.askstring(
            "Save New Person",
            "Enter relationship (e.g., Friend, Cousin):",
            parent=self.root,
        )
        if not relationship:
            return

        notes = simpledialog.askstring(
            "Save New Person",
            "Optional notes:",
            parent=self.root,
        ) or ""

        self._enroll_unknown_person(name, relationship, notes)

    def _on_save_unknown_voice_click(self):
        """Enroll unknown face by speaking name/relationship via microphone."""
        if not self._validate_enrollment_ready():
            return

        if not STT_AVAILABLE:
            messagebox.showinfo(
                "Voice Enrollment",
                "speech_recognition is not installed. Use 'Save Unknown Face' typing flow.",
                parent=self.root,
            )
            return

        messagebox.showinfo(
            "Voice Enrollment",
            "You will speak two fields: Name and Relationship.\nClick OK, then speak clearly.",
            parent=self.root,
        )

        name = self._listen_for_text("name")
        if not name:
            messagebox.showwarning(
                "Voice Enrollment",
                "Could not capture name from microphone. Please try again.",
                parent=self.root,
            )
            return

        relationship = self._listen_for_text("relationship")
        if not relationship:
            messagebox.showwarning(
                "Voice Enrollment",
                "Could not capture relationship from microphone. Please try again.",
                parent=self.root,
            )
            return

        confirm = messagebox.askyesno(
            "Confirm Voice Input",
            f"Name: {name}\nRelationship: {relationship}\n\nSave this person?",
            parent=self.root,
        )
        if not confirm:
            return

        notes = simpledialog.askstring(
            "Save New Person",
            "Optional notes (type):",
            parent=self.root,
        ) or ""

        self._enroll_unknown_person(name, relationship, notes)

    def _enroll_unknown_person(self, name, relationship, notes):
        """Persist a new person entry using current unknown face candidate."""
        name = (name or "").strip()
        relationship = (relationship or "").strip()
        notes = (notes or "").strip()

        if not name or not relationship:
            return

        base_key = self._sanitize_person_key(name)
        person_key = base_key
        suffix = 2
        while person_key in self.people_db:
            person_key = f"{base_key}_{suffix}"
            suffix += 1

        os.makedirs(FACES_DIR, exist_ok=True)
        image_filename = f"{person_key}.jpg"
        image_path = os.path.join(FACES_DIR, image_filename)
        cv2.imwrite(image_path, self.unknown_candidate["face_crop"])

        encoding = self.unknown_candidate.get("encoding")
        if encoding is None:
            if FACE_RECOGNITION_AVAILABLE:
                rgb_crop = cv2.cvtColor(self.unknown_candidate["face_crop"], cv2.COLOR_BGR2RGB)
                computed = face_recognition.face_encodings(rgb_crop)
                if not computed:
                    self.status_label.config(
                        text="● Could not encode face. Try facing camera clearly and save again",
                        fg="#e94560",
                    )
                    return
                encoding = computed[0]
            else:
                encoding = compute_fallback_embedding(self.unknown_candidate["face_crop"])
                if encoding is None:
                    self.status_label.config(
                        text="● Could not create fallback face profile. Try again in better lighting",
                        fg="#e94560",
                    )
                    return

        timestamp_text = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.people_db[person_key] = {
            "name": name,
            "relationship": relationship,
            "last_seen_date": timestamp_text,
            "last_seen_place": self.current_place,
            "last_seen_with": OWNER_NAME,
            "last_seen": f"on {timestamp_text} at {self.current_place} with {OWNER_NAME}",
            "notes": notes,
            "image": f"faces/{image_filename}",
        }

        self.face_processor.known_encodings.append(encoding)
        self.face_processor.known_keys.append(person_key)
        self._save_people_database()
        self.unknown_candidate = None

        self.status_label.config(
            text=f"● Saved {name} successfully · now recognized automatically",
            fg="#53bf9d",
        )
        self._update_card(person_key, self.people_db[person_key])

    def _set_current_place(self):
        """Prompt user to set current meeting place used for memory updates."""
        place = simpledialog.askstring(
            "Update Place",
            "Enter current place (for last met updates):",
            initialvalue=self.current_place,
            parent=self.root,
        )
        if place:
            self.current_place = place.strip() or self.current_place
            self.place_label.config(text=f"Current Place: {self.current_place}")

    def _get_last_seen_text(self, person_info, use_previous=False):
        """Return readable last seen text from structured or legacy fields."""
        if use_previous:
            prev_date = person_info.get("previous_seen_date", "")
            prev_place = person_info.get("previous_seen_place", "")
            if prev_date and prev_place:
                return f"{prev_date} at {prev_place}"
            if prev_date:
                return prev_date
            if prev_place:
                return f"at {prev_place}"
            if person_info.get("previous_seen"):
                return person_info.get("previous_seen")

        date_text = person_info.get("last_seen_date", "")
        place_text = person_info.get("last_seen_place", "")

        if date_text and place_text:
            return f"{date_text} at {place_text}"
        if date_text:
            return date_text
        if place_text:
            return f"at {place_text}"
        return person_info.get("last_seen", "Unknown")

    def _save_people_database(self):
        """Persist people database updates to local JSON file."""
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(self.people_db, f, indent=4)

    def _update_last_seen_record(self, person_key):
        """Store current encounter while returning previous meeting text for display."""
        now = time.time()
        last_write = self.last_seen_write_times.get(person_key, 0)
        person = self.people_db.get(person_key)
        if not person:
            return "Unknown"

        if now - last_write < LAST_SEEN_UPDATE_COOLDOWN:
            previous_text = self._get_last_seen_text(person, use_previous=True)
            return previous_text if previous_text else "Unknown"

        previous_text = self._get_last_seen_text(person)

        # Shift current last-seen values into previous_* before writing this new encounter.
        old_date = person.get("last_seen_date", "")
        old_place = person.get("last_seen_place", "")
        old_with = person.get("last_seen_with", "")
        old_text = person.get("last_seen", "")
        if old_date or old_place or old_text:
            person["previous_seen_date"] = old_date
            person["previous_seen_place"] = old_place
            person["previous_seen_with"] = old_with
            person["previous_seen"] = old_text

        timestamp_text = datetime.now().strftime("%Y-%m-%d %H:%M")
        person["last_seen_date"] = timestamp_text
        person["last_seen_place"] = self.current_place
        person["last_seen_with"] = OWNER_NAME
        person["last_seen"] = f"on {timestamp_text} at {self.current_place} with {OWNER_NAME}"

        self.last_seen_write_times[person_key] = now
        self._save_people_database()
        return previous_text if previous_text else "Unknown"

    # ─────────────────────────────────────────────────────────────────────
    #  MAIN VIDEO LOOP
    # ─────────────────────────────────────────────────────────────────────

    def _video_loop(self):
        """
        Main loop: captures a webcam frame, optionally runs face recognition,
        draws annotations, and updates the tkinter canvas.

        Called repeatedly via root.after() for smooth, non-blocking updates.
        """
        ret, frame = self.cap.read()
        if not ret:
            self.status_label.config(text="● Camera error – no frame received", fg="#e94560")
            self.root.after(100, self._video_loop)
            return

        # Flip horizontally for a mirror-like experience
        frame = cv2.flip(frame, 1)

        # ── Run face recognition every FRAME_SKIP frames ────────────────
        self.frame_count += 1
        if self.frame_count % FRAME_SKIP == 0:
            results = self.face_processor.process_frame(frame)

            # Update the info panel based on results
            if len(results) > 0:
                # Show info for the first recognized (known) person, or first face
                primary = None
                for r in results:
                    if r["person_key"] is not None:  # Known person found
                        primary = r
                        break

                if primary:
                    person_key = primary["person_key"]
                    person_info = self.people_db.get(person_key, {})
                    self._update_card(person_key, person_info)
                    face_count = len(results)
                    known_count = sum(1 for r in results if r["person_key"] is not None)
                    self.status_label.config(
                        text=f"● {face_count} face(s) detected · {known_count} recognized",
                        fg="#53bf9d"
                    )
                else:
                    self._capture_unknown_candidate(results[0], frame)
                    self._update_card_unknown()
                    self.status_label.config(
                        text=f"● {len(results)} face(s) detected · None recognized · Press N or V to save",
                        fg="#f0a500"
                    )
            else:
                self.unknown_candidate = None
                self._clear_card()
                self.status_label.config(text="● Scanning for faces...", fg="#8d8daa")

        # ── Draw bounding boxes and labels on every frame ───────────────
        annotated = self.face_processor.draw_annotations(frame.copy())

        # ── Convert frame to tkinter-compatible image ───────────────────
        rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        # Resize to fit the canvas exactly
        display_frame = cv2.resize(rgb_frame, (VIDEO_WIDTH, VIDEO_HEIGHT))

        img = Image.fromarray(display_frame)
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.video_canvas.imgtk = imgtk  # Prevent garbage collection

        # ── Schedule the next frame ─────────────────────────────────────
        self.root.after(30, self._video_loop)  # ~33 FPS target

    # ─────────────────────────────────────────────────────────────────────
    #  START & STOP
    # ─────────────────────────────────────────────────────────────────────

    def run(self):
        """Start the application: begin video loop and enter tkinter mainloop."""
        self._current_reminder = None
        self._video_loop()
        self.root.mainloop()

    def _quit(self):
        """Clean up resources and close the application."""
        print("\n[INFO] Shutting down Remembrain...")
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()
        print("[INFO] Goodbye! 👋")


# =============================================================================
#  ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print()
    print("  Starting Remembrain...")
    print("  Make sure you have face images in the 'faces/' folder")
    print("  and person data in 'data/people.json'.")
    if not FACE_RECOGNITION_AVAILABLE:
        print("  face_recognition unavailable -> fallback mode enabled (test recognition).")
    if not TTS_AVAILABLE:
        print("  pyttsx3 unavailable -> voice reminders disabled.")
    print()

    app = RemembrainApp()
    app.run()
