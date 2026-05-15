"""
    REMEMBRAIN - Paper Street Memory Assistant
    Local-first desktop companion that helps people reconnect with familiar faces.
    This file plays the narrator role: UI orchestration stays here while core
    logic lives in the service layer:
        - services/recognition.py
        - services/memory.py
        - services/voice.py
    Run:
        python main.py
"""

import os
import sys
import threading
import time
import webbrowser

import cv2
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import font as tkfont
from tkinter import messagebox, simpledialog

from services.location import GoogleMapsLocationService
from services.memory import MemoryService
from services.recognition import (
    FACE_RECOGNITION_AVAILABLE,
    FaceProcessor,
    compute_fallback_embedding,
    encode_known_faces,
)
from services.voice import TTS_AVAILABLE, VoiceEngine

try:
    import face_recognition
except ImportError:
    face_recognition = None

try:
    import speech_recognition as sr

    STT_AVAILABLE = True
except ImportError:
    sr = None
    STT_AVAILABLE = False




BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACES_DIR = os.path.join(BASE_DIR, "faces")
DATA_FILE = os.path.join(BASE_DIR, "data", "people.json")

FRAME_SKIP = 2
SCALE_FACTOR = 0.50
TOLERANCE = 0.55
VOICE_COOLDOWN = 30
LAST_SEEN_UPDATE_COOLDOWN = 30
FALLBACK_SIMILARITY_THRESHOLD = 0.80
QUALITY_MIN_FACE_EDGE = 40
QUALITY_MIN_BRIGHTNESS = 20
QUALITY_MAX_BRIGHTNESS = 235
QUALITY_MIN_DETAIL_VAR = 10
QUALITY_RATIO_MIN = 0.70
QUALITY_RATIO_MAX = 1.45
UNKNOWN_CANDIDATE_TIMEOUT_SECONDS = 5.0

OWNER_NAME = os.getenv("REMEMBRAIN_OWNER", "software owner")
DEFAULT_MEETING_PLACE = os.getenv("REMEMBRAIN_PLACE", "Home")
GOOGLE_MAPS_API_KEY = os.getenv("REMEMBRAIN_GOOGLE_MAPS_API_KEY", "").strip()
GOOGLE_MAPS_TIMEOUT_SECONDS = 8.0
GOOGLE_MAPS_AUTO_REFRESH_SECONDS = max(
    60,
    int(os.getenv("REMEMBRAIN_MAPS_AUTO_REFRESH_SECONDS", "300")),
)
GOOGLE_MAPS_STALE_SECONDS = max(
    90,
    int(os.getenv("REMEMBRAIN_MAPS_STALE_SECONDS", "420")),
)

WINDOW_TITLE = "Remembrain - AI Memory Assistant"
VIDEO_WIDTH = 720
VIDEO_HEIGHT = 540
PANEL_WIDTH = 340





class RemembrainApp:
    """Main desktop ringmaster for this small, kind version of Project Mayhem."""

    def __init__(self):
        print("=" * 60)
        print("  REMEMBRAIN - AI Memory Assistant for Dementia Patients")
        print("=" * 60)
        print()

        self.memory_service = MemoryService(
            data_file=DATA_FILE,
            owner_name=OWNER_NAME,
            default_place=DEFAULT_MEETING_PLACE,
            last_seen_write_cooldown=LAST_SEEN_UPDATE_COOLDOWN,
        )
        self.people_db = self.memory_service.people_db

        self.known_encodings, self.known_keys = encode_known_faces(FACES_DIR, self.people_db)

        self.face_processor = FaceProcessor(
            self.known_encodings,
            self.known_keys,
            self.people_db,
            tolerance=TOLERANCE,
            scale=SCALE_FACTOR,
            fallback_similarity_threshold=FALLBACK_SIMILARITY_THRESHOLD,
            min_face_edge=QUALITY_MIN_FACE_EDGE,
            min_face_brightness=QUALITY_MIN_BRIGHTNESS,
            max_face_brightness=QUALITY_MAX_BRIGHTNESS,
            min_face_detail_var=QUALITY_MIN_DETAIL_VAR,
            face_ratio_min=QUALITY_RATIO_MIN,
            face_ratio_max=QUALITY_RATIO_MAX,
        )
        self.voice_engine = VoiceEngine(cooldown_seconds=VOICE_COOLDOWN)
        self.location_service = GoogleMapsLocationService(
            api_key=GOOGLE_MAPS_API_KEY,
            timeout_seconds=GOOGLE_MAPS_TIMEOUT_SECONDS,
        )

        self.frame_count = 0
        self.current_person_key = None
        self.unknown_candidate = None
        self.unknown_candidate_seen_epoch = 0.0
        self._maps_lookup_running = False
        self._maps_after_id = None
        self._last_maps_update_epoch = 0.0
        self._last_maps_error = None
        self._current_maps_accuracy_m = None
        self._current_map_url = None

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("[ERROR] Could not open webcam. Please check your camera.")
            sys.exit(1)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)

        print("[INFO] Webcam opened successfully.")
        print("[INFO] Starting Remembrain interface...\n")

        self._build_gui()

    def _build_gui(self):
        """Build the Paper Street control room: camera canvas plus memory panel."""
        self.root = tk.Tk()
        self.root.title(WINDOW_TITLE)
        self.root.configure(bg="#1a1a2e")
        self.root.resizable(False, False)

        self.font_title = tkfont.Font(family="Segoe UI", size=20, weight="bold")
        self.font_subtitle = tkfont.Font(family="Segoe UI", size=11)
        self.font_label = tkfont.Font(family="Segoe UI", size=10, weight="bold")
        self.font_value = tkfont.Font(family="Segoe UI", size=12)
        self.font_status = tkfont.Font(family="Segoe UI", size=9)
        self.font_button = tkfont.Font(family="Segoe UI", size=11, weight="bold")

        main_frame = tk.Frame(self.root, bg="#1a1a2e")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        video_frame = tk.Frame(main_frame, bg="#0f0f23", highlightthickness=0)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(10, 5), pady=10)

        video_header = tk.Frame(video_frame, bg="#16213e")
        video_header.pack(fill=tk.X)
        tk.Label(
            video_header,
            text="Live Camera Feed",
            font=self.font_subtitle,
            bg="#16213e",
            fg="#e94560",
            pady=8,
            padx=10,
        ).pack(anchor=tk.W)

        self.video_canvas = tk.Canvas(
            video_frame,
            width=VIDEO_WIDTH,
            height=VIDEO_HEIGHT,
            bg="#0f0f23",
            highlightthickness=2,
            highlightbackground="#e94560",
        )
        self.video_canvas.pack()

        self.status_label = tk.Label(
            video_frame,
            text="Scanning for faces...",
            font=self.font_status,
            bg="#0f0f23",
            fg="#53bf9d",
            pady=5,
        )
        self.status_label.pack(fill=tk.X)

        self.panel_frame = tk.Frame(
            main_frame,
            bg="#16213e",
            width=PANEL_WIDTH,
            highlightthickness=2,
            highlightbackground="#e94560",
        )
        self.panel_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 10), pady=10)
        self.panel_frame.pack_propagate(False)

        title_frame = tk.Frame(self.panel_frame, bg="#e94560", pady=2)
        title_frame.pack(fill=tk.X)

        tk.Label(
            title_frame,
            text="REMEMBRAIN",
            font=self.font_title,
            bg="#e94560",
            fg="white",
            pady=12,
        ).pack()
        tk.Label(
            title_frame,
            text="AI Memory Assistant",
            font=self.font_subtitle,
            bg="#e94560",
            fg="#ffd5de",
            pady=0,
        ).pack()

        tk.Frame(self.panel_frame, bg="#16213e", height=15).pack(fill=tk.X)

        self.card_frame = tk.Frame(
            self.panel_frame,
            bg="#1a1a40",
            padx=20,
            pady=15,
            highlightthickness=1,
            highlightbackground="#533483",
        )
        self.card_frame.pack(fill=tk.X, padx=15, pady=(5, 10))

        self.card_header = tk.Label(
            self.card_frame,
            text="Waiting for face...",
            font=tkfont.Font(family="Segoe UI", size=14, weight="bold"),
            bg="#1a1a40",
            fg="#53bf9d",
            wraplength=280,
            justify=tk.LEFT,
        )
        self.card_header.pack(anchor=tk.W, pady=(0, 10))

        tk.Frame(self.card_frame, bg="#533483", height=1).pack(fill=tk.X, pady=5)

        self.info_labels = {}
        fields = [
            ("name", "Name"),
            ("relationship", "Relationship"),
            ("last_seen", "Last Seen"),
            ("notes", "Notes"),
        ]

        for key, label_text in fields:
            field_frame = tk.Frame(self.card_frame, bg="#1a1a40")
            field_frame.pack(fill=tk.X, pady=4)

            tk.Label(
                field_frame,
                text=label_text,
                font=self.font_label,
                bg="#1a1a40",
                fg="#8d8daa",
            ).pack(anchor=tk.W)

            value_label = tk.Label(
                field_frame,
                text="-",
                font=self.font_value,
                bg="#1a1a40",
                fg="white",
                wraplength=260,
                justify=tk.LEFT,
            )
            value_label.pack(anchor=tk.W, padx=(15, 0))
            self.info_labels[key] = value_label

        tk.Frame(self.panel_frame, bg="#16213e", height=10).pack(fill=tk.X)

        self.reminder_frame = tk.Frame(self.panel_frame, bg="#0f3460", padx=15, pady=12)
        self.reminder_frame.pack(fill=tk.X, padx=15)

        tk.Label(
            self.reminder_frame,
            text="Voice Reminder:",
            font=self.font_label,
            bg="#0f3460",
            fg="#53bf9d",
        ).pack(anchor=tk.W)

        self.reminder_text = tk.Label(
            self.reminder_frame,
            text="No reminder yet.",
            font=self.font_status,
            bg="#0f3460",
            fg="#b0b0d0",
            wraplength=280,
            justify=tk.LEFT,
        )
        self.reminder_text.pack(anchor=tk.W, pady=(5, 0))

        tk.Frame(self.panel_frame, bg="#16213e", height=10).pack(fill=tk.X)

        self.speak_btn = tk.Button(
            self.panel_frame,
            text="Speak Reminder",
            font=self.font_button,
            bg="#e94560",
            fg="white",
            activebackground="#c73e54",
            activeforeground="white",
            relief=tk.FLAT,
            padx=20,
            pady=10,
            cursor="hand2",
            command=self._on_speak_click,
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
            self.panel_frame,
            text="Update Current Place",
            font=self.font_status,
            bg="#533483",
            fg="white",
            activebackground="#4a2f77",
            activeforeground="white",
            relief=tk.FLAT,
            padx=10,
            pady=8,
            cursor="hand2",
            command=self._set_current_place,
        )
        self.place_btn.pack(padx=15, fill=tk.X)

        tk.Frame(self.panel_frame, bg="#16213e", height=6).pack(fill=tk.X)

        self.maps_place_btn = tk.Button(
            self.panel_frame,
            text="Use Google Maps Location",
            font=self.font_status,
            bg="#2d6a8f",
            fg="white",
            activebackground="#245776",
            activeforeground="white",
            relief=tk.FLAT,
            padx=10,
            pady=8,
            cursor="hand2",
            command=self._set_current_place_from_maps,
        )
        self.maps_place_btn.pack(padx=15, fill=tk.X)

        tk.Frame(self.panel_frame, bg="#16213e", height=6).pack(fill=tk.X)

        self.open_map_btn = tk.Button(
            self.panel_frame,
            text="Open Current Place in Maps",
            font=self.font_status,
            bg="#24546f",
            fg="white",
            activebackground="#1e465c",
            activeforeground="white",
            relief=tk.FLAT,
            padx=10,
            pady=8,
            cursor="hand2",
            command=self._open_current_place_map,
            state=tk.DISABLED,
        )
        self.open_map_btn.pack(padx=15, fill=tk.X)

        self.place_label = tk.Label(
            self.panel_frame,
            text=f"Current Place: {self.memory_service.current_place}",
            font=self.font_status,
            bg="#16213e",
            fg="#b0b0d0",
            pady=6,
        )
        self.place_label.pack(fill=tk.X)

        self.place_meta_label = tk.Label(
            self.panel_frame,
            text="",
            font=self.font_status,
            bg="#16213e",
            fg="#8d8daa",
            pady=0,
        )
        self.place_meta_label.pack(fill=tk.X)
        self._refresh_place_labels()

        footer = tk.Frame(self.panel_frame, bg="#16213e")
        footer.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        tk.Label(
            footer,
            text="Press Q or close window to exit",
            font=self.font_status,
            bg="#16213e",
            fg="#555577",
        ).pack()

        self.root.bind("<q>", lambda _event: self._quit())
        self.root.bind("<Q>", lambda _event: self._quit())
        self.root.bind("<n>", lambda _event: self._on_save_unknown_click())
        self.root.bind("<N>", lambda _event: self._on_save_unknown_click())
        self.root.bind("<v>", lambda _event: self._on_save_unknown_voice_click())
        self.root.bind("<V>", lambda _event: self._on_save_unknown_voice_click())
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

        if self.location_service.is_configured():
            self._schedule_next_maps_sync(delay_ms=1200)

    def _update_card(self, person_key, person_info, confidence=None):
        """Update the memory card when a familiar face steps into the Paper Street frame."""
        last_seen_text = self.memory_service.update_last_seen_record(person_key)
        person_info = self.people_db.get(person_key, person_info)

        if confidence is None:
            header_text = "Person Recognized"
        else:
            header_text = f"Person Recognized ({int(round(confidence * 100))}%)"
        self.card_header.config(text=header_text, fg="#53bf9d")
        self.info_labels["name"].config(text=person_info.get("name", "Unknown"))
        self.info_labels["relationship"].config(text=person_info.get("relationship", "Unknown"))
        if not last_seen_text or last_seen_text == "Unknown":
            last_seen_text = "No earlier meeting recorded"
        self.info_labels["last_seen"].config(text=last_seen_text)
        self.info_labels["notes"].config(text=person_info.get("notes", "No notes available"))

        name = person_info.get("name", "this person")
        relationship = person_info.get("relationship", "someone you know")
        reminder = f"This is {name}, your {relationship}. You last met {last_seen_text.lower()}."

        self.reminder_text.config(text=f'"{reminder}"')
        self._current_reminder = reminder
        self.current_person_key = person_key
        self.voice_engine.speak(person_key, reminder)

    def _update_card_unknown(self):
        """Switch card state for an unknown face, narrator-style uncertainty included."""
        self.card_header.config(text="Unknown Person", fg="#e94560")
        self.info_labels["name"].config(text="Not recognized")
        self.info_labels["relationship"].config(text="-")
        self.info_labels["last_seen"].config(text="-")
        self.info_labels["notes"].config(text="This person is not in the memory database.")
        self.reminder_text.config(text="No reminder for unknown person.")
        self._current_reminder = None
        self.current_person_key = None

    def _clear_card(self):
        """Reset the card to a quiet waiting state between Paper Street encounters."""
        self.card_header.config(text="Waiting for face...", fg="#8d8daa")
        for label in self.info_labels.values():
            label.config(text="-")
        self.reminder_text.config(text="No reminder yet.")
        self._current_reminder = None
        self.current_person_key = None

    def _on_speak_click(self):
        """Replay the active reminder; first rule is repeat what matters."""
        if self._current_reminder and self.current_person_key:
            self.voice_engine.force_speak(self.current_person_key, self._current_reminder)

    def _capture_unknown_candidate(self, result, frame):
        """Capture the latest unknown candidate so Paper Street enrollment can happen on demand."""
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
        self.unknown_candidate_seen_epoch = time.time()

    def _validate_enrollment_ready(self):
        """Confirm we have a candidate before kicking off Project Enrollment."""
        if not self.unknown_candidate:
            self.status_label.config(
                text="No unknown face available to save right now",
                fg="#f0a500",
            )
            return False
        if self.unknown_candidate_seen_epoch:
            elapsed = time.time() - self.unknown_candidate_seen_epoch
            if elapsed > UNKNOWN_CANDIDATE_TIMEOUT_SECONDS:
                self.unknown_candidate = None
                self.unknown_candidate_seen_epoch = 0.0
                self.status_label.config(
                    text="Unknown face timed out; look at the camera again",
                    fg="#f0a500",
                )
                return False
        return True

    def _listen_for_text(self, field_name, timeout=6, phrase_time_limit=4):
        """Capture one short spoken field from the mic, narrator to narrator."""
        if not STT_AVAILABLE:
            return None

        recognizer = sr.Recognizer()

        try:
            with sr.Microphone() as source:
                self.status_label.config(
                    text=f"Listening for {field_name}... speak now",
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
                text=f"Voice capture failed: {exc}",
                fg="#e94560",
            )
            return None

    def _on_save_unknown_click(self):
        """Enroll the currently visible unknown face through typed input, Project Mayhem style."""
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

        notes = (
            simpledialog.askstring(
                "Save New Person",
                "Optional notes:",
                parent=self.root,
            )
            or ""
        )

        self._enroll_unknown_person(name, relationship, notes)

    def _on_save_unknown_voice_click(self):
        """Enroll an unknown face through voice capture, one field at a time on Paper Street."""
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

        notes = (
            simpledialog.askstring(
                "Save New Person",
                "Optional notes (type):",
                parent=self.root,
            )
            or ""
        )

        self._enroll_unknown_person(name, relationship, notes)

    def _enroll_unknown_person(self, name, relationship, notes):
        """Persist a new person record from the current unknown face candidate in the club ledger."""
        name = (name or "").strip()
        relationship = (relationship or "").strip()
        notes = (notes or "").strip()

        if not name or not relationship:
            return

        person_key = self.memory_service.allocate_person_key(name)

        os.makedirs(FACES_DIR, exist_ok=True)
        image_filename = f"{person_key}.jpg"
        image_path = os.path.join(FACES_DIR, image_filename)
        cv2.imwrite(image_path, self.unknown_candidate["face_crop"])

        encoding = self.unknown_candidate.get("encoding")
        if encoding is None:
            if FACE_RECOGNITION_AVAILABLE and face_recognition is not None:
                rgb_crop = cv2.cvtColor(self.unknown_candidate["face_crop"], cv2.COLOR_BGR2RGB)
                computed = face_recognition.face_encodings(rgb_crop)
                if not computed:
                    self.status_label.config(
                        text="Could not encode face. Try facing camera clearly and save again",
                        fg="#e94560",
                    )
                    return
                encoding = computed[0]
            else:
                encoding = compute_fallback_embedding(self.unknown_candidate["face_crop"])
                if encoding is None:
                    self.status_label.config(
                        text="Could not create fallback face profile. Try again in better lighting",
                        fg="#e94560",
                    )
                    return

        person_info = self.memory_service.add_person(
            person_key=person_key,
            name=name,
            relationship=relationship,
            notes=notes,
            image_filename=image_filename,
        )

        self.face_processor.known_encodings.append(encoding)
        self.face_processor.known_keys.append(person_key)
        self.unknown_candidate = None

        self.status_label.config(
            text=f"Saved {name} successfully; now recognized automatically",
            fg="#53bf9d",
        )
        self._update_card(person_key, person_info)

    def _set_current_place(self):
        """Set current place context used when writing last-seen memory for the narrator."""
        place = simpledialog.askstring(
            "Update Place",
            "Enter current place (for last met updates):",
            initialvalue=self.memory_service.current_place,
            parent=self.root,
        )
        if place:
            self.memory_service.set_current_place(place, source="manual")
            self._current_map_url = None
            self._current_maps_accuracy_m = None
            self.open_map_btn.config(state=tk.DISABLED)
            self._refresh_place_labels()

    def _refresh_place_labels(self):
        """Refresh place labels with source, coordinates, and timing for Paper Street context."""
        self.place_label.config(text=f"Current Place: {self.memory_service.current_place}")

        source = (self.memory_service.current_place_source or "manual").replace("_", " ").title()
        coords = self.memory_service.current_location_coords
        if coords is None:
            self.place_meta_label.config(text=f"Source: {source}")
            return

        latitude, longitude = coords
        source_meta = f"Source: {source} | {latitude:.5f}, {longitude:.5f}"

        if source == "Google Maps" and self._current_maps_accuracy_m is not None:
            source_meta += f" | ~{self._current_maps_accuracy_m:.0f}m"

        if source == "Google Maps" and self._last_maps_update_epoch > 0:
            updated_at = time.strftime("%H:%M", time.localtime(self._last_maps_update_epoch))
            source_meta += f" | updated {updated_at}"

        self.place_meta_label.config(
            text=source_meta
        )

    def _schedule_next_maps_sync(self, delay_ms=None):
        """Schedule the next automatic maps sync, like clockwork on Paper Street."""
        if not self.location_service.is_configured():
            return

        if self._maps_after_id is not None:
            try:
                self.root.after_cancel(self._maps_after_id)
            except Exception:
                pass

        refresh_ms = int(GOOGLE_MAPS_AUTO_REFRESH_SECONDS * 1000)
        next_delay_ms = refresh_ms if delay_ms is None else max(0, int(delay_ms))
        self._maps_after_id = self.root.after(next_delay_ms, self._run_scheduled_maps_sync)

    def _run_scheduled_maps_sync(self):
        """Timer callback that runs the background maps refresh cycle for Project Mayhem."""
        self._maps_after_id = None
        self._set_current_place_from_maps(user_initiated=False, force=False)

    def _is_maps_location_stale(self):
        """Return True when stored maps context is stale enough for a Paper Street refresh."""
        if self._last_maps_update_epoch <= 0:
            return True
        return (time.time() - self._last_maps_update_epoch) >= GOOGLE_MAPS_STALE_SECONDS

    def _set_current_place_from_maps(self, user_initiated=True, force=True):
        """Resolve and apply current place via Google Maps when the narrator needs context."""
        if self._maps_lookup_running:
            if not user_initiated:
                self._schedule_next_maps_sync()
            return

        if not self.location_service.is_configured():
            if user_initiated:
                messagebox.showinfo(
                    "Google Maps Not Configured",
                    "Set REMEMBRAIN_GOOGLE_MAPS_API_KEY to enable precise location updates.",
                    parent=self.root,
                )
            return

        if not force and not self._is_maps_location_stale():
            self._schedule_next_maps_sync()
            return

        self._maps_lookup_running = True
        self.maps_place_btn.config(state=tk.DISABLED)
        if user_initiated:
            self.status_label.config(text="Resolving location via Google Maps...", fg="#53bf9d")

        worker = threading.Thread(
            target=self._set_current_place_from_maps_worker,
            args=(user_initiated,),
            daemon=True,
        )
        worker.start()

    def _set_current_place_from_maps_worker(self, user_initiated):
        """Background worker that performs network location lookup behind the Paper Street curtain."""
        result = self.location_service.get_precise_location()
        self.root.after(0, lambda: self._apply_maps_place_result(result, user_initiated))

    def _apply_maps_place_result(self, result, user_initiated):
        """Apply async location results to UI state and memory context for the club."""
        self._maps_lookup_running = False
        self.maps_place_btn.config(state=tk.NORMAL)

        if not result.get("ok"):
            error = result.get("error", "Unknown Google Maps error")
            self._last_maps_error = error
            if user_initiated:
                self.status_label.config(text=f"Location update failed: {error}", fg="#e94560")
            else:
                self.status_label.config(
                    text="Using last known location; Google Maps retry is scheduled",
                    fg="#f0a500",
                )

            if user_initiated:
                messagebox.showwarning(
                    "Google Maps Location",
                    f"Could not fetch location.\n\n{error}",
                    parent=self.root,
                )

            self._schedule_next_maps_sync()
            return

        place_text = result.get("place_text")
        latitude = result.get("latitude")
        longitude = result.get("longitude")
        self._last_maps_error = None
        self._last_maps_update_epoch = time.time()
        self._current_maps_accuracy_m = result.get("accuracy_m")
        self._current_map_url = result.get("map_url")

        if self._current_map_url:
            self.open_map_btn.config(state=tk.NORMAL)
        else:
            self.open_map_btn.config(state=tk.DISABLED)

        self.memory_service.set_current_place(
            place_text,
            latitude=latitude,
            longitude=longitude,
            source="google_maps",
        )
        self._refresh_place_labels()

        if self._current_maps_accuracy_m is None:
            self.status_label.config(text="Location synced via Google Maps", fg="#53bf9d")
        else:
            self.status_label.config(
                text=f"Location synced via Google Maps (~{self._current_maps_accuracy_m:.0f}m accuracy)",
                fg="#53bf9d",
            )

        self._schedule_next_maps_sync()

    def _open_current_place_map(self):
        """Open the currently resolved map location in the default browser, first rule compliant."""
        if not self._current_map_url:
            messagebox.showinfo(
                "Google Maps",
                "No Google Maps location is available yet. Sync location first.",
                parent=self.root,
            )
            return

        try:
            webbrowser.open(self._current_map_url, new=2)
        except Exception as exc:
            messagebox.showwarning(
                "Google Maps",
                f"Could not open browser.\n\n{exc}",
                parent=self.root,
            )

    def _video_loop(self):
        """Capture frames, run recognition passes, and repaint the tkinter canvas on Paper Street."""
        ret, frame = self.cap.read()
        if not ret:
            self.status_label.config(text="Camera error - no frame received", fg="#e94560")
            self.root.after(100, self._video_loop)
            return

        frame = cv2.flip(frame, 1)

        self.frame_count += 1
        if self.frame_count % FRAME_SKIP == 0:
            results = self.face_processor.process_frame(frame)

            if len(results) > 0:
                primary = None
                known_results = [result for result in results if result["person_key"] is not None]
                if known_results:
                    primary = max(
                        known_results,
                        key=lambda item: item.get("confidence") or 0.0,
                    )

                if primary:
                    person_key = primary["person_key"]
                    person_info = self.people_db.get(person_key, {})
                    confidence = primary.get("confidence")
                    self._update_card(person_key, person_info, confidence=confidence)
                    self.unknown_candidate = None
                    self.unknown_candidate_seen_epoch = 0.0
                    face_count = len(results)
                    known_count = sum(1 for result in results if result["person_key"] is not None)
                    confidence_note = ""
                    if confidence is not None:
                        confidence_note = f"; {int(round(confidence * 100))}% match"
                    self.status_label.config(
                        text=f"{face_count} face(s) detected; {known_count} recognized{confidence_note}",
                        fg="#53bf9d",
                    )
                else:
                    self._capture_unknown_candidate(results[0], frame)
                    self._update_card_unknown()
                    self.status_label.config(
                        text=f"{len(results)} face(s) detected; none recognized; press N or V to save",
                        fg="#f0a500",
                    )
            else:
                self.unknown_candidate = None
                self.unknown_candidate_seen_epoch = 0.0
                self._clear_card()
                self.status_label.config(text="Scanning for faces...", fg="#8d8daa")

        annotated = self.face_processor.draw_annotations(frame.copy())

        rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        display_frame = cv2.resize(rgb_frame, (VIDEO_WIDTH, VIDEO_HEIGHT))

        img = Image.fromarray(display_frame)
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.video_canvas.imgtk = imgtk

        self.root.after(30, self._video_loop)

    def run(self):
        """Start the app event loop and continuous video processing for this narrator timeline."""
        self._current_reminder = None
        self._video_loop()
        self.root.mainloop()

    def _quit(self):
        """Release resources and close the app cleanly, no basement chaos."""
        print("\n[INFO] Shutting down Remembrain...")
        if self._maps_after_id is not None:
            try:
                self.root.after_cancel(self._maps_after_id)
            except Exception:
                pass

        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()
        print("[INFO] Goodbye!")


if __name__ == "__main__":
    print()
    print("  Starting Remembrain...")
    print("  Make sure you have face images in the 'faces/' folder")
    print("  and person data in 'data/people.json'.")
    if not FACE_RECOGNITION_AVAILABLE:
        print("  face_recognition unavailable -> fallback mode enabled (test recognition).")
    if not TTS_AVAILABLE:
        print("  pyttsx3 unavailable -> voice reminders disabled.")
    if GOOGLE_MAPS_API_KEY:
        print(
            "  Google Maps location integration enabled "
            f"(auto refresh every {GOOGLE_MAPS_AUTO_REFRESH_SECONDS}s)."
        )
    else:
        print("  Google Maps API key not set -> manual place updates only.")
    print()

    app = RemembrainApp()
    app.run()