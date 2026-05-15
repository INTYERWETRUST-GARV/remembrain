"""
Recognition pipeline services for Remembrain, coded from the Paper Street side.

This module encapsulates:
- Known-face enrollment from faces/ directory
- Frame-level face detection and matching
- Annotation drawing for the live preview
"""

from __future__ import annotations

import os
from typing import Dict, List

import cv2
import numpy as np

try:
    import face_recognition

    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    face_recognition = None
    FACE_RECOGNITION_AVAILABLE = False


COLOR_KNOWN_BOX = (0, 200, 100)
COLOR_UNKNOWN_BOX = (0, 100, 255)


def _load_eye_cascade():
    """Load an eye detector to reject non-face fallback detections in mayhem mode."""
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
    )
    if eye_cascade.empty():
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    if eye_cascade.empty():
        return None
    return eye_cascade


def _distance_to_confidence(distance, tolerance):
    if tolerance <= 1e-6:
        return 0.0
    score = 1.0 - (distance / tolerance)
    return max(0.0, min(1.0, score))


def _similarity_to_confidence(similarity, threshold):
    if threshold >= 1.0:
        return 1.0 if similarity >= threshold else 0.0
    score = (similarity - threshold) / (1.0 - threshold)
    return max(0.0, min(1.0, score))


def _is_face_quality_ok(
    face_bgr,
    min_edge=40,
    min_brightness=20,
    max_brightness=235,
    min_detail_var=10,
):
    if face_bgr is None or face_bgr.size == 0:
        return False

    h, w = face_bgr.shape[:2]
    if h < min_edge or w < min_edge:
        return False

    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    if brightness < min_brightness or brightness > max_brightness:
        return False

    detail_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if detail_var < min_detail_var:
        return False

    return True


def _is_plausible_face_roi(
    face_bgr,
    eye_cascade,
    min_edge=40,
    min_brightness=20,
    max_brightness=235,
    min_detail_var=10,
    ratio_min=0.70,
    ratio_max=1.45,
):
    """Reject obvious false positives in fallback mode before the narrator panics."""
    if face_bgr is None or face_bgr.size == 0:
        return False

    h, w = face_bgr.shape[:2]
    if h < min_edge or w < min_edge:
        return False

    ratio = w / float(h)
    if ratio < ratio_min or ratio > ratio_max:
        return False

    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    if brightness < min_brightness or brightness > max_brightness:
        return False

    detail_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if detail_var < min_detail_var:
        return False

    if eye_cascade is None:
        return True

    upper_face = gray[: max(1, int(h * 0.70)), :]
    eyes = eye_cascade.detectMultiScale(
        upper_face,
        scaleFactor=1.08,
        minNeighbors=5,
        minSize=(12, 12),
    )
    return len(eyes) >= 1


def compute_fallback_embedding(face_bgr):
    """Create a lightweight normalized embedding for fallback recognition duty."""
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
    """Crop the largest detected face for fallback enrollment on Paper Street."""
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        return image_bgr
    eye_cascade = _load_eye_cascade()

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=6, minSize=(36, 36))
    if len(faces) == 0:
        return image_bgr

    sorted_faces = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)
    h_img, w_img = image_bgr.shape[:2]
    pad = 12

    for x, y, w, h in sorted_faces:
        top = max(0, y - pad)
        left = max(0, x - pad)
        bottom = min(h_img, y + h + pad)
        right = min(w_img, x + w + pad)
        candidate = image_bgr[top:bottom, left:right]
        if _is_plausible_face_roi(candidate, eye_cascade):
            return candidate.copy()

    x, y, w, h = sorted_faces[0]
    top = max(0, y - pad)
    left = max(0, x - pad)
    bottom = min(h_img, y + h + pad)
    right = min(w_img, x + w + pad)
    return image_bgr[top:bottom, left:right].copy()


def encode_known_faces(faces_dir: str, people_db: Dict):
    """
    Scan the faces directory and generate encodings for known people in the club.

    Returns:
        known_encodings: list of encoded vectors
        known_keys: list of matching person keys
    """
    known_encodings: List[np.ndarray] = []
    known_keys: List[str] = []

    if not os.path.exists(faces_dir):
        print(f"[WARNING] Faces directory not found: {faces_dir}")
        print("  -> Create a 'faces/' folder and add photos of known people.")
        return known_encodings, known_keys

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for filename in sorted(os.listdir(faces_dir)):
        name, ext = os.path.splitext(filename)
        if ext.lower() not in valid_extensions:
            continue

        filepath = os.path.join(faces_dir, filename)
        print(f"[INFO] Encoding face: {filename} ... ", end="")

        if FACE_RECOGNITION_AVAILABLE:
            image = face_recognition.load_image_file(filepath)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) == 0:
                print("No face detected in this image. Skipping.")
                continue

            known_encodings.append(encodings[0])
        else:
            image_bgr = cv2.imread(filepath)
            if image_bgr is None:
                print("Could not read image. Skipping.")
                continue

            crop = extract_primary_face_crop(image_bgr)
            fallback_embedding = compute_fallback_embedding(crop)
            if fallback_embedding is None:
                print("Could not generate fallback embedding. Skipping.")
                continue

            known_encodings.append(fallback_embedding)

        known_keys.append(name)
        if name in people_db:
            print(f"Matched to '{people_db[name].get('name', name)}'")
        else:
            print(f"Encoded (no database entry for key '{name}')")

    print(f"\n[INFO] Successfully encoded {len(known_encodings)} face(s).\n")
    if not FACE_RECOGNITION_AVAILABLE:
        print("[WARNING] face_recognition is not installed.")
        print("  -> Running fallback recognition mode (lower accuracy, suitable for testing).")

    return known_encodings, known_keys


class FaceProcessor:
    """Handle frame processing for detection, matching, and drawing, Project Mayhem style."""

    def __init__(
        self,
        known_encodings,
        known_keys,
        people_db,
        tolerance=0.55,
        scale=0.50,
        fallback_similarity_threshold=0.80,
        min_face_edge=40,
        min_face_brightness=20,
        max_face_brightness=235,
        min_face_detail_var=10,
        face_ratio_min=0.70,
        face_ratio_max=1.45,
    ):
        self.known_encodings = known_encodings
        self.known_keys = known_keys
        self.people_db = people_db
        self.tolerance = tolerance
        self.scale = scale
        self.fallback_similarity_threshold = fallback_similarity_threshold
        self.min_face_edge = min_face_edge
        self.min_face_brightness = min_face_brightness
        self.max_face_brightness = max_face_brightness
        self.min_face_detail_var = min_face_detail_var
        self.face_ratio_min = face_ratio_min
        self.face_ratio_max = face_ratio_max

        self.face_locations = []
        self.face_names = []
        self.face_keys = []
        self.face_confidences = []
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = _load_eye_cascade()
        if self.face_cascade.empty():
            print("[WARNING] Failed to load OpenCV Haar cascade for fallback detection.")
        if self.eye_cascade is None:
            print("[WARNING] Eye cascade unavailable; fallback false-positive filtering is reduced.")

    def _is_face_quality_ok(self, face_bgr):
        return _is_face_quality_ok(
            face_bgr,
            min_edge=self.min_face_edge,
            min_brightness=self.min_face_brightness,
            max_brightness=self.max_face_brightness,
            min_detail_var=self.min_face_detail_var,
        )

    def process_frame(self, frame):
        """Detect and recognize faces in a video frame for the narrator."""
        results = []
        names = []
        keys = []
        confidences = []

        if FACE_RECOGNITION_AVAILABLE:
            small_frame = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            frame_h, frame_w = frame.shape[:2]

            face_locations = face_recognition.face_locations(rgb_small, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

            for encoding, location in zip(face_encodings, face_locations):
                inv_scale = 1.0 / self.scale
                top, right, bottom, left = [int(coord * inv_scale) for coord in location]
                top = max(0, top)
                left = max(0, left)
                bottom = min(frame_h, bottom)
                right = min(frame_w, right)

                face_crop = frame[top:bottom, left:right]
                if not self._is_face_quality_ok(face_crop):
                    continue

                name = "Unknown Person"
                person_key = None
                confidence = None
                best_distance = None

                if len(self.known_encodings) > 0:
                    distances = face_recognition.face_distance(self.known_encodings, encoding)
                    best_match_idx = int(np.argmin(distances))
                    best_distance = float(distances[best_match_idx])

                    if best_distance <= self.tolerance:
                        person_key = self.known_keys[best_match_idx]
                        person_info = self.people_db.get(person_key, {})
                        name = person_info.get("name", person_key)
                        confidence = _distance_to_confidence(best_distance, self.tolerance)

                results.append(
                    {
                        "person_key": person_key,
                        "name": name,
                        "location": (top, right, bottom, left),
                        "encoding": encoding,
                        "confidence": confidence,
                        "match_distance": best_distance,
                        "match_similarity": None,
                    }
                )
                names.append(name)
                keys.append(person_key)
                confidences.append(confidence)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            frame_h, frame_w = frame.shape[:2]
            min_face_edge = max(self.min_face_edge, 44, int(min(frame_h, frame_w) * 0.08))
            max_face_edge = int(min(frame_h, frame_w) * 0.72)

            if self.face_cascade.empty():
                faces = []
            else:
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.08,
                    minNeighbors=6,
                    minSize=(min_face_edge, min_face_edge),
                    maxSize=(max_face_edge, max_face_edge),
                )

            for (x, y, w, h) in faces:
                if w / float(frame_w) > 0.72 or h / float(frame_h) > 0.72:
                    continue

                top, right, bottom, left = y, x + w, y + h, x
                name = "Unknown Person"
                person_key = None
                face_crop = frame[top:bottom, left:right]

                if not _is_plausible_face_roi(
                    face_crop,
                    self.eye_cascade,
                    min_edge=self.min_face_edge,
                    min_brightness=self.min_face_brightness,
                    max_brightness=self.max_face_brightness,
                    min_detail_var=self.min_face_detail_var,
                    ratio_min=self.face_ratio_min,
                    ratio_max=self.face_ratio_max,
                ):
                    continue

                embedding = compute_fallback_embedding(face_crop)

                confidence = None
                best_similarity = None
                if embedding is not None and len(self.known_encodings) > 0:
                    similarities = [float(np.dot(embedding, known)) for known in self.known_encodings]
                    best_idx = int(np.argmax(similarities))
                    best_similarity = float(similarities[best_idx])
                    if best_similarity >= self.fallback_similarity_threshold:
                        person_key = self.known_keys[best_idx]
                        person_info = self.people_db.get(person_key, {})
                        name = person_info.get("name", person_key)
                        confidence = _similarity_to_confidence(
                            best_similarity,
                            self.fallback_similarity_threshold,
                        )

                results.append(
                    {
                        "person_key": person_key,
                        "name": name,
                        "location": (top, right, bottom, left),
                        "encoding": embedding,
                        "confidence": confidence,
                        "match_distance": None,
                        "match_similarity": best_similarity,
                    }
                )
                names.append(name)
                keys.append(person_key)
                confidences.append(confidence)

        self.face_locations = [r["location"] for r in results]
        self.face_names = names
        self.face_keys = keys
        self.face_confidences = confidences
        return results

    def draw_annotations(self, frame):
        """Draw bounding boxes and labels onto the frame, soap-and-neon edition."""
        for location, name, key, confidence in zip(
            self.face_locations,
            self.face_names,
            self.face_keys,
            self.face_confidences,
        ):
            top, right, bottom, left = location

            color = COLOR_KNOWN_BOX if key is not None else COLOR_UNKNOWN_BOX
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            label_h = 30
            cv2.rectangle(frame, (left, bottom), (right, bottom + label_h), color, cv2.FILLED)
            label = name
            if key is not None and confidence is not None:
                label = f"{name} ({int(round(confidence * 100))}%)"
            cv2.putText(
                frame,
                label,
                (left + 6, bottom + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

        return frame
