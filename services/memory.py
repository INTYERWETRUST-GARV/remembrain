"""Memory and persistence services for Remembrain, with narrator-level Paper Street recall."""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from typing import Optional


class MemoryService:
    """Own people.json persistence and last-seen updates for the Project Mayhem ledger."""

    def __init__(
        self,
        data_file: str,
        owner_name: str,
        default_place: str,
        last_seen_write_cooldown: int,
    ):
        self.data_file = data_file
        self.owner_name = owner_name
        self.current_place = default_place
        self.current_place_source = "manual"
        self.current_location_coords = None
        self.last_seen_write_cooldown = last_seen_write_cooldown
        self.last_seen_write_times = {}
        self.people_db = self.load_people_database()

    def load_people_database(self):
        """Load the people database from JSON storage, no first-rule surprises."""
        if not os.path.exists(self.data_file):
            print(f"[WARNING] Data file not found: {self.data_file}")
            print("  -> Create 'data/people.json' with known people's information.")
            return {}

        try:
            with open(self.data_file, "r", encoding="utf-8") as file_handle:
                people = json.load(file_handle)
        except Exception as exc:
            print(f"[WARNING] Failed to load people database: {exc}")
            return {}

        if not isinstance(people, dict):
            print("[WARNING] Invalid people.json format. Expected object at top level.")
            return {}

        print(f"[INFO] Loaded {len(people)} people from database.")
        for key, info in people.items():
            print(f"  - {info.get('name', key)} ({info.get('relationship', 'Unknown')})")

        return people

    def save_people_database(self):
        """Persist in-memory people data back to JSON for the narrator archive."""
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        with open(self.data_file, "w", encoding="utf-8") as file_handle:
            json.dump(self.people_db, file_handle, indent=4)

    def set_current_place(
        self,
        place: str,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        source: str = "manual",
    ):
        """Update current place context for upcoming memory writes on Paper Street."""
        if place and place.strip():
            self.current_place = place.strip()

        self.current_place_source = (source or "manual").strip().lower()

        if latitude is None or longitude is None:
            self.current_location_coords = None
        else:
            self.current_location_coords = (float(latitude), float(longitude))

    def _apply_current_place_fields(self, person_record):
        """Apply current place metadata to one person record in the club ledger."""
        person_record["last_seen_place"] = self.current_place
        person_record["last_seen_place_source"] = self.current_place_source

        if self.current_location_coords is None:
            person_record.pop("last_seen_latitude", None)
            person_record.pop("last_seen_longitude", None)
            return

        lat, lng = self.current_location_coords
        person_record["last_seen_latitude"] = round(float(lat), 6)
        person_record["last_seen_longitude"] = round(float(lng), 6)

    def sanitize_person_key(self, name: str):
        """Convert a display name into a JSON-safe key, Tyler-proofed."""
        key = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        return key or "person"

    def allocate_person_key(self, name: str):
        """Allocate a unique key for a newly enrolled person, no identity collisions."""
        base_key = self.sanitize_person_key(name)
        person_key = base_key
        suffix = 2
        while person_key in self.people_db:
            person_key = f"{base_key}_{suffix}"
            suffix += 1
        return person_key

    def add_person(self, person_key: str, name: str, relationship: str, notes: str, image_filename: str):
        """Create and persist a newly enrolled person record for the Paper Street file."""
        timestamp_text = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.people_db[person_key] = {
            "name": name,
            "relationship": relationship,
            "last_seen_date": timestamp_text,
            "last_seen_with": self.owner_name,
            "notes": notes,
            "image": f"faces/{image_filename}",
        }
        self._apply_current_place_fields(self.people_db[person_key])
        seen_place = self.people_db[person_key].get("last_seen_place", self.current_place)
        self.people_db[person_key]["last_seen"] = (
            f"on {timestamp_text} at {seen_place} with {self.owner_name}"
        )

        self.save_people_database()
        return self.people_db[person_key]

    def _get_last_seen_text(self, person_info, use_previous=False):
        """Return readable last-seen text from structured or legacy narrator memory fields."""
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

    def update_last_seen_record(self, person_key: str):
        """Store the current encounter while returning previous meeting text for recall."""
        now = time.time()
        last_write = self.last_seen_write_times.get(person_key, 0)
        person = self.people_db.get(person_key)
        if not person:
            return "Unknown"

        if now - last_write < self.last_seen_write_cooldown:
            previous_text = self._get_last_seen_text(person, use_previous=True)
            return previous_text if previous_text else "Unknown"

        previous_text = self._get_last_seen_text(person)

        old_date = person.get("last_seen_date", "")
        old_place = person.get("last_seen_place", "")
        old_place_source = person.get("last_seen_place_source", "")
        old_latitude = person.get("last_seen_latitude")
        old_longitude = person.get("last_seen_longitude")
        old_with = person.get("last_seen_with", "")
        old_text = person.get("last_seen", "")
        if old_date or old_place or old_text or old_latitude is not None or old_longitude is not None:
            person["previous_seen_date"] = old_date
            person["previous_seen_place"] = old_place
            person["previous_seen_place_source"] = old_place_source
            person["previous_seen_with"] = old_with
            person["previous_seen"] = old_text
            person["previous_seen_latitude"] = old_latitude
            person["previous_seen_longitude"] = old_longitude

        timestamp_text = datetime.now().strftime("%Y-%m-%d %H:%M")
        person["last_seen_date"] = timestamp_text
        self._apply_current_place_fields(person)
        person["last_seen_with"] = self.owner_name
        seen_place = person.get("last_seen_place", self.current_place)
        person["last_seen"] = f"on {timestamp_text} at {seen_place} with {self.owner_name}"

        self.last_seen_write_times[person_key] = now
        self.save_people_database()
        return previous_text if previous_text else "Unknown"
