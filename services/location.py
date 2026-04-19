

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request


class GoogleMapsLocationService:
    """Resolve current location to a precise address, no Paper Street guesswork."""

    GEOLOCATION_URL = "https://www.googleapis.com/geolocation/v1/geolocate"
    GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"

    def __init__(self, api_key: str, timeout_seconds: float = 8.0):
        self.api_key = (api_key or "").strip()
        self.timeout_seconds = max(2.0, float(timeout_seconds))
        self.retry_attempts = 2

    def is_configured(self):
        """Return True when a valid API key is present for this Paper Street run."""
        return bool(self.api_key)

    def get_precise_location(self):
        """Fetch current location details from Google APIs for the narrator timeline."""
        if not self.is_configured():
            return {
                "ok": False,
                "error": "Google Maps API key not configured.",
            }

        geo_result = self._run_with_retry(self._fetch_geolocation)
        if not geo_result.get("ok"):
            return geo_result

        latitude = geo_result["latitude"]
        longitude = geo_result["longitude"]
        accuracy_m = geo_result.get("accuracy_m")

        geocode_result = self._run_with_retry(lambda: self._reverse_geocode(latitude, longitude))
        if not geocode_result.get("ok"):
            return geocode_result

        return {
            "ok": True,
            "place_text": geocode_result["formatted_address"],
            "place_id": geocode_result.get("place_id"),
            "latitude": latitude,
            "longitude": longitude,
            "accuracy_m": accuracy_m,
            "map_url": f"https://maps.google.com/?q={latitude},{longitude}",
        }

    def _run_with_retry(self, operation):
        """Retry failed API work to smooth temporary Project Mayhem turbulence."""
        attempts = max(1, int(self.retry_attempts))
        last_result = None

        for attempt in range(attempts):
            result = operation()
            if result.get("ok"):
                return result
            last_result = result

            if attempt < attempts - 1:
                time.sleep(0.35 * (attempt + 1))

        return last_result or {
            "ok": False,
            "error": "Google Maps request failed.",
        }

    def _fetch_geolocation(self):
        """Call Google Geolocation API using network signals and IP hints, first rule."""
        url = f"{self.GEOLOCATION_URL}?key={urllib.parse.quote(self.api_key)}"
        payload = json.dumps({"considerIp": True}).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        response = self._call_json(request)
        if not response.get("ok"):
            return response

        body = response.get("body", {})
        location = body.get("location", {})
        latitude = location.get("lat")
        longitude = location.get("lng")
        accuracy_m = body.get("accuracy")

        if latitude is None or longitude is None:
            return {
                "ok": False,
                "error": "Google Geolocation API did not return coordinates.",
            }

        return {
            "ok": True,
            "latitude": float(latitude),
            "longitude": float(longitude),
            "accuracy_m": float(accuracy_m) if accuracy_m is not None else None,
        }

    def _reverse_geocode(self, latitude: float, longitude: float):
        """Resolve coordinates into a formatted address for the Paper Street logbook."""
        params = urllib.parse.urlencode(
            {
                "latlng": f"{latitude},{longitude}",
                "language": "en",
                "result_type": "street_address|premise|route|locality|administrative_area_level_2|administrative_area_level_1|country",
                "key": self.api_key,
            }
        )
        request = urllib.request.Request(f"{self.GEOCODE_URL}?{params}", method="GET")

        response = self._call_json(request)
        if not response.get("ok"):
            return response

        body = response.get("body", {})
        results = body.get("results", [])
        if not results:
            return {
                "ok": False,
                "error": "Google Geocoding API returned no address for current coordinates.",
            }

        top = results[0]
        formatted_address = top.get("formatted_address")
        if not formatted_address:
            return {
                "ok": False,
                "error": "Google Geocoding response missing formatted address.",
            }

        return {
            "ok": True,
            "formatted_address": formatted_address,
            "place_id": top.get("place_id"),
        }

    def _call_json(self, request):
        """Execute HTTP request and return parsed JSON with readable, no-secrets errors."""
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = ""

            return {
                "ok": False,
                "error": f"Google Maps API HTTP {exc.code}: {detail or exc.reason}",
            }
        except urllib.error.URLError as exc:
            return {
                "ok": False,
                "error": f"Network error while contacting Google Maps API: {exc.reason}",
            }
        except Exception as exc:
            return {
                "ok": False,
                "error": f"Unexpected Google Maps API error: {exc}",
            }

        try:
            body = json.loads(raw)
        except Exception as exc:
            return {
                "ok": False,
                "error": f"Invalid JSON from Google Maps API: {exc}",
            }

        status = body.get("status")
        if status and status not in {"OK", "ZERO_RESULTS"}:
            error_message = body.get("error_message") or body.get("error", {}).get("message")
            return {
                "ok": False,
                "error": f"Google Maps API status {status}: {error_message or 'No details'}",
            }

        if isinstance(body.get("error"), dict):
            message = body["error"].get("message", "Unknown Google Maps API error")
            return {
                "ok": False,
                "error": message,
            }

        return {
            "ok": True,
            "body": body,
        }
