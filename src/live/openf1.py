"""Optional OpenF1 HTTP adapter.

This module uses the standard library so the live core remains usable without
an extra client dependency. Network access is opt-in at the call site.
"""
from __future__ import annotations

import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class OpenF1Client:
    def __init__(self, base_url: str = "https://api.openf1.org/v1",
                 timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def fetch(self, resource: str, **params):
        query = urlencode({k: v for k, v in params.items() if v is not None})
        url = f"{self.base_url}/{resource.lstrip('/')}"
        if query:
            url += f"?{query}"
        request = Request(url, headers={"Accept": "application/json"})
        with urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def fetch_laps(self, session_key: int | str, driver_number: int | str):
        return self.fetch("laps", session_key=session_key,
                          driver_number=driver_number)

    def iter_laps(self, session_key: int | str, driver_number: int | str):
        """Yield normalized lap records ordered by lap number."""
        rows = self.fetch_laps(session_key, driver_number)
        for row in sorted(rows, key=lambda item: item.get("lap_number", 0)):
            if row.get("lap_duration") is None:
                continue
            yield {"lap": row["lap_number"], "laptime": row["lap_duration"],
                   "compound": row.get("compound") or "MEDIUM"}
