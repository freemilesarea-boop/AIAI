"""Analysis cache, keyed so that stale results cannot survive.

A library of forty thousand files takes hours to analyse and will be
analysed many times as thresholds are tuned. Almost none of that work
needs redoing: the audio has not changed, and neither has most of the
configuration.

The key is what makes this safe rather than merely fast:

    sha256 + stage algorithm version + configuration digest

The content hash means a renamed or moved file keeps its cached
analysis, and a file whose bytes changed cannot reuse one. The stage
version and configuration digest are *per stage*, so changing a quality
threshold does not invalidate decode results, and changing the decoder
does not throw away every fingerprint. A single global key would make
every tuning pass a full re-analysis, and people respond to that by not
tuning.

Writes are atomic. An interrupted run — and long runs get interrupted —
must leave a cache that is readable, not one truncated mid-record.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CACHE_FORMAT_VERSION = 1


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    invalidated: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "invalidated": self.invalidated,
            "hit_rate": round(self.hit_rate, 4),
        }


class AnalysisCache:
    """Stage results keyed by content and configuration.

    Held in memory during a run and flushed once at the end. The
    alternative — writing on every entry — turns an analysis run into a
    write-amplified crawl, and the entries are cheap enough to hold.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._entries: dict[str, dict[str, Any]] = {}
        self.stats = CacheStats()
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A corrupt cache is a performance problem, never a
            # correctness one: discard it and recompute.
            self._entries = {}
            return
        if not isinstance(payload, dict) or payload.get("format") != CACHE_FORMAT_VERSION:
            self._entries = {}
            return
        entries = payload.get("entries")
        self._entries = entries if isinstance(entries, dict) else {}

    @staticmethod
    def key(sha256: str, stage: str, stage_version: str, configuration_key: str) -> str:
        return f"{sha256}|{stage}|{stage_version}|{configuration_key}"

    def get(self, key: str) -> dict[str, Any] | None:
        entry = self._entries.get(key)
        if entry is None:
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        return entry

    def put(self, key: str, value: dict[str, Any]) -> None:
        self._entries[key] = value

    def prune(self, live_hashes: set[str]) -> int:
        """Drop entries for files no longer present.

        A cache that only ever grows becomes the largest artifact of a
        dataset build. Keyed by hash, so this survives renames — a moved
        file is still live.
        """
        stale = [key for key in self._entries if key.split("|", 1)[0] not in live_hashes]
        for key in stale:
            del self._entries[key]
        self.stats.invalidated += len(stale)
        return len(stale)

    def flush(self) -> None:
        """Write atomically: rename is the only step that must not tear."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format": CACHE_FORMAT_VERSION, "entries": self._entries}
        handle = tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(self.path.parent),
            prefix=".cache-",
            suffix=".tmp",
            delete=False,
        )
        try:
            with handle:
                json.dump(payload, handle, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(handle.name, self.path)
        except BaseException:
            Path(handle.name).unlink(missing_ok=True)
            raise

    def __len__(self) -> int:
        return len(self._entries)
