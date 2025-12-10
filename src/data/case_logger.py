"""Utility helpers for recording historical adjustment cases."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from loguru import logger


@dataclass
class CaseRecordResult:
    """Metadata about a persisted historical case."""

    case_id: str
    source_id: Optional[str]
    path: Path


class CaseLogger:
    """Append-only logger for successful adjustment cases."""

    def __init__(self, history_path: Optional[Path] = None):
        self.history_path = Path(history_path or "data/historical_cases/cases.json")
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

    def _read_cases(self) -> List[Dict[str, Any]]:
        if not self.history_path.exists():
            return []
        try:
            with open(self.history_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, list):
                return data
            logger.warning("historical_cases JSON is not a list – reinitializing")
        except json.JSONDecodeError:
            logger.warning("historical_cases JSON parse failure – starting fresh")
        return []

    def _write_cases(self, cases: Sequence[Dict[str, Any]]) -> None:
        with open(self.history_path, "w", encoding="utf-8") as handle:
            json.dump(list(cases), handle, indent=2, ensure_ascii=False)

    def _collect_source_ids(self, cases: Sequence[Dict[str, Any]]) -> set[str]:
        ids: set[str] = set()
        for entry in cases:
            for key in ("source_id", "source_sample_id", "source_event_id"):
                if key in entry and entry[key] is not None:
                    ids.add(str(entry[key]))
        return ids

    def _next_case_id(self, cases: Sequence[Dict[str, Any]]) -> str:
        max_seq = 0
        current_year = datetime.now().year
        prefix = f"CASE-{current_year}-"
        for entry in cases:
            case_id = str(entry.get("case_id", ""))
            if case_id.startswith(prefix):
                suffix = case_id.split("-")[-1]
                if suffix.isdigit():
                    max_seq = max(max_seq, int(suffix))
        return f"{prefix}{max_seq + 1:03d}"

    def record_case(
        self,
        entry: Dict[str, Any],
        source_id: Optional[Any] = None,
        *,
        allow_duplicates: bool = False,
    ) -> Optional[CaseRecordResult]:
        """Persist a single case entry.

        Args:
            entry: Case payload. 'case_id' and 'date' will be filled if missing.
            source_id: Optional unique identifier used for deduplication.
            allow_duplicates: If True, skip deduplication guard.

        Returns:
            CaseRecordResult if saved, None if skipped due to duplication.
        """

        cases = self._read_cases()
        dedup_source = self._normalize_source_id(entry, source_id)

        if not allow_duplicates and dedup_source is not None:
            existing_ids = self._collect_source_ids(cases)
            if dedup_source in existing_ids:
                logger.info("Skipping duplicate historical case for source_id=%s", dedup_source)
                return None

        entry_to_save = dict(entry)
        if dedup_source is not None:
            entry_to_save.setdefault("source_id", dedup_source)

        if not str(entry_to_save.get("case_id", "")).strip():
            entry_to_save["case_id"] = self._next_case_id(cases)

        if not str(entry_to_save.get("date", "")).strip():
            entry_to_save["date"] = datetime.now().strftime("%Y-%m-%d")

        entry_to_save.setdefault("logged_at", datetime.now().isoformat())

        updated_cases = list(cases) + [entry_to_save]
        self._write_cases(updated_cases)

        logger.info("Added historical case %s", entry_to_save["case_id"])
        return CaseRecordResult(
            case_id=entry_to_save["case_id"],
            source_id=dedup_source,
            path=self.history_path,
        )

    @staticmethod
    def _normalize_source_id(entry: Dict[str, Any], source_id: Optional[Any]) -> Optional[str]:
        candidate = source_id
        if candidate is None:
            for key in ("source_sample_id", "source_event_id", "source_id"):
                if entry.get(key) is not None:
                    candidate = entry.get(key)
                    break
        if candidate is None:
            return None
        return str(candidate)