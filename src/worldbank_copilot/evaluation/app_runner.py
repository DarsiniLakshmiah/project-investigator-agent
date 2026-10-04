"""Sequential application-evaluation runner with append-only persistence and resume.

Every case goes through the canonical ``copilot.investigate(query=..., project_id=...)``.
A pass-through recorder wraps the Copilot's injected model adapters only to keep a copy of
what each model received or returned (the Synthesizer's evidence view, the Investigator's
proposed actions and call summaries); requests, replies and failures are unchanged, and
nothing is recorded into traces. One failing case is recorded and the run continues.

Persistence: ``<root>/<run_id>/results.jsonl`` (one line per attempt, appended, never
rewritten), ``run.json`` (run identity), ``golden.json`` (resolved golden facts with
record identity), and derived ``scored.jsonl`` / ``summary.json`` / ``report.md``.
Resume skips cases whose latest attempt is stored; ``rerun=True`` appends a new attempt
(earlier lines are kept; the latest attempt is scored).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from worldbank_copilot.evaluation.app_cases import Case
from worldbank_copilot.evaluation.app_metrics import aggregate, render_report, score_case


class RecordingAdapter:
    """Delegates to the real adapter; keeps the last request/reply text for evaluation."""

    def __init__(self, inner, sink: list):
        self._inner, self._sink = inner, sink

    def invoke(self, request):
        entry = {"context_json": request.context_json, "reply": None}
        self._sink.append(entry)
        reply = self._inner.invoke(request)
        entry["reply"] = reply.text
        return reply

    def __getattr__(self, name):  # last_reason / last_usage diagnostics stay visible
        return getattr(self._inner, name)


class Capture:
    """Per-case record of model inputs/outputs, reset before every case."""

    ROLES = ("investigator", "synthesizer", "critic")

    def __init__(self, copilot):
        self.calls: dict[str, list] = {r: [] for r in self.ROLES}
        for role in self.ROLES:
            inner = getattr(copilot, role, None)
            if inner is not None and not isinstance(inner, RecordingAdapter):
                setattr(copilot, role, RecordingAdapter(inner, self.calls[role]))
            elif isinstance(inner, RecordingAdapter):
                inner._sink = self.calls[role]

    def reset(self) -> None:
        for sink in self.calls.values():
            sink.clear()

    def snapshot(self) -> dict:
        decisions, calls_made = [], []
        for entry in self.calls["investigator"]:
            request = _json(entry["context_json"]) or {}
            calls_made = request.get("calls_made") or calls_made  # round 2 sees round-1 calls
            reply = _json(entry["reply"])
            if reply:
                decisions.append(
                    {k: reply.get(k) for k in ("disposition", "actions", "temporal_anchor")}
                )
        synthesis = [_json(e["context_json"]) or {} for e in self.calls["synthesizer"]]
        return {
            "decisions": decisions,
            "calls_made": [
                {k: c.get(k) for k in ("tool", "status", "evidence")} for c in calls_made
            ],
            "synthesis_evidence": synthesis[-1].get("evidence") if synthesis else None,
            "synthesis_anchor": synthesis[-1].get("anchor") if synthesis else None,
        }


def _json(text):
    try:
        return json.loads(text) if text else None
    except ValueError:
        return None


class RunStore:
    """Attempt-preserving JSONL store for one evaluation run.

    Unity Catalog Volumes (FUSE) support whole-file sequential writes but not append or
    seek: ``open("a")`` seeks to the end of the file and fails (Errno 29, Illegal seek). A
    record is therefore added by reading the existing lines and rewriting the whole file
    with the new line at the end (fine for a sequential, 50-case run). Every earlier
    attempt is rewritten unchanged; the result is read back and verified.
    """

    def __init__(self, root: Path, run_id: str):
        self.dir = Path(root) / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.results = self.dir / "results.jsonl"

    def _lines(self) -> tuple[list[str], list[dict], bool]:
        """Valid lines and records; True if a torn trailing line (interrupted write) exists.

        Only the LAST line may be malformed; a malformed line anywhere else is corruption
        and raises, so no stored attempt is ever silently ignored.
        """
        if not self.results.exists():
            return [], [], False
        lines = [line for line in self.results.read_text("utf-8").splitlines() if line.strip()]
        records = []
        for number, line in enumerate(lines, 1):
            try:
                records.append(json.loads(line))
            except ValueError:
                if number == len(lines):
                    return lines[:-1], records, True
                raise ValueError(f"{self.results}: malformed record on line {number}") from None
        return lines, records, False

    def records(self) -> list[dict]:
        return self._lines()[1]

    def latest(self) -> dict[str, dict]:
        latest: dict[str, dict] = {}
        for record in self.records():
            latest[record["case_id"]] = record
        return latest

    def append(self, record: dict, log=print) -> None:
        """Add one record; earlier attempts are kept (whole-file rewrite, no append/seek)."""
        lines, _, torn = self._lines()
        if torn:  # an interrupted write left a partial last line: it was never a record
            log(f"{self.results.name}: dropped one incomplete trailing line before writing")
        content = "".join(f"{line}\n" for line in lines)
        content += json.dumps(record, ensure_ascii=False, default=str) + "\n"
        self.results.write_text(content, encoding="utf-8")
        if self.results.read_text("utf-8") != content:
            raise OSError(f"{self.results}: read-back differs from the written records")

    def write_once(self, name: str, payload: Any) -> None:
        path = self.dir / name
        if not path.exists():
            path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def write(self, name: str, text: str) -> None:  # derived artefacts only
        (self.dir / name).write_text(text, encoding="utf-8")

    def read(self, name: str):
        path = self.dir / name
        return json.loads(path.read_text("utf-8")) if path.exists() else None


def select_cases(
    cases: list[Case], case_ids: Iterable[str] | None = None, categories=None
) -> list[Case]:
    ids, cats = set(case_ids or ()), set(categories or ())
    return [c for c in cases if (not ids or c.case_id in ids) and (not cats or c.category in cats)]


def run_cases(
    copilot,
    cases: list[Case],
    store: RunStore,
    *,
    rerun: bool = False,
    capture: Capture | None = None,
    clock=time.perf_counter,
    log=print,
) -> list[dict]:
    """Run cases sequentially; skip stored ones unless ``rerun``; never stop on one failure."""
    done = store.latest()
    capture = capture if capture is not None else Capture(copilot)
    written = []
    for case in cases:
        if case.case_id in done and not rerun:
            log(f"{case.case_id}: stored, skipped")
            continue
        capture.reset()
        started = clock()
        record = {
            "run_id": store.dir.name,
            "case_id": case.case_id,
            "category": case.category,
            "project_id": case.project_id,
            "question": case.question,
            "attempt": sum(1 for r in store.records() if r["case_id"] == case.case_id) + 1,
            "started_at": datetime.now(UTC).isoformat(),
            "result": None,
            "error": None,
        }
        try:
            result = copilot.investigate(query=case.question, project_id=case.project_id)
            record["result"] = result.model_dump(mode="json")
        except Exception as exc:  # recorded; the run continues
            record["error"] = {"type": type(exc).__name__, "message": str(exc)[:500]}
        record["captured"] = capture.snapshot()
        record["elapsed_ms"] = (clock() - started) * 1000
        store.append(record, log=log)
        written.append(record)
        status = record["result"]["status"] if record["result"] else "ERROR"
        log(f"{case.case_id} [{case.category}] {status} {record['elapsed_ms']:.0f} ms")
    return written


def score_run(cases, store: RunStore, golden: dict, questions: dict, reviews: dict | None = None):
    """Score the latest attempt of every stored case; write derived artefacts."""
    by_id = {c.case_id: c for c in cases}
    latest = store.latest()
    scored = [
        score_case(
            by_id[cid], latest[cid], golden.get(cid, {}), questions, (reviews or {}).get(cid)
        )
        for cid in by_id
        if cid in latest
    ]
    summary = aggregate(scored)
    run = store.read("run.json") or {"run_id": store.dir.name}
    store.write("scored.jsonl", "".join(json.dumps(s, default=str) + "\n" for s in scored))
    store.write("summary.json", json.dumps(summary, indent=2, default=str))
    store.write("report.md", render_report(summary, scored, run))
    return scored, summary


def run_identity(run_id: str, repo_root: Path, cases_text: str, copilot=None) -> dict:
    """Run metadata: revision (if resolvable), case-set hash, model endpoints."""
    try:
        revision = (
            subprocess.run(
                ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
            or "UNAVAILABLE"
        )
    except Exception:
        revision = "UNAVAILABLE"
    models = getattr(getattr(copilot, "config", None), "models", None)
    return {
        "run_id": run_id,
        "started_at": datetime.now(UTC).isoformat(),
        "revision": revision,
        "cases_sha256": hashlib.sha256(cases_text.encode()).hexdigest(),
        "models": models.model_dump() if models is not None else None,
    }


def review_template(cases: list[Case], store: RunStore) -> list[dict]:
    """Rows for human review of semantic dimensions (0/1/2; blank = not reviewed)."""
    latest = store.latest()
    rows = []
    for case in cases:
        result = (latest.get(case.case_id) or {}).get("result") or {}
        rows.append(
            {
                "case_id": case.case_id,
                "question": case.question,
                "status": result.get("status"),
                "expected_behavior": " | ".join(case.expected.required_behavior),
                "answer": " | ".join(c["text"] for c in result.get("claims", ()))
                or result.get("message"),
                "relevance": None,
                "completeness": None,
                "conciseness": None,
                "citation_entailment": None,
            }
        )
    return rows


def load_reviews(rows: Iterable[dict]) -> dict[str, dict]:
    keys = ("relevance", "completeness", "conciseness", "citation_entailment")
    out = {}
    for row in rows:
        scores = {k: int(row[k]) for k in keys if row.get(k) not in (None, "")}
        if any(v not in (0, 1, 2) for v in scores.values()):
            raise ValueError(f"{row.get('case_id')}: review scores must be 0, 1 or 2")
        if scores:
            out[row["case_id"]] = scores
    return out
