"""Small read-only attempt reader; no rerun, reservation, model or endpoint setup."""

import re
from pathlib import Path

from worldbank_copilot.validation import phase10d_models as h


def read_attempt(settings, run_id="10d3"):
    if not re.fullmatch(r"10d[1-9][0-9]*", run_id):
        raise ValueError("invalid run ID")
    path = Path(settings.artifact_volume_path) / "phase10d_models" / f"{run_id}.json"
    files = []
    for candidate in sorted(path.parent.glob(run_id + ".*")):
        row = {"file": candidate.name, "bytes": candidate.stat().st_size}
        if candidate.name.endswith(".complete.json"):
            files.append(row)
            continue
        try:
            value = h.read_completed(candidate, require_final=candidate == path)
            row.update(
                summary=value["summary"],
                preflight=value["preflight"],
                revision=value["revision"],
                persistence=value["persistence"],
                schedule_present="schedule" in value,
                content_identity_present="content_identity" in value,
                failure=value.get("failure"),
            )
        except Exception as exc:
            row["read_error_class"] = type(exc).__name__
        files.append(row)
    return {"read_only": True, "run_id": run_id, "artifact": str(path), "files": files}
