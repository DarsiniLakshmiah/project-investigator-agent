"""Dependency health of a notebook environment (run right after %pip installs).

Checks, all deterministic:

1. ``pip check``: no installed package has unsatisfied requirements;
2. pins: every ``name==version`` line of the given requirement files is installed at
   exactly that version, and requirement files contain exact pins only;
3. shadowing: a notebook-scoped install that hides a Databricks runtime package with a
   different version is detected from the distributions on ``sys.path`` (the same
   package found twice). Shadowing a protected package (configs/environments/
   dependencies.yaml) is a failure; other shadowing is reported.

The report also records the resolved versions of the packages listed in the policy.
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

POLICY_FILE = Path("environments") / "dependencies.yaml"
_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9.+!_-]+)$")


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


@dataclass(frozen=True)
class Shadow:
    name: str
    active_version: str
    active_location: str
    hidden: tuple[tuple[str, str], ...]  # (version, location) of hidden copies


@dataclass
class HealthReport:
    pip_check_ok: bool
    pip_check_output: str
    pin_mismatches: list[str] = field(default_factory=list)
    unpinned_lines: list[str] = field(default_factory=list)
    shadowed: list[Shadow] = field(default_factory=list)
    protected: tuple[str, ...] = ()
    versions: dict[str, str | None] = field(default_factory=dict)

    @property
    def protected_shadowed(self) -> list[Shadow]:
        protected = {canonical(p) for p in self.protected}
        return [s for s in self.shadowed if canonical(s.name) in protected]

    @property
    def ok(self) -> bool:
        return (
            self.pip_check_ok
            and not self.pin_mismatches
            and not self.unpinned_lines
            and not self.protected_shadowed
        )

    def format(self) -> str:
        lines = [
            f"dependency health: {'OK' if self.ok else 'FAILED'}",
            f"  pip check: {'OK' if self.pip_check_ok else 'FAILED'}",
        ]
        if not self.pip_check_ok:
            lines += [f"    {line}" for line in self.pip_check_output.splitlines()]
        lines += [f"  pin mismatch: {m}" for m in self.pin_mismatches]
        lines += [f"  not an exact pin: {u}" for u in self.unpinned_lines]
        protected = {s.name for s in self.protected_shadowed}
        for s in self.shadowed:
            hidden = ", ".join(v for v, _ in s.hidden)
            tag = (
                "PROTECTED runtime package replaced" if s.name in protected else "overrides runtime"
            )
            lines.append(f"  {tag}: {s.name} {s.active_version} (runtime had {hidden})")
        lines.append("  resolved versions:")
        lines += [
            f"    {n}=={v}" if v else f"    {n}: not installed" for n, v in self.versions.items()
        ]
        return "\n".join(lines)


def pip_check(
    python: str = sys.executable, runner: Callable[..., Any] = subprocess.run
) -> tuple[bool, str]:
    result = runner([python, "-m", "pip", "check"], capture_output=True, text=True, timeout=300)
    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode == 0, output.strip()


def read_pins(paths: Iterable[Path]) -> tuple[dict[str, str], list[str]]:
    """Exact pins of requirement files, and every non-comment line that is not one."""
    pins: dict[str, str] = {}
    unpinned: list[str] = []
    for path in paths:
        for raw in Path(path).read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            match = _PIN.match(line)
            if match:
                pins[canonical(match.group(1))] = match.group(2)
            else:
                unpinned.append(f"{Path(path).name}: {line}")
    return pins, unpinned


def pin_mismatches(pins: dict[str, str], version_of: Callable[[str], str | None]) -> list[str]:
    out = []
    for name, wanted in sorted(pins.items()):
        installed = version_of(name)
        if installed is None:
            out.append(f"{name}=={wanted} required, not installed")
        elif _base(installed) != _base(wanted):
            out.append(f"{name}=={wanted} required, {installed} installed")
    return out


def _base(version: str) -> str:
    """Compare without a local build tag (torch 2.12.0+cpu satisfies ==2.12.0)."""
    return version.split("+", 1)[0]


def installed_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def shadowed_distributions(paths: Sequence[str] | None = None) -> list[Shadow]:
    """Packages present more than once on the path with different versions.

    The first copy in path order is the one Python imports (the notebook-scoped one).
    """
    seen: dict[str, list[tuple[str, str]]] = {}
    for dist in metadata.distributions(path=list(paths) if paths is not None else sys.path):
        name = dist.metadata["Name"]
        if not name:
            continue
        location = str(getattr(dist, "_path", "") or "")
        seen.setdefault(canonical(name), []).append((dist.version, location))
    out = []
    for name, copies in sorted(seen.items()):
        active_version, active_location = copies[0]
        hidden = tuple(c for c in copies[1:] if c[0] != active_version)
        if hidden:
            out.append(Shadow(name, active_version, active_location, hidden))
    return out


def load_policy(config_dir: Path) -> dict[str, list[str]]:
    data = yaml.safe_load((Path(config_dir) / POLICY_FILE).read_text(encoding="utf-8")) or {}
    return {"protected": list(data.get("protected", [])), "report": list(data.get("report", []))}


def check_environment(
    repo_root: Path,
    requirement_files: Sequence[str],
    config_dir: Path,
    runner: Callable[..., Any] = subprocess.run,
    paths: Sequence[str] | None = None,
) -> HealthReport:
    policy = load_policy(config_dir)
    ok, output = pip_check(runner=runner)
    pins, unpinned = read_pins(Path(repo_root) / f for f in requirement_files)
    return HealthReport(
        pip_check_ok=ok,
        pip_check_output=output,
        pin_mismatches=pin_mismatches(pins, installed_version),
        unpinned_lines=unpinned,
        shadowed=shadowed_distributions(paths),
        protected=tuple(policy["protected"]),
        versions={n: installed_version(n) for n in policy["report"]},
    )
