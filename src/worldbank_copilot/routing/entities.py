"""Stages 1-2: input validation and project / entity resolution (Phase 9B).

Entities that identify a project deterministically:

* project ids ``P\\d{6}`` (registry; a well-formed id outside it is UNSUPPORTED);
* corpus document ids ``P\\d{6}-<12 hex>`` (owner = the project prefix, by construction);
* report numbers (``RES00355``, ``PAD5226``, ...) from the curated document manifest;
  an unknown report number of the same shape is UNSUPPORTED (owner unknowable);
* loan numbers (``IBRD86010``, ``IBRD-8601-0``) from the registry's expected loans;
  an unknown IBRD/IDA loan is UNSUPPORTED (it may belong to any project);
* reviewed name aliases (configs/routing/project_aliases.yaml).

Overlapping matches keep the longest (a document id contains a project id).

Resolution never switches scope: a mention of another approved project, by any entity
kind, is CONFLICT (one foreign project) or MULTI (two or more projects), never a new
scope. Without an active scope, exactly one mentioned project becomes the scope.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from worldbank_copilot.common.project_registry import ProjectRegistry
from worldbank_copilot.ingestion.documents import DocumentManifest
from worldbank_copilot.routing.config import RoutingConfig, normalize_phrase, validate_aliases
from worldbank_copilot.routing.models import (
    AccessContext,
    InputStatus,
    InputValidation,
    Mention,
    MentionKind,
    ProjectResolution,
    ProjectStatus,
)

_PROJECT = re.compile(r"\bP\d{6}\b", re.IGNORECASE)
_DOCUMENT = re.compile(r"\bP\d{6}-[0-9a-f]{12}\b", re.IGNORECASE)
_LOAN = re.compile(r"\b(IBRD|IDA)[\s-]?(\d{4})[\s-]?(\d)\b", re.IGNORECASE)
_REPORT = re.compile(r"\b(RES|PADHP|PAD)\s?-?(\d{3,6})\b", re.IGNORECASE)


def validate_input(question: str, config: RoutingConfig) -> InputValidation:
    text = unicodedata.normalize("NFKC", question or "")
    text = "".join(ch if ch.isprintable() else " " for ch in text)
    text = " ".join(text.split())
    if not text:
        return InputValidation(status=InputStatus.EMPTY, normalized="", length=0)
    if len(text) > config.settings.input.max_chars:
        return InputValidation(status=InputStatus.TOO_LONG, normalized="", length=len(text))
    if not re.search(r"[A-Za-z]", text):
        return InputValidation(status=InputStatus.NO_TEXT, normalized=text, length=len(text))
    flags = tuple(
        "INJECTION_PATTERN"
        for p in config.settings.input.injection_patterns
        if re.search(p, text, re.IGNORECASE)
    )[:1]
    return InputValidation(status=InputStatus.VALID, normalized=text, length=len(text), flags=flags)


def normalize_loan(raw: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", raw.upper())


@dataclass(frozen=True)
class EntityIndex:
    """Deterministic entity -> project lookups built from governed configuration."""

    registry: ProjectRegistry
    loans: dict[str, str]  # normalised loan number -> project
    reports: dict[str, str]  # report number -> project
    aliases: dict[str, str]  # normalised alias -> project
    relative: tuple[re.Pattern[str], ...] = ()  # "the other program" (never a project)

    @classmethod
    def build(
        cls, registry: ProjectRegistry, manifest: DocumentManifest, config: RoutingConfig
    ) -> EntityIndex:
        loans = {
            normalize_loan(n): p.project_id
            for p in registry.projects
            for n in p.expected_loan_numbers or ()
        }
        reports: dict[str, str] = {}
        for project_id, entries in manifest.documents.items():
            for entry in entries:
                if entry.report_number:
                    reports[entry.report_number.upper()] = project_id
        relative = tuple(
            re.compile(p, re.IGNORECASE) for p in config.settings.relative_project_references
        )
        return cls(registry, loans, reports, validate_aliases(config.aliases, registry), relative)

    def mentions(self, text: str) -> list[Mention]:
        found: list[Mention] = []
        for m in _DOCUMENT.finditer(text):
            pid = m.group(0)[:7].upper()
            found.append(
                self._mention(
                    MentionKind.DOCUMENT_ID,
                    m,
                    m.group(0).lower(),
                    pid if pid in self.registry else None,
                    "document-id prefix",
                )
            )
        for m in _PROJECT.finditer(text):
            pid = m.group(0).upper()
            found.append(
                self._mention(
                    MentionKind.PROJECT_ID,
                    m,
                    pid,
                    pid if pid in self.registry else None,
                    "project registry",
                )
            )
        for m in _LOAN.finditer(text):
            norm = f"{m.group(1).upper()}{m.group(2)}{m.group(3)}"
            found.append(
                self._mention(
                    MentionKind.LOAN_NUMBER,
                    m,
                    norm,
                    self.loans.get(norm),
                    "registry expected_loan_numbers",
                )
            )
        for m in _REPORT.finditer(text):
            norm = f"{m.group(1).upper()}{m.group(2)}"
            found.append(
                self._mention(
                    MentionKind.REPORT_NUMBER,
                    m,
                    norm,
                    self.reports.get(norm),
                    "document manifest report_number",
                )
            )
        for pattern in self.relative:
            for m in pattern.finditer(text):
                found.append(
                    self._mention(
                        MentionKind.RELATIVE_REFERENCE,
                        m,
                        m.group(0).lower(),
                        None,
                        "relative project reference",
                    )
                )
        lowered = text.lower()
        for alias, pid in sorted(self.aliases.items(), key=lambda a: -len(a[0])):
            pattern = r"\b" + r"\s+".join(re.escape(w) for w in alias.split()) + r"\b"
            for m in re.finditer(pattern, lowered):
                found.append(self._mention(MentionKind.ALIAS, m, alias, pid, "reviewed alias"))
        return _longest_non_overlapping(found)

    @staticmethod
    def _mention(kind, match, normalized, project_id, basis) -> Mention:
        return Mention(
            kind=kind,
            span=(match.start(), match.end()),
            text=match.group(0),
            normalized=normalized,
            project_id=project_id,
            basis=basis,
        )


def _longest_non_overlapping(mentions: list[Mention]) -> list[Mention]:
    ordered = sorted(mentions, key=lambda m: (-(m.span[1] - m.span[0]), m.span[0], m.kind))
    kept: list[Mention] = []
    for m in ordered:
        if all(m.span[1] <= k.span[0] or m.span[0] >= k.span[1] for k in kept):
            kept.append(m)
    return sorted(kept, key=lambda m: m.span)


def resolve_project(text: str, access: AccessContext, index: EntityIndex) -> ProjectResolution:
    mentions = tuple(index.mentions(text))
    owners = sorted({m.project_id for m in mentions if m.project_id})
    relative = [m for m in mentions if m.kind == MentionKind.RELATIVE_REFERENCE]
    unknown = [
        m for m in mentions if m.project_id is None and m.kind != MentionKind.RELATIVE_REFERENCE
    ]
    active = access.active_project_id

    def out(status: ProjectStatus, project_id: str | None, basis: str | None, detail: str):
        return ProjectResolution(
            status=status,
            project_id=project_id,
            basis=basis,
            active_project_id=active,
            mentions=mentions,
            mentioned_projects=tuple(owners),
            detail=detail,
        )

    if unknown:
        names = ", ".join(f"{m.kind.value} {m.text}" for m in unknown)
        return out(
            ProjectStatus.UNSUPPORTED,
            None,
            None,
            f"not in the approved corpus or of unknown ownership: {names}",
        )
    if active is not None and active not in index.registry:
        return out(
            ProjectStatus.UNSUPPORTED,
            None,
            None,
            f"active scope {active} is not in the approved corpus",
        )
    if len(owners) >= 2:
        return out(
            ProjectStatus.MULTI,
            None,
            None,
            f"the question refers to several projects {owners}; one project per request",
        )
    if active is not None and owners and owners[0] != active:
        return out(
            ProjectStatus.CONFLICT,
            None,
            None,
            f"the question refers to {owners[0]} but the active scope is {active}; "
            "the scope is never switched implicitly",
        )
    if relative:
        # A relative reference never selects a project by itself and never falls back to
        # the active scope. Only when an active scope leaves exactly ONE other authorised
        # project is the referent unique - and then it is a foreign project (refused).
        if active is not None and active not in set(access.authorized_projects):
            return out(
                ProjectStatus.NOT_AUTHORIZED,
                None,
                None,
                f"project {active} is not authorised for this user",
            )
        others = sorted(set(access.authorized_projects) - {active}) if active else []
        if active is not None and len(others) == 1:
            return ProjectResolution(
                status=ProjectStatus.CONFLICT,
                project_id=None,
                basis=None,
                active_project_id=active,
                mentions=mentions,
                mentioned_projects=tuple(sorted({*owners, others[0]})),
                detail=f"{relative[0].text!r} can only mean {others[0]} (the one other "
                f"authorised project), which differs from the active scope {active}; "
                "the scope is never switched implicitly",
            )
        return out(
            ProjectStatus.AMBIGUOUS_REFERENCE,
            None,
            None,
            f"{relative[0].text!r} does not identify a project "
            f"({len(others) if active else 'no active scope and'} other authorised "
            f"project(s)); the active scope is not substituted",
        )
    if active is not None:
        project, basis = active, "ACTIVE_SCOPE"
    elif owners:
        project = owners[0]
        basis = next(m.kind.value for m in mentions if m.project_id == project)
    else:
        return out(ProjectStatus.MISSING, None, None, "no active project and no project mentioned")
    if project not in set(access.authorized_projects):
        return out(
            ProjectStatus.NOT_AUTHORIZED,
            None,
            None,
            f"project {project} is not authorised for this user",
        )
    return out(ProjectStatus.RESOLVED, project, basis, f"scope {project} ({basis})")


__all__ = ["EntityIndex", "normalize_phrase", "resolve_project", "validate_input"]
