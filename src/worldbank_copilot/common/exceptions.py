"""Exception hierarchy for the copilot.

Errors are raised explicitly rather than papered over: a missing configuration
value or an unknown project must fail clearly, never fall back to a guess.
"""


class CopilotError(Exception):
    """Base class for all application errors."""


class ConfigurationError(CopilotError):
    """Configuration is missing, malformed or cannot be resolved."""


class SourceContractError(CopilotError):
    """A source file does not satisfy its declared contract (headers, required columns)."""


class SourceFileError(CopilotError):
    """A source file cannot be located unambiguously or read."""


class UnknownProjectError(CopilotError):
    """A project ID is not part of the configured prototype scope."""

    def __init__(self, project_id: str, allowed: list[str]):
        self.project_id = project_id
        self.allowed = allowed
        super().__init__(
            f"Project {project_id!r} is not in scope. Allowed projects: {', '.join(allowed)}"
        )


class LakehouseError(CopilotError):
    """A Unity Catalog / Delta operation failed or an object is not as expected."""


class SourceIntegrityError(LakehouseError):
    """Source files differ from the validated source snapshot (hash mismatch or missing)."""


class ReconciliationError(LakehouseError):
    """Persisted data does not reconcile with the validated in-memory / expected data."""
