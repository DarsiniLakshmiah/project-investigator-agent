"""Interview-prototype runtime: ``Copilot.investigate(query=..., project_id=...)``."""

from worldbank_copilot.copilot.config import CopilotConfig, load_copilot_config
from worldbank_copilot.copilot.contracts import CriticStatus, InvestigationResult, ResultStatus
from worldbank_copilot.copilot.service import Copilot

__all__ = [
    "Copilot",
    "CopilotConfig",
    "CriticStatus",
    "InvestigationResult",
    "ResultStatus",
    "load_copilot_config",
]
