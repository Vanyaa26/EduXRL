"""
EduXRL Client.

Client for connecting to a running EduXRL server.
Extends HTTPEnvClient for type-safe interactions.

Usage:
    >>> from eduxrl import EduXRLEnv
    >>> from eduxrl.models import TeachingAction
    >>>
    >>> env = EduXRLEnv(base_url="http://localhost:8000")
    >>> result = env.reset()
    >>> result = env.step(TeachingAction(
    ...     action_type="teach", topic="variables",
    ...     format="exercise", difficulty="easy",
    ... ))
"""

from typing import Any, Dict

try:
    from openenv.core import EnvClient as HTTPEnvClient
    from openenv.core.env_client import StepResult
except ImportError:
    try:
        from openenv_core.http_env_client import HTTPEnvClient
        from openenv_core.client_types import StepResult
    except ImportError:
        HTTPEnvClient = None
        StepResult = None

if HTTPEnvClient is None or StepResult is None:
    from dataclasses import dataclass
    from typing import Generic, Optional, TypeVar
    from abc import ABC, abstractmethod

    ObsT = TypeVar("ObsT")

    if StepResult is None:
        @dataclass
        class StepResult(Generic[ObsT]):
            observation: ObsT
            reward: Optional[float] = None
            done: bool = False

    if HTTPEnvClient is None:
        class HTTPEnvClient(ABC):
            def __init__(self, base_url: str, **kwargs):
                self._base = base_url.rstrip("/")

            @abstractmethod
            def _step_payload(self, action) -> dict: ...

            @abstractmethod
            def _parse_result(self, payload: dict) -> StepResult: ...

            @abstractmethod
            def _parse_state(self, payload: dict): ...

from .models import TeachingAction, StudentObservation, SessionState


class EduXRLEnv(HTTPEnvClient):
    """Client for the EduXRL Adaptive Learning Environment."""

    def _step_payload(self, action: TeachingAction) -> dict:
        if hasattr(action, "model_dump"):
            return action.model_dump(exclude_none=True)
        from dataclasses import asdict
        d = asdict(action)
        return {k: v for k, v in d.items() if v is not None}

    def _parse_result(self, payload: dict) -> StepResult:
        obs_data = payload.get("observation", {})
        observation = StudentObservation(**obs_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> SessionState:
        return SessionState(**payload)
