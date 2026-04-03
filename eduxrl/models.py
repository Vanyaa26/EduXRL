"""
EduXRL Data Models.

Typed models for the Adaptive Learning Path Environment:
- TeachingAction: Actions the teaching agent can take
- StudentObservation: What the agent observes after each step
- SessionState: Episode metadata

These extend the OpenEnv base classes (Pydantic BaseModels).
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union

# ---------------------------------------------------------------------------
# Import base classes from OpenEnv
# ---------------------------------------------------------------------------

try:
    from openenv.core import Action, Observation, State
except ImportError:
    try:
        from openenv_core.env_server.types import Action, Observation, State
    except ImportError:
        from pydantic import BaseModel, Field, ConfigDict

        class Action(BaseModel):
            model_config = ConfigDict(extra="forbid")
            metadata: Dict[str, Any] = Field(default_factory=dict)

        class Observation(BaseModel):
            model_config = ConfigDict(extra="forbid")
            done: bool = False
            reward: Union[bool, int, float, None] = None
            metadata: Dict[str, Any] = Field(default_factory=dict)

        class State(BaseModel):
            model_config = ConfigDict(extra="allow")
            episode_id: Optional[str] = None
            step_count: int = 0


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ActionType(str, Enum):
    TEACH = "teach"
    QUIZ = "quiz"
    REVIEW = "review"
    END_SESSION = "end_session"


class ContentFormat(str, Enum):
    EXPLANATION = "explanation"
    WORKED_EXAMPLE = "worked_example"
    EXERCISE = "exercise"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

from pydantic import ConfigDict as _ConfigDict


class TeachingAction(Action):
    """
    An action taken by the teaching agent.

    action_type determines what happens:
    - teach: present new content (topic + format + difficulty)
    - quiz: test student on a topic (topic + difficulty)
    - review: re-teach a previously covered topic (topic + format + difficulty)
    - end_session: end the current session
    """

    model_config = _ConfigDict(extra="forbid")

    action_type: str = ""
    topic: str = ""
    format: str = "explanation"
    difficulty: str = "medium"


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


class StudentObservation(Observation):
    """
    What the teaching agent observes after each step.

    The agent does NOT see motivation or fatigue directly.
    It must infer student state from behavioral signals.
    """

    model_config = _ConfigDict(extra="forbid")

    # Curriculum info
    available_topics: List[str] = []
    prerequisite_map: Dict[str, List[str]] = {}
    all_topics: List[str] = []

    # Student knowledge (from quiz scores)
    topic_scores: Dict[str, float] = {}
    topics_taught: List[str] = []
    topics_quizzed: List[str] = []

    # Behavioral signals (agent infers motivation/fatigue from these)
    last_action_result: Dict[str, Any] = {}
    last_quiz_score: Optional[float] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    actions_since_last_success: int = 0

    # Session info
    step_number: int = 0
    steps_remaining: int = 0
    session_number: int = 1
    days_since_last_session: int = 0
    task_id: str = ""
    task_description: str = ""


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class SessionState(State):
    """
    Current state of the learning session episode.
    """

    task_id: str = ""
    session_number: int = 1
    topics_covered: List[str] = []
    total_knowledge_gain: float = 0.0
    total_reward: float = 0.0
    actions_taken: int = 0
    max_steps: int = 20
