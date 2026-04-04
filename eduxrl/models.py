"""
Data models for the EduXRL Environment.

Typed models for the Adaptive Learning Path Environment:
- EduxrlAction: Actions the teaching agent can take
- EduxrlObservation: What the agent observes after each step
"""

from typing import Any, Dict, List, Optional, Union
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class EduxrlAction(Action):
    """Action taken by the teaching agent."""

    action_type: str = Field(default="teach", description="Action type", json_schema_extra={"placeholder": "teach / quiz / review / end_session", "examples": ["teach", "quiz", "review", "end_session"]})
    topic: str = Field(default="variables", description="Topic name", json_schema_extra={"placeholder": "variables / conditionals / loops / functions ...", "examples": ["variables", "conditionals", "loops", "data_types", "strings", "lists", "functions"]})
    format: str = Field(default="exercise", description="Content format", json_schema_extra={"placeholder": "explanation / worked_example / exercise", "examples": ["explanation", "worked_example", "exercise"]})
    difficulty: str = Field(default="easy", description="Difficulty level", json_schema_extra={"placeholder": "easy / medium / hard", "examples": ["easy", "medium", "hard"]})


class EduxrlObservation(Observation):
    """What the teaching agent observes after each step."""

    # Curriculum info
    available_topics: List[str] = Field(default_factory=list)
    prerequisite_map: Dict[str, List[str]] = Field(default_factory=dict)
    all_topics: List[str] = Field(default_factory=list)

    # Student knowledge
    topic_scores: Dict[str, float] = Field(default_factory=dict)
    topics_taught: List[str] = Field(default_factory=list)
    topics_quizzed: List[str] = Field(default_factory=list)

    # Behavioral signals
    last_action_result: Dict[str, Any] = Field(default_factory=dict)
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
