"""
Task Definitions Registry for EduXRL.

Loads and provides access to all learning tasks with their
student profiles, curriculum subsets, and session configurations.
"""

from typing import Any, Dict, List, Optional

from .task1_steady_learner import TASK1_DATA
from .task2_struggling_student import TASK2_DATA
from .task3_forgetting_student import TASK3_DATA


class TaskDefinition:
    """Structured representation of a learning task."""

    def __init__(self, data: Dict[str, Any]):
        self.id: str = data["id"]
        self.name: str = data["name"]
        self.difficulty: str = data["difficulty"]
        self.description: str = data["description"]
        self.curriculum_topics: List[str] = data["curriculum_topics"]
        self.sessions: List[Dict[str, Any]] = data["sessions"]
        self.student_profile: Dict[str, Any] = data["student_profile"]

    @property
    def num_sessions(self) -> int:
        return len(self.sessions)

    @property
    def total_max_steps(self) -> int:
        return sum(s["max_steps"] for s in self.sessions)

    def get_session(self, session_number: int) -> Optional[Dict[str, Any]]:
        for s in self.sessions:
            if s["session_number"] == session_number:
                return s
        return None

    def to_summary(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "difficulty": self.difficulty,
            "description": self.description,
            "num_topics": len(self.curriculum_topics),
            "num_sessions": self.num_sessions,
            "total_steps": self.total_max_steps,
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_TASK_REGISTRY: Dict[str, TaskDefinition] = {
    "task1": TaskDefinition(TASK1_DATA),
    "task2": TaskDefinition(TASK2_DATA),
    "task3": TaskDefinition(TASK3_DATA),
}


def get_task(task_id: str) -> TaskDefinition:
    if task_id not in _TASK_REGISTRY:
        raise KeyError(f"Unknown task_id '{task_id}'. Available: {list(_TASK_REGISTRY.keys())}")
    return _TASK_REGISTRY[task_id]


def get_all_tasks() -> Dict[str, TaskDefinition]:
    return dict(_TASK_REGISTRY)


def list_task_ids() -> List[str]:
    return list(_TASK_REGISTRY.keys())
