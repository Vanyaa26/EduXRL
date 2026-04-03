"""
Curriculum definitions for EduXRL.

Defines the Python programming curriculum: topics, prerequisites,
difficulty ratings, and content metadata. The environment engine is
subject-agnostic — this module provides the specific curriculum data.
"""

from typing import Any, Dict, List, Optional


class Topic:
    """A single topic in the curriculum."""

    def __init__(
        self,
        name: str,
        display_name: str,
        prerequisites: List[str],
        base_difficulty: float,
        description: str,
    ):
        self.name = name
        self.display_name = display_name
        self.prerequisites = prerequisites
        self.base_difficulty = base_difficulty
        self.description = description

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "prerequisites": self.prerequisites,
            "base_difficulty": self.base_difficulty,
            "description": self.description,
        }


# ---------------------------------------------------------------------------
# Python Programming Curriculum — 10 topics
# ---------------------------------------------------------------------------

PYTHON_CURRICULUM: List[Topic] = [
    Topic(
        name="variables",
        display_name="Variables & Assignment",
        prerequisites=[],
        base_difficulty=0.15,
        description="Variable declaration, assignment, naming conventions, basic types (int, float, str, bool).",
    ),
    Topic(
        name="data_types",
        display_name="Data Types & Type Conversion",
        prerequisites=["variables"],
        base_difficulty=0.20,
        description="Type checking, casting between types, None, type(), isinstance().",
    ),
    Topic(
        name="conditionals",
        display_name="Conditionals & Boolean Logic",
        prerequisites=["variables"],
        base_difficulty=0.30,
        description="if/elif/else, comparison operators, boolean operators (and, or, not), truthiness.",
    ),
    Topic(
        name="strings",
        display_name="String Operations",
        prerequisites=["variables", "data_types"],
        base_difficulty=0.25,
        description="String methods, slicing, f-strings, concatenation, escape characters.",
    ),
    Topic(
        name="loops",
        display_name="Loops & Iteration",
        prerequisites=["conditionals"],
        base_difficulty=0.40,
        description="for loops, while loops, range(), break/continue, nested loops.",
    ),
    Topic(
        name="lists",
        display_name="Lists & List Operations",
        prerequisites=["variables", "loops"],
        base_difficulty=0.45,
        description="List creation, indexing, slicing, append/extend/insert, list comprehensions.",
    ),
    Topic(
        name="functions",
        display_name="Functions & Scope",
        prerequisites=["conditionals", "loops"],
        base_difficulty=0.50,
        description="def, parameters, return values, default args, scope (local/global), lambda.",
    ),
    Topic(
        name="dictionaries",
        display_name="Dictionaries & Sets",
        prerequisites=["variables", "lists"],
        base_difficulty=0.45,
        description="Dict creation, keys/values, iteration, get(), set operations.",
    ),
    Topic(
        name="file_io",
        display_name="File I/O",
        prerequisites=["functions", "strings"],
        base_difficulty=0.55,
        description="open(), read/write modes, with statement, reading lines, CSV basics.",
    ),
    Topic(
        name="error_handling",
        display_name="Error Handling",
        prerequisites=["functions", "conditionals"],
        base_difficulty=0.50,
        description="try/except, exception types, finally, raising exceptions, custom exceptions.",
    ),
]


class Curriculum:
    """Manages a set of topics with prerequisite relationships."""

    def __init__(self, topics: Optional[List[Topic]] = None):
        self._topics = {t.name: t for t in (topics or PYTHON_CURRICULUM)}

    @property
    def all_topic_names(self) -> List[str]:
        return list(self._topics.keys())

    @property
    def num_topics(self) -> int:
        return len(self._topics)

    def get_topic(self, name: str) -> Optional[Topic]:
        return self._topics.get(name)

    def get_prerequisites(self, topic_name: str) -> List[str]:
        topic = self._topics.get(topic_name)
        return topic.prerequisites if topic else []

    def get_base_difficulty(self, topic_name: str) -> float:
        topic = self._topics.get(topic_name)
        return topic.base_difficulty if topic else 0.5

    def get_prerequisite_map(self) -> Dict[str, List[str]]:
        return {name: t.prerequisites for name, t in self._topics.items()}

    def get_unlocked_topics(self, knowledge: Dict[str, float], threshold: float = 0.5) -> List[str]:
        """Return topics whose prerequisites are all at or above threshold."""
        unlocked = []
        for name, topic in self._topics.items():
            prereqs_met = all(
                knowledge.get(p, 0.0) >= threshold
                for p in topic.prerequisites
            )
            if prereqs_met:
                unlocked.append(name)
        return unlocked

    def get_subset(self, topic_names: List[str]) -> "Curriculum":
        """Return a curriculum with only the specified topics."""
        subset = [self._topics[n] for n in topic_names if n in self._topics]
        return Curriculum(subset)

    def to_dict_list(self) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self._topics.values()]
