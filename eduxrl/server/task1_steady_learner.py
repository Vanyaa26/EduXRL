"""
Task 1: EASY — The Steady Learner

A fresh student learning 5 Python topics in a single session.
Average ability, stable motivation, no prior knowledge.

Tests: basic prerequisite ordering, difficulty matching,
       not wasting time on mastered content.
"""

TASK1_DATA = {
    "id": "task1",
    "name": "The Steady Learner",
    "difficulty": "easy",
    "description": (
        "Teach a fresh student 5 Python topics in one session. "
        "The student has no prior knowledge and stable motivation. "
        "Sequence lessons properly, match difficulty to current knowledge, "
        "and avoid wasting time on already-mastered content."
    ),
    "curriculum_topics": [
        "variables",
        "data_types",
        "conditionals",
        "strings",
        "loops",
    ],
    "sessions": [
        {"session_number": 1, "max_steps": 20, "days_since_last": 0},
    ],
    "student_profile": {
        "initial_knowledge": {},  # all zero
        "initial_motivation": 0.8,
        "fatigue_rate": 1.0,
        "preferred_format": "exercise",  # mild preference, agent doesn't know
        "learning_rate": 1.0,
    },
}
