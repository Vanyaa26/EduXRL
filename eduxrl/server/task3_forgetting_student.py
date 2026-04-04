"""
Task 3: HARD — The Forgetting Student

Full curriculum across 3 sessions spread over a simulated week.
Knowledge decays between sessions (Ebbinghaus curve).
Must balance new content vs review.

Tests: spaced review planning, long-term retention, prerequisite
       management across sessions, fatigue management.
"""

TASK3_DATA = {
    "id": "task3",
    "name": "The Forgetting Student",
    "difficulty": "hard",
    "description": (
        "Teach the full 10-topic Python curriculum across 3 sessions spread "
        "over a simulated week (Day 1, Day 3, Day 7). Knowledge decays between "
        "sessions following the Ebbinghaus forgetting curve. Balance teaching "
        "new content against reviewing forgotten material. Plan ahead — "
        "prerequisites taught in Session 1 are needed for advanced topics in Session 3."
    ),
    "curriculum_topics": [
        "variables",
        "data_types",
        "conditionals",
        "strings",
        "loops",
        "lists",
        "functions",
        "dictionaries",
        "file_io",
        "error_handling",
    ],
    "sessions": [
        {"session_number": 1, "max_steps": 20, "days_since_last": 0},
        {"session_number": 2, "max_steps": 20, "days_since_last": 2},
        {"session_number": 3, "max_steps": 20, "days_since_last": 4},
    ],
    "student_profile": {
        "initial_knowledge": {},
        "initial_motivation": 0.7,
        "fatigue_rate": 1.0,  # increases across sessions (handled in environment)
        "preferred_format": "worked_example",
        "learning_rate": 1.0,
    },
}
