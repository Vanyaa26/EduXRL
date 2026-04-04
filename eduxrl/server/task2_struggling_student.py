"""
Task 2: MEDIUM — The Struggling Student

A student with prior knowledge AND gaps. Sensitive motivation.
Has a strong learning style preference the agent must discover.

Tests: gap detection, motivation management, format discovery,
       adaptivity after student struggles.
"""

TASK2_DATA = {
    "id": "task2",
    "name": "The Struggling Student",
    "difficulty": "medium",
    "description": (
        "A student who already knows some topics but has hidden gaps. "
        "Motivation drops quickly on failure. The student strongly prefers "
        "one content format over others — discover it through trial and observation. "
        "Fill the gaps, keep the student engaged, and push forward carefully."
    ),
    "curriculum_topics": [
        "variables",
        "data_types",
        "conditionals",
        "strings",
        "loops",
        "lists",
        "functions",
    ],
    "sessions": [
        {"session_number": 1, "max_steps": 25, "days_since_last": 0},
    ],
    "student_profile": {
        "initial_knowledge": {
            "variables": 0.7,
            "data_types": 0.3,   # GAP — weak despite knowing variables
            "conditionals": 0.15,
        },
        "initial_motivation": 0.6,  # starts lower, more fragile
        "fatigue_rate": 1.2,  # gets tired faster
        "preferred_format": "exercise",  # STRONG preference — agent must discover
        "learning_rate": 0.9,
    },
}
