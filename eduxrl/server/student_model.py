"""
Simulated Student Model for EduXRL.

The "physics engine" of the environment. A mathematical model of a student
whose learning, forgetting, motivation, and fatigue follow published
cognitive science research — not random numbers or human opinion.

Models used:
    - Power Law of Learning (Newell & Rosenbloom, 1981)
    - Ebbinghaus Forgetting Curve (Ebbinghaus, 1885)
    - Zone of Proximal Development (Vygotsky, 1978)
    - Spacing Effect (Cepeda et al., 2006)
    - Session fatigue (educational psychology)

The student has NO intelligence. It responds deterministically to teaching
actions based on these formulas, like gravity responds to mass in CartPole.
"""

import math
import random
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Difficulty string → numeric mapping
# ---------------------------------------------------------------------------

DIFFICULTY_MAP = {"easy": 0.3, "medium": 0.6, "hard": 0.9}

# ---------------------------------------------------------------------------
# Format preference multipliers
# ---------------------------------------------------------------------------

FORMAT_PREFERRED_BONUS = 1.3
FORMAT_NEUTRAL = 1.0
FORMAT_NON_PREFERRED_PENALTY = 0.7


class SimulatedStudent:
    """
    A simulated student governed by cognitive science models.

    Attributes:
        knowledge: Per-topic knowledge level [0.0, 1.0]
        motivation: Current motivation [0.0, 1.0]
        fatigue: Current fatigue [0.0, 1.0]
        preferred_format: Hidden learning style preference
        review_counts: How many times each topic has been reviewed (affects spacing)
    """

    def __init__(
        self,
        topic_names: List[str],
        initial_knowledge: Optional[Dict[str, float]] = None,
        initial_motivation: float = 0.8,
        fatigue_rate: float = 1.0,
        preferred_format: str = "exercise",
        learning_rate: float = 1.0,
        seed: Optional[int] = None,
    ):
        self._rng = random.Random(seed)
        self._topic_names = topic_names

        # Knowledge per topic [0.0, 1.0]
        self.knowledge: Dict[str, float] = {}
        for t in topic_names:
            self.knowledge[t] = (initial_knowledge or {}).get(t, 0.0)

        # Internal state
        self.motivation = initial_motivation
        self.fatigue = 0.0
        self.fatigue_rate = fatigue_rate
        self.preferred_format = preferred_format
        self.learning_rate = learning_rate

        # Tracking for spacing effect
        self.review_counts: Dict[str, int] = {t: 0 for t in topic_names}
        self.last_taught_step: Dict[str, int] = {}
        self._step = 0

        # Behavioral signals (observable by agent)
        self.last_quiz_score: Optional[float] = None
        self.consecutive_failures: int = 0
        self.consecutive_successes: int = 0
        self.actions_since_last_success: int = 0
        self.topics_taught: List[str] = []
        self.topics_quizzed: List[str] = []

    # -------------------------------------------------------------------
    # Core cognitive models
    # -------------------------------------------------------------------

    def receive_teaching(
        self,
        topic: str,
        format: str,
        difficulty: str,
        topic_base_difficulty: float,
    ) -> Dict[str, Any]:
        """
        Process a teaching action. Updates knowledge using the Power Law
        of Learning with modifiers for difficulty match, format preference,
        motivation, and fatigue.

        Returns dict with details of what happened.
        """
        self._step += 1
        if topic not in self.knowledge:
            return {"error": f"Unknown topic: {topic}", "knowledge_gain": 0.0}

        current_k = self.knowledge[topic]
        diff_numeric = DIFFICULTY_MAP.get(difficulty, 0.5)

        # --- Zone of Proximal Development (Vygotsky) ---
        # Optimal difficulty is slightly above current knowledge
        optimal_difficulty = min(1.0, current_k + 0.2)
        zpd_match = 1.0 - abs(diff_numeric - optimal_difficulty)
        zpd_match = max(0.2, zpd_match)  # floor at 0.2

        # --- Format preference ---
        if format == self.preferred_format:
            format_mult = FORMAT_PREFERRED_BONUS
        else:
            format_mult = FORMAT_NON_PREFERRED_PENALTY

        # --- Motivation and fatigue effects ---
        motivation_mult = 0.3 + 0.7 * self.motivation
        fatigue_mult = max(0.1, 1.0 - self.fatigue)

        # --- Power Law of Learning ---
        # ΔK = base_rate × (1 - K)^0.5 × modifiers
        # Diminishing returns as knowledge approaches 1.0
        base_rate = 0.15 * self.learning_rate
        effectiveness = (
            base_rate
            * zpd_match
            * format_mult
            * motivation_mult
            * fatigue_mult
        )
        delta_k = effectiveness * math.pow(max(0.01, 1.0 - current_k), 0.5)

        # Small noise for realism
        noise = self._rng.gauss(0, 0.01)
        delta_k = max(0.0, delta_k + noise)

        self.knowledge[topic] = min(1.0, current_k + delta_k)

        # Track
        if topic not in self.topics_taught:
            self.topics_taught.append(topic)
        self.review_counts[topic] = self.review_counts.get(topic, 0) + 1
        self.last_taught_step[topic] = self._step

        # --- Update motivation ---
        if delta_k > 0.05:
            # Good learning → motivation boost
            self.motivation = min(1.0, self.motivation + 0.05)
            self.consecutive_successes += 1
            self.consecutive_failures = 0
            self.actions_since_last_success = 0
        elif diff_numeric > current_k + 0.4:
            # Way too hard → motivation drop
            self.motivation = max(0.0, self.motivation - 0.12)
            self.consecutive_failures += 1
            self.consecutive_successes = 0
            self.actions_since_last_success += 1
        elif diff_numeric < current_k - 0.3:
            # Too easy → boredom
            self.motivation = max(0.0, self.motivation - 0.03)
            self.actions_since_last_success += 1
        else:
            self.actions_since_last_success += 1

        # --- Update fatigue ---
        fatigue_increment = 0.03  # base
        if difficulty == "hard":
            fatigue_increment = 0.07
        elif difficulty == "medium":
            fatigue_increment = 0.05
        if format != self.preferred_format:
            fatigue_increment += 0.02
        self.fatigue = min(1.0, self.fatigue + fatigue_increment * self.fatigue_rate)

        return {
            "knowledge_before": round(current_k, 4),
            "knowledge_after": round(self.knowledge[topic], 4),
            "knowledge_gain": round(delta_k, 4),
            "zpd_match": round(zpd_match, 4),
            "format_bonus": format == self.preferred_format,
            "effective_learning_rate": round(effectiveness, 4),
        }

    def take_quiz(
        self,
        topic: str,
        difficulty: str,
    ) -> Dict[str, Any]:
        """
        Simulate a quiz. Score depends on knowledge + difficulty + noise.
        Reveals the student's actual knowledge level (with some variance).
        """
        self._step += 1
        if topic not in self.knowledge:
            return {"error": f"Unknown topic: {topic}", "score": 0.0}

        current_k = self.knowledge[topic]
        diff_numeric = DIFFICULTY_MAP.get(difficulty, 0.5)

        # Base score = knowledge level
        # Harder difficulty penalizes score
        difficulty_penalty = max(0, diff_numeric - current_k) * 0.4
        noise = self._rng.gauss(0, 0.08)

        score = current_k - difficulty_penalty + noise
        score = max(0.0, min(1.0, score))

        # Track
        if topic not in self.topics_quizzed:
            self.topics_quizzed.append(topic)
        self.last_quiz_score = score

        # Update motivation based on score
        if score >= 0.7:
            self.motivation = min(1.0, self.motivation + 0.06)
            self.consecutive_successes += 1
            self.consecutive_failures = 0
            self.actions_since_last_success = 0
        elif score < 0.4:
            self.motivation = max(0.0, self.motivation - 0.10)
            self.consecutive_failures += 1
            self.consecutive_successes = 0
            self.actions_since_last_success += 1
        else:
            self.actions_since_last_success += 1

        # Fatigue from quiz effort
        self.fatigue = min(1.0, self.fatigue + 0.03 * self.fatigue_rate)

        return {
            "score": round(score, 4),
            "knowledge_level": round(current_k, 4),
            "difficulty": difficulty,
        }

    def receive_review(
        self,
        topic: str,
        format: str,
        difficulty: str,
        topic_base_difficulty: float,
    ) -> Dict[str, Any]:
        """
        Review a previously taught topic. Same mechanics as teaching,
        but with a slight bonus for spaced repetition.
        """
        # Spacing effect: reviews are slightly more effective
        result = self.receive_teaching(topic, format, difficulty, topic_base_difficulty)
        if "error" not in result:
            # Spacing bonus: +20% to what was already gained
            bonus = result["knowledge_gain"] * 0.2
            self.knowledge[topic] = min(1.0, self.knowledge[topic] + bonus)
            result["knowledge_after"] = round(self.knowledge[topic], 4)
            result["knowledge_gain"] = round(result["knowledge_gain"] + bonus, 4)
            result["spacing_bonus"] = True
        return result

    def apply_forgetting(self, days: int) -> Dict[str, float]:
        """
        Apply Ebbinghaus forgetting curve across all topics.

        R(t) = e^(-t/S) where:
            t = time elapsed (days)
            S = stability (increases with review count)

        Returns dict of {topic: knowledge_lost}
        """
        decay_report: Dict[str, float] = {}
        for topic in self._topic_names:
            k_before = self.knowledge[topic]
            if k_before <= 0.01:
                continue

            # Stability increases with reviews (spacing effect)
            reviews = self.review_counts.get(topic, 0)
            base_stability = 3.0  # base half-life in days
            stability = base_stability * (1.0 + 0.5 * reviews)

            # Ebbinghaus: R = e^(-t/S)
            retention = math.exp(-days / stability)
            self.knowledge[topic] = k_before * retention

            lost = k_before - self.knowledge[topic]
            if lost > 0.01:
                decay_report[topic] = round(lost, 4)

        return decay_report

    def end_session(self) -> Dict[str, Any]:
        """
        Process end-of-session. Returns session summary.
        """
        return {
            "motivation_at_end": round(self.motivation, 4),
            "fatigue_at_end": round(self.fatigue, 4),
            "topics_taught": list(self.topics_taught),
            "topics_quizzed": list(self.topics_quizzed),
            "final_knowledge": {t: round(v, 4) for t, v in self.knowledge.items()},
        }

    def start_new_session(self, motivation_boost: float = 0.0) -> None:
        """Reset session-level state (fatigue) while keeping knowledge and motivation."""
        self.fatigue = 0.0
        self.motivation = min(1.0, self.motivation + motivation_boost)
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.actions_since_last_success = 0
        self.last_quiz_score = None

    @property
    def is_disengaged(self) -> bool:
        """Student has effectively quit (motivation too low)."""
        return self.motivation < 0.15

    @property
    def is_exhausted(self) -> bool:
        """Student is too fatigued to learn effectively."""
        return self.fatigue > 0.85

    def get_observable_state(self) -> Dict[str, Any]:
        """Return what the agent CAN see (no motivation/fatigue directly)."""
        return {
            "last_quiz_score": self.last_quiz_score,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "actions_since_last_success": self.actions_since_last_success,
            "topics_taught": list(self.topics_taught),
            "topics_quizzed": list(self.topics_quizzed),
        }
