"""
Grader for EduXRL.

Computes per-step rewards and final episode scores across 5 dimensions.
All rewards are derived from observable, measurable student outcomes —
no human labels or opinion.

Dimensions:
    - Knowledge Acquisition (30%): how much the student learned
    - Retention (25%): knowledge that persists after forgetting
    - Engagement (20%): session completion + motivation preserved
    - Efficiency (15%): useful actions / total actions
    - Adaptivity (10%): strategy changes after student struggles
"""

from typing import Any, Dict, List, Optional, Set

try:
    from ..tasks.task_definitions import TaskDefinition
except ImportError:
    from tasks.task_definitions import TaskDefinition


# ---------------------------------------------------------------------------
# Reward constants
# ---------------------------------------------------------------------------

# Teaching rewards (proportional to knowledge gain, scaled)
REWARD_KNOWLEDGE_SCALE = 2.0
REWARD_REVIEW_SCALE = 2.5

# Quiz rewards
REWARD_FIRST_QUIZ = 0.05
REWARD_REPEAT_QUIZ = 0.02
PENALTY_REDUNDANT_QUIZ = -0.03

# Session management
REWARD_GOOD_SESSION_END = 0.10
PENALTY_EARLY_SESSION_END = -0.10

# Penalties
PENALTY_WASTED_TIME = -0.05  # teaching mastered content
PENALTY_PREREQ_VIOLATION = -0.10  # teaching without prerequisites
PENALTY_FRUSTRATION = -0.08  # causing quiz score < 0.25

# Grading weights
GRADING_WEIGHTS = {
    "knowledge_acquisition": 0.30,
    "retention": 0.25,
    "engagement": 0.20,
    "efficiency": 0.15,
    "adaptivity": 0.10,
}


class LearningGrader:
    """
    Deterministic grader for the adaptive learning environment.

    Tracks all actions and computes per-step rewards plus a final
    normalized score in [0.0, 1.0].
    """

    def __init__(self, task: TaskDefinition, curriculum_topics: List[str]):
        self.task = task
        self.curriculum_topics = curriculum_topics
        self.reset()

    def reset(self) -> None:
        self._step_rewards: List[float] = []
        self._actions: List[Dict[str, Any]] = []
        self._useful_actions: int = 0
        self._total_actions: int = 0
        self._quizzed_topics: Set[str] = set()
        self._last_quizzed_topic: Optional[str] = None
        self._taught_since_last_quiz: bool = False
        self._strategy_changes_after_struggle: int = 0
        self._struggle_opportunities: int = 0
        self._last_action_format: Optional[str] = None
        self._last_action_difficulty: Optional[str] = None
        self._consecutive_failures_at_last_action: int = 0
        self._initial_knowledge: Dict[str, float] = {}
        self._sessions_completed: int = 0
        self._total_sessions: int = 0
        self._motivation_at_session_ends: List[float] = []

    def set_initial_knowledge(self, knowledge: Dict[str, float]) -> None:
        self._initial_knowledge = dict(knowledge)

    # -------------------------------------------------------------------
    # Per-step grading
    # -------------------------------------------------------------------

    def grade_action(
        self,
        action: Dict[str, Any],
        result: Dict[str, Any],
        student_knowledge: Dict[str, float],
        student_observable: Dict[str, Any],
        available_topics: List[str],
    ) -> float:
        """Grade a single action and return the step reward."""
        self._total_actions += 1
        action_type = action.get("action_type", "")
        topic = action.get("topic", "")

        # Track adaptivity: did agent change strategy after student struggled?
        current_failures = student_observable.get("consecutive_failures", 0)
        if self._consecutive_failures_at_last_action >= 2:
            self._struggle_opportunities += 1
            current_format = action.get("format", "")
            current_difficulty = action.get("difficulty", "")
            if (current_format != self._last_action_format or
                    current_difficulty != self._last_action_difficulty or
                    action_type != "teach"):
                self._strategy_changes_after_struggle += 1

        self._last_action_format = action.get("format")
        self._last_action_difficulty = action.get("difficulty")
        self._consecutive_failures_at_last_action = current_failures

        self._actions.append(action)
        reward = 0.0

        if action_type == "teach":
            reward = self._grade_teach(action, result, student_knowledge, available_topics)
        elif action_type == "quiz":
            reward = self._grade_quiz(action, result, topic)
        elif action_type == "review":
            reward = self._grade_review(action, result, student_knowledge)
        elif action_type == "end_session":
            reward = self._grade_end_session(result, student_observable)

        self._step_rewards.append(reward)
        return reward

    def _grade_teach(
        self,
        action: Dict[str, Any],
        result: Dict[str, Any],
        student_knowledge: Dict[str, float],
        available_topics: List[str],
    ) -> float:
        topic = action.get("topic", "")
        knowledge_gain = result.get("knowledge_gain", 0.0)
        knowledge_before = result.get("knowledge_before", 0.0)

        # Prerequisite violation
        if topic not in available_topics:
            return PENALTY_PREREQ_VIOLATION

        # Wasted time: teaching mastered content
        if knowledge_before > 0.9:
            return PENALTY_WASTED_TIME

        # Reward proportional to knowledge gain
        reward = knowledge_gain * REWARD_KNOWLEDGE_SCALE
        if reward > 0:
            self._useful_actions += 1

        self._taught_since_last_quiz = True
        return reward

    def _grade_quiz(
        self,
        action: Dict[str, Any],
        result: Dict[str, Any],
        topic: str,
    ) -> float:
        score = result.get("score", 0.0)

        # First quiz on a topic is diagnostic (valuable)
        if topic not in self._quizzed_topics:
            self._quizzed_topics.add(topic)
            self._useful_actions += 1
            self._last_quizzed_topic = topic
            self._taught_since_last_quiz = False

            # Frustration penalty for very low scores
            if score < 0.25:
                return REWARD_FIRST_QUIZ + PENALTY_FRUSTRATION
            return REWARD_FIRST_QUIZ

        # Redundant quiz: same topic without teaching in between
        if topic == self._last_quizzed_topic and not self._taught_since_last_quiz:
            return PENALTY_REDUNDANT_QUIZ

        self._last_quizzed_topic = topic
        self._taught_since_last_quiz = False

        if score < 0.25:
            return REWARD_REPEAT_QUIZ + PENALTY_FRUSTRATION
        return REWARD_REPEAT_QUIZ

    def _grade_review(
        self,
        action: Dict[str, Any],
        result: Dict[str, Any],
        student_knowledge: Dict[str, float],
    ) -> float:
        topic = action.get("topic", "")
        knowledge_gain = result.get("knowledge_gain", 0.0)
        knowledge_before = result.get("knowledge_before", 0.0)

        # Reviewing topic that hasn't decayed (still high)
        if knowledge_before > 0.8:
            return PENALTY_WASTED_TIME * 0.6  # lighter penalty than teach

        # Reward proportional to knowledge recovered
        reward = knowledge_gain * REWARD_REVIEW_SCALE
        if reward > 0:
            self._useful_actions += 1
        return reward

    def _grade_end_session(
        self,
        result: Dict[str, Any],
        student_observable: Dict[str, Any],
    ) -> float:
        fatigue_at_end = result.get("fatigue_at_end", 0.0)
        motivation_at_end = result.get("motivation_at_end", 0.5)

        self._sessions_completed += 1
        self._motivation_at_session_ends.append(motivation_at_end)

        # Good timing: student was fatigued or motivation low
        if fatigue_at_end > 0.7 or motivation_at_end < 0.3:
            return REWARD_GOOD_SESSION_END

        # Bad timing: student still had capacity
        if motivation_at_end > 0.6 and fatigue_at_end < 0.4:
            return PENALTY_EARLY_SESSION_END

        return 0.0  # neutral

    # -------------------------------------------------------------------
    # Final scoring
    # -------------------------------------------------------------------

    def compute_final_score(
        self,
        final_knowledge: Dict[str, float],
        total_sessions: int,
    ) -> Dict[str, Any]:
        """Compute the final normalized score [0.0, 1.0]."""
        self._total_sessions = total_sessions

        knowledge_score = self._score_knowledge_acquisition(final_knowledge)
        retention_score = self._score_retention(final_knowledge)
        engagement_score = self._score_engagement()
        efficiency_score = self._score_efficiency()
        adaptivity_score = self._score_adaptivity()

        w = GRADING_WEIGHTS
        raw_score = (
            knowledge_score * w["knowledge_acquisition"]
            + retention_score * w["retention"]
            + engagement_score * w["engagement"]
            + efficiency_score * w["efficiency"]
            + adaptivity_score * w["adaptivity"]
        )

        final_score = max(0.0, min(1.0, raw_score))

        return {
            "score": round(final_score, 4),
            "breakdown": {
                "knowledge_acquisition": round(knowledge_score, 4),
                "retention": round(retention_score, 4),
                "engagement": round(engagement_score, 4),
                "efficiency": round(efficiency_score, 4),
                "adaptivity": round(adaptivity_score, 4),
            },
            "total_steps": self._total_actions,
            "total_raw_reward": round(sum(self._step_rewards), 4),
        }

    def _score_knowledge_acquisition(self, final_knowledge: Dict[str, float]) -> float:
        """Average knowledge gain across curriculum topics."""
        if not self.curriculum_topics:
            return 0.0
        total_gain = 0.0
        for topic in self.curriculum_topics:
            initial = self._initial_knowledge.get(topic, 0.0)
            final = final_knowledge.get(topic, 0.0)
            total_gain += max(0, final - initial)
        max_possible = sum(1.0 - self._initial_knowledge.get(t, 0.0) for t in self.curriculum_topics)
        if max_possible <= 0:
            return 1.0
        return total_gain / max_possible

    def _score_retention(self, final_knowledge: Dict[str, float]) -> float:
        """Average knowledge retained at the end (post-forgetting for Task 3)."""
        if not self.curriculum_topics:
            return 0.0
        avg = sum(final_knowledge.get(t, 0.0) for t in self.curriculum_topics) / len(self.curriculum_topics)
        return min(1.0, avg)

    def _score_engagement(self) -> float:
        """Session completion + motivation preserved."""
        session_completion = 1.0  # sessions completed by reaching end or end_session
        avg_motivation = 0.5
        if self._motivation_at_session_ends:
            avg_motivation = sum(self._motivation_at_session_ends) / len(self._motivation_at_session_ends)
        return 0.5 * session_completion + 0.5 * avg_motivation

    def _score_efficiency(self) -> float:
        """Proportion of useful actions."""
        if self._total_actions == 0:
            return 0.0
        return self._useful_actions / self._total_actions

    def _score_adaptivity(self) -> float:
        """Did agent change strategy when student struggled?"""
        if self._struggle_opportunities == 0:
            return 1.0  # no struggles = no opportunity needed
        return self._strategy_changes_after_struggle / self._struggle_opportunities
