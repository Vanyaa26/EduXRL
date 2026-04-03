"""
Learning Environment — Core Logic.

Implements the OpenEnv Environment interface for the adaptive learning task.
Manages student state, processes teaching actions, computes rewards via the
grader, and returns typed observations.

This is the "game engine" — it decides what happens when the agent teaches.
"""

import random
from typing import Any, Dict, List, Optional
from uuid import uuid4

try:
    from openenv.core import Environment
except ImportError:
    try:
        from openenv_core.env_server.interfaces import Environment
    except ImportError:
        from abc import ABC, abstractmethod

        class Environment(ABC):
            def __init__(self, transform=None):
                self.transform = transform

            @abstractmethod
            def reset(self): ...

            @abstractmethod
            def step(self, action): ...

            @property
            @abstractmethod
            def state(self): ...

try:
    from ..graders.learning_grader import LearningGrader
    from ..models import TeachingAction, StudentObservation, SessionState, ActionType
    from ..tasks.task_definitions import TaskDefinition, get_task
except ImportError:
    from graders.learning_grader import LearningGrader
    from models import TeachingAction, StudentObservation, SessionState, ActionType
    from tasks.task_definitions import TaskDefinition, get_task

from .student_model import SimulatedStudent
from .curriculum import Curriculum


class LearningEnvironment(Environment):
    """
    Adaptive Learning Path RL Environment.

    The agent controls the teaching path for a simulated student.
    The student's cognition follows published cognitive science models.

    Episode flow:
        reset(task_id) → observe student state + curriculum
        step(action) → reward + updated observation
        ... repeat until session ends or max steps reached
        (Task 3: multiple sessions with forgetting between them)
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._task: Optional[TaskDefinition] = None
        self._curriculum: Optional[Curriculum] = None
        self._student: Optional[SimulatedStudent] = None
        self._grader: Optional[LearningGrader] = None
        self._state = SessionState()

        # Session tracking
        self._current_session_idx: int = 0
        self._session_step: int = 0
        self._session_max_steps: int = 20
        self._episode_done: bool = False

        # For multi-session tasks
        self._all_session_rewards: List[float] = []

    def reset(self, seed=None, episode_id=None, **kwargs: Any) -> StudentObservation:
        """Reset the environment for a new episode."""
        task_id = kwargs.get("task_id", "task1")
        self._task = get_task(task_id)

        variation_seed = seed if seed is not None else random.randint(0, 2**31)

        # Build curriculum subset for this task
        self._curriculum = Curriculum().get_subset(self._task.curriculum_topics)

        # Create simulated student
        profile = self._task.student_profile
        self._student = SimulatedStudent(
            topic_names=self._task.curriculum_topics,
            initial_knowledge=profile.get("initial_knowledge", {}),
            initial_motivation=profile.get("initial_motivation", 0.8),
            fatigue_rate=profile.get("fatigue_rate", 1.0),
            preferred_format=profile.get("preferred_format", "exercise"),
            learning_rate=profile.get("learning_rate", 1.0),
            seed=variation_seed,
        )

        # Create grader
        self._grader = LearningGrader(self._task, self._task.curriculum_topics)
        self._grader.set_initial_knowledge(dict(self._student.knowledge))

        # Session setup
        self._current_session_idx = 0
        session_config = self._task.sessions[0]
        self._session_step = 0
        self._session_max_steps = session_config["max_steps"]
        self._episode_done = False
        self._all_session_rewards = []

        self._state = SessionState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=self._task.id,
            session_number=1,
            topics_covered=[],
            total_knowledge_gain=0.0,
            total_reward=0.0,
            actions_taken=0,
            max_steps=self._task.total_max_steps,
        )

        return self._build_observation(reward=0.0, done=False, action_result={})

    def step(self, action, timeout_s=None, **kwargs: Any) -> StudentObservation:
        """Execute a teaching action and return the resulting observation."""
        if self._task is None or self._student is None or self._grader is None:
            return StudentObservation(
                task_id="",
                task_description="Environment not initialized. Call reset() first.",
                done=True,
                reward=0.0,
                metadata={"error": "Call reset() before step()"},
            )

        self._state.step_count += 1
        self._session_step += 1
        self._state.actions_taken += 1

        # Convert action to dict
        if isinstance(action, dict):
            action_dict = action
        else:
            try:
                action_dict = action.model_dump(exclude_none=True)
            except AttributeError:
                from dataclasses import asdict
                action_dict = {k: v for k, v in asdict(action).items() if v is not None}

        action_type = action_dict.get("action_type", "")
        topic = action_dict.get("topic", "")
        fmt = action_dict.get("format", "explanation")
        difficulty = action_dict.get("difficulty", "medium")

        action_result = {}

        # Process action
        if action_type == ActionType.END_SESSION or action_type == "end_session":
            action_result = self._student.end_session()
            reward = self._grader.grade_action(
                action_dict, action_result,
                self._student.knowledge,
                self._student.get_observable_state(),
                self._curriculum.get_unlocked_topics(self._student.knowledge),
            )
            self._state.total_reward += reward
            return self._handle_session_end(reward, action_result)

        elif action_type == ActionType.TEACH or action_type == "teach":
            base_diff = self._curriculum.get_base_difficulty(topic)
            action_result = self._student.receive_teaching(topic, fmt, difficulty, base_diff)

        elif action_type == ActionType.QUIZ or action_type == "quiz":
            action_result = self._student.take_quiz(topic, difficulty)

        elif action_type == ActionType.REVIEW or action_type == "review":
            base_diff = self._curriculum.get_base_difficulty(topic)
            action_result = self._student.receive_review(topic, fmt, difficulty, base_diff)

        else:
            action_result = {"error": f"Unknown action_type: {action_type}"}
            return self._build_observation(reward=-0.05, done=False, action_result=action_result)

        # Track topics covered
        if topic and topic not in self._state.topics_covered:
            self._state.topics_covered.append(topic)

        # Compute reward
        available = self._curriculum.get_unlocked_topics(self._student.knowledge)
        reward = self._grader.grade_action(
            action_dict, action_result,
            self._student.knowledge,
            self._student.get_observable_state(),
            available,
        )
        self._state.total_reward += reward
        self._all_session_rewards.append(reward)

        # Check if session should end
        done = False
        if self._session_step >= self._session_max_steps:
            return self._handle_session_end(reward, action_result)
        if self._student.is_disengaged:
            action_result["student_quit"] = True
            return self._handle_session_end(reward, action_result)

        return self._build_observation(reward=reward, done=False, action_result=action_result)

    @property
    def state(self) -> SessionState:
        return self._state

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _handle_session_end(self, last_reward: float, action_result: Dict) -> StudentObservation:
        """Handle end of a session. May start next session or end episode."""
        self._current_session_idx += 1

        # Check if there are more sessions
        if self._current_session_idx < len(self._task.sessions):
            # Apply forgetting between sessions
            next_session = self._task.sessions[self._current_session_idx]
            days = next_session.get("days_since_last", 0)
            if days > 0:
                decay_report = self._student.apply_forgetting(days)
                action_result["forgetting_report"] = decay_report

            # Start new session
            self._student.start_new_session(motivation_boost=0.05)
            # Increase fatigue rate for later sessions (simulates real life)
            self._student.fatigue_rate = self._task.student_profile.get("fatigue_rate", 1.0) * (1.0 + 0.1 * self._current_session_idx)

            self._session_step = 0
            self._session_max_steps = next_session["max_steps"]
            self._state.session_number = next_session["session_number"]

            action_result["session_transition"] = {
                "new_session": next_session["session_number"],
                "days_elapsed": days,
            }

            return self._build_observation(
                reward=last_reward, done=False, action_result=action_result
            )

        # Episode complete
        self._episode_done = True
        return self._build_final_observation(last_reward, action_result)

    def _build_observation(
        self,
        reward: float,
        done: bool,
        action_result: Dict[str, Any],
    ) -> StudentObservation:
        """Build observation from current state."""
        observable = self._student.get_observable_state() if self._student else {}
        available = (
            self._curriculum.get_unlocked_topics(self._student.knowledge)
            if self._curriculum and self._student else []
        )

        # Build topic scores from quiz results
        topic_scores = {}
        for t in (self._task.curriculum_topics if self._task else []):
            topic_scores[t] = round(self._student.knowledge.get(t, 0.0), 4) if self._student else 0.0

        session_config = (
            self._task.sessions[self._current_session_idx]
            if self._task and self._current_session_idx < len(self._task.sessions)
            else {}
        )

        return StudentObservation(
            task_id=self._task.id if self._task else "",
            task_description=self._task.description if self._task else "",
            available_topics=available,
            prerequisite_map=self._curriculum.get_prerequisite_map() if self._curriculum else {},
            all_topics=self._task.curriculum_topics if self._task else [],
            topic_scores=topic_scores,
            topics_taught=observable.get("topics_taught", []),
            topics_quizzed=observable.get("topics_quizzed", []),
            last_action_result=action_result,
            last_quiz_score=observable.get("last_quiz_score"),
            consecutive_failures=observable.get("consecutive_failures", 0),
            consecutive_successes=observable.get("consecutive_successes", 0),
            actions_since_last_success=observable.get("actions_since_last_success", 0),
            step_number=self._session_step,
            steps_remaining=max(0, self._session_max_steps - self._session_step),
            session_number=self._state.session_number,
            days_since_last_session=session_config.get("days_since_last", 0),
            done=done,
            reward=reward,
            metadata={
                "step_count": self._state.step_count,
                "total_reward": round(self._state.total_reward, 4),
            },
        )

    def _build_final_observation(self, last_reward: float, action_result: Dict) -> StudentObservation:
        """Build the final observation when episode ends."""
        final_score = self._grader.compute_final_score(
            self._student.knowledge,
            self._task.num_sessions,
        )

        obs = self._build_observation(reward=last_reward, done=True, action_result=action_result)
        obs.metadata["final_score"] = final_score
        obs.metadata["episode_complete"] = True
        return obs

    def get_grader_results(self) -> Dict[str, Any]:
        """Get grader results. Call after episode is done."""
        if self._grader and self._student:
            return self._grader.compute_final_score(
                self._student.knowledge,
                self._task.num_sessions if self._task else 1,
            )
        return {"score": 0.0, "breakdown": {}, "error": "No grader initialized"}
