"""
FastAPI application for EduXRL.

Creates the HTTP server with OpenEnv standard endpoints (reset/step/state/health)
plus custom endpoints (/tasks, /grader, /baseline).
"""

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    from openenv.core import create_app
except ImportError:
    try:
        from openenv_core.env_server.http_server import create_app
    except ImportError:
        create_app = None

try:
    from openenv.core import serialize_observation, deserialize_action
except ImportError:
    serialize_observation = None
    deserialize_action = None

try:
    from .learning_environment import LearningEnvironment
    from ..models import TeachingAction, StudentObservation, ActionType
    from ..tasks.task_definitions import get_all_tasks, list_task_ids
except ImportError:
    from server.learning_environment import LearningEnvironment
    from models import TeachingAction, StudentObservation, ActionType
    from tasks.task_definitions import get_all_tasks, list_task_ids


# ---------------------------------------------------------------------------
# Create the main OpenEnv app
# ---------------------------------------------------------------------------

if create_app is not None:
    app = create_app(
        LearningEnvironment,
        TeachingAction,
        StudentObservation,
        env_name="eduxrl",
    )
    _override_paths = {("/reset", "POST"), ("/step", "POST"), ("/state", "GET"), ("/", "GET")}
    app.routes[:] = [
        r for r in app.routes
        if not (
            hasattr(r, "path")
            and hasattr(r, "methods")
            and (r.path, next(iter(r.methods))) in _override_paths
        )
    ]
else:
    app = FastAPI(title="EduXRL - Adaptive Learning Path Environment")

    @app.get("/health")
    async def health():
        return {"status": "healthy"}


# ---------------------------------------------------------------------------
# Stateful REST endpoints
# ---------------------------------------------------------------------------

_shared_env = LearningEnvironment()


def _obs_to_response(obs):
    if serialize_observation is not None:
        return serialize_observation(obs)
    if hasattr(obs, "model_dump"):
        obs_dict = obs.model_dump(exclude={"reward", "done"})
    else:
        from dataclasses import asdict
        obs_dict = asdict(obs)
        obs_dict.pop("reward", None)
        obs_dict.pop("done", None)
    return {"observation": obs_dict, "reward": obs.reward, "done": obs.done}


@app.get("/")
async def root():
    return {
        "name": "EduXRL",
        "description": "Adaptive Learning Path RL Environment — AI agents learn to teach simulated students grounded in cognitive science.",
        "endpoints": {
            "/health": "GET — health check",
            "/tasks": "GET — list tasks and action schema",
            "/reset": "POST — start a new episode",
            "/step": "POST — execute a teaching action",
            "/state": "GET — current session state",
            "/grader": "POST — get grader score after episode",
            "/baseline": "POST — run heuristic baseline on all tasks",
        },
        "tasks": ["task1 (Easy)", "task2 (Medium)", "task3 (Hard)"],
    }


@app.post("/reset")
async def stateful_reset(request: Dict[str, Any] = {}):
    obs = _shared_env.reset(**request)
    return _obs_to_response(obs)


@app.post("/step")
async def stateful_step(request: Dict[str, Any] = {}):
    action_data = request.get("action", request)
    if deserialize_action is not None:
        action = deserialize_action(action_data, TeachingAction)
    else:
        action = TeachingAction(**action_data)
    obs = _shared_env.step(action)
    return _obs_to_response(obs)


@app.get("/state")
async def stateful_state():
    s = _shared_env.state
    if hasattr(s, "model_dump"):
        return s.model_dump()
    from dataclasses import asdict
    return asdict(s)


# ---------------------------------------------------------------------------
# Custom endpoints
# ---------------------------------------------------------------------------


class GraderRequest(BaseModel):
    task_id: str = "task1"


class BaselineRequest(BaseModel):
    model: str = "heuristic"


@app.get("/tasks")
def get_tasks_endpoint():
    tasks = get_all_tasks()
    return {
        "tasks": [task.to_summary() for task in tasks.values()],
        "action_schema": {
            "action_types": [at.value for at in ActionType],
            "formats": ["explanation", "worked_example", "exercise"],
            "difficulties": ["easy", "medium", "hard"],
            "example": {
                "action_type": "teach",
                "topic": "variables",
                "format": "exercise",
                "difficulty": "easy",
            },
        },
    }


@app.post("/grader")
def grader_endpoint(request: GraderRequest):
    try:
        score_data = _run_heuristic_baseline(request.task_id)
        score_data["task_id"] = request.task_id
        return score_data
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {request.task_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/baseline")
def baseline_endpoint(request: Optional[BaselineRequest] = None):
    results = {}
    for task_id in list_task_ids():
        try:
            results[task_id] = _run_heuristic_baseline(task_id)
        except Exception as e:
            results[task_id] = {"score": 0.0, "steps": 0, "status": "error", "error": str(e)}
    return results


# ---------------------------------------------------------------------------
# Heuristic baseline
# ---------------------------------------------------------------------------


def _run_heuristic_baseline(task_id: str) -> Dict[str, Any]:
    """
    Run a deterministic heuristic baseline for a task.

    Strategy: quiz each topic first, teach topics in prerequisite order
    at appropriate difficulty, review when needed.
    """
    from .curriculum import Curriculum

    env = LearningEnvironment()
    obs = env.reset(seed=0, task_id=task_id)
    total_reward = 0.0
    steps = 0

    curriculum = Curriculum().get_subset(obs.all_topics)
    taught_topics = set()

    while not obs.done and steps < 200:
        available = obs.available_topics
        topic_scores = obs.topic_scores

        # Find the best topic to work on
        action_dict = None

        # Priority 1: quiz untested available topics to discover knowledge
        for topic in available:
            if topic not in obs.topics_quizzed:
                action_dict = {
                    "action_type": "quiz",
                    "topic": topic,
                    "difficulty": "medium",
                }
                break

        # Priority 2: teach topics with low knowledge that haven't been taught much
        if action_dict is None:
            best_topic = None
            lowest_score = 1.0
            for topic in available:
                score = topic_scores.get(topic, 0.0)
                if score < lowest_score and score < 0.8:
                    lowest_score = score
                    best_topic = topic

            if best_topic is not None:
                # Match difficulty to current knowledge (ZPD)
                score = topic_scores.get(best_topic, 0.0)
                if score < 0.3:
                    diff = "easy"
                elif score < 0.6:
                    diff = "medium"
                else:
                    diff = "hard"

                if best_topic in taught_topics:
                    action_dict = {
                        "action_type": "review",
                        "topic": best_topic,
                        "format": "exercise",
                        "difficulty": diff,
                    }
                else:
                    action_dict = {
                        "action_type": "teach",
                        "topic": best_topic,
                        "format": "exercise",
                        "difficulty": diff,
                    }
                    taught_topics.add(best_topic)

        # Priority 3: end session if nothing useful to do
        if action_dict is None:
            action_dict = {"action_type": "end_session", "topic": "", "format": "explanation", "difficulty": "easy"}

        # Check behavioral signals for fatigue/disengagement
        if obs.consecutive_failures >= 3 or obs.steps_remaining <= 1:
            action_dict = {"action_type": "end_session", "topic": "", "format": "explanation", "difficulty": "easy"}

        action = TeachingAction(**action_dict)
        obs = env.step(action)
        total_reward += obs.reward or 0.0
        steps += 1

    final_score = env.get_grader_results()
    return {
        "score": final_score.get("score", 0.0),
        "steps": steps,
        "status": "completed",
        "breakdown": final_score.get("breakdown", {}),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
