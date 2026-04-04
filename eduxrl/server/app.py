"""
FastAPI application for the EduXRL Environment.

Uses create_app() from OpenEnv for standard endpoints.
Adds /tasks, /grader, /baseline for hackathon requirements.
"""

from typing import Any, Dict, Optional

from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

try:
    from ..models import EduxrlAction, EduxrlObservation
    from .eduxrl_environment import EduxrlEnvironment
    from .task_definitions import get_all_tasks, list_task_ids
except (ImportError, ModuleNotFoundError):
    from models import EduxrlAction, EduxrlObservation
    from server.eduxrl_environment import EduxrlEnvironment
    from server.task_definitions import get_all_tasks, list_task_ids


app = create_app(
    EduxrlEnvironment,
    EduxrlAction,
    EduxrlObservation,
    env_name="eduxrl",
)

# Override root with landing page
from .dashboard import get_dashboard_html

# Remove the default root route if it exists
app.routes[:] = [r for r in app.routes if not (hasattr(r, "path") and hasattr(r, "methods") and r.path == "/" and "GET" in r.methods)]

@app.get("/", response_class=HTMLResponse)
async def landing_page():
    return get_dashboard_html()


# ---------------------------------------------------------------------------
# Custom endpoints for hackathon
# ---------------------------------------------------------------------------


class GraderRequest(BaseModel):
    task_id: str = "task1"


@app.get("/tasks")
def get_tasks_endpoint():
    tasks = get_all_tasks()
    return {
        "tasks": [task.to_summary() for task in tasks.values()],
        "action_schema": {
            "action_types": ["teach", "quiz", "review", "end_session"],
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
def baseline_endpoint():
    results = {}
    for task_id in list_task_ids():
        try:
            results[task_id] = _run_heuristic_baseline(task_id)
        except Exception as e:
            results[task_id] = {"score": 0.0, "steps": 0, "status": "error", "error": str(e)}
    return results


def _run_heuristic_baseline(task_id: str) -> Dict[str, Any]:
    from .curriculum import Curriculum

    env = EduxrlEnvironment()
    obs = env.reset(seed=0, task_id=task_id)
    total_reward = 0.0
    steps = 0
    taught_topics = set()

    while not obs.done and steps < 200:
        available = obs.available_topics
        topic_scores = obs.topic_scores
        action_dict = None

        for topic in available:
            if topic not in obs.topics_quizzed:
                action_dict = {"action_type": "quiz", "topic": topic, "difficulty": "medium", "format": "explanation"}
                break

        if action_dict is None:
            best_topic = None
            lowest_score = 1.0
            for topic in available:
                score = topic_scores.get(topic, 0.0)
                if score < lowest_score and score < 0.8:
                    lowest_score = score
                    best_topic = topic

            if best_topic is not None:
                score = topic_scores.get(best_topic, 0.0)
                diff = "easy" if score < 0.3 else ("medium" if score < 0.6 else "hard")
                if best_topic in taught_topics:
                    action_dict = {"action_type": "review", "topic": best_topic, "format": "exercise", "difficulty": diff}
                else:
                    action_dict = {"action_type": "teach", "topic": best_topic, "format": "exercise", "difficulty": diff}
                    taught_topics.add(best_topic)

        if action_dict is None:
            action_dict = {"action_type": "end_session", "topic": "", "format": "explanation", "difficulty": "easy"}

        if obs.consecutive_failures >= 3 or obs.steps_remaining <= 1:
            action_dict = {"action_type": "end_session", "topic": "", "format": "explanation", "difficulty": "easy"}

        action = EduxrlAction(**action_dict)
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


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
