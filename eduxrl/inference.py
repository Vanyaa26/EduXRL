#!/usr/bin/env python3
"""
Inference Script for EduXRL
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your
  environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- The inference script must be named `inference.py` and placed in the root
  directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script emits [START], [STEP], and [END] lines per the hackathon spec.
"""

import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")

MAX_STEPS = 30
TEMPERATURE = 0.3
MAX_TOKENS = 300
BENCHMARK = "eduxrl"


# ---------------------------------------------------------------------------
# Stdout logging (mandatory format)
# ---------------------------------------------------------------------------


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are a teaching strategy agent controlling an adaptive learning environment.
    You decide what a student learns next based on their current knowledge,
    recent quiz scores, and behavioral signals.

    Available actions (respond with JSON only):
    - teach: {"action_type": "teach", "topic": "<topic>", "format": "<format>", "difficulty": "<difficulty>"}
    - quiz: {"action_type": "quiz", "topic": "<topic>", "difficulty": "<difficulty>"}
    - review: {"action_type": "review", "topic": "<topic>", "format": "<format>", "difficulty": "<difficulty>"}
    - end_session: {"action_type": "end_session", "topic": "", "format": "", "difficulty": ""}

    Formats: explanation, worked_example, exercise
    Difficulties: easy, medium, hard

    Strategy guidelines:
    - Quiz first to discover what the student knows
    - Teach topics in prerequisite order (check available_topics)
    - Match difficulty to current knowledge (easy if <0.3, medium if 0.3-0.6, hard if >0.6)
    - If student has consecutive failures, switch to easier content or different format
    - End session if student seems fatigued (many failures, low engagement)
    - Review previously learned topics to prevent forgetting

    Respond with ONLY the JSON action object, no other text.
""").strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def format_observation(obs: Dict[str, Any]) -> str:
    """Format observation as readable text for the LLM."""
    inner = obs.get("observation", obs)
    parts = []

    desc = inner.get("task_description", "")
    if desc:
        parts.append(f"TASK: {desc}")

    parts.append(f"\nSESSION: {inner.get('session_number', 1)} | Step: {inner.get('step_number', 0)} | Remaining: {inner.get('steps_remaining', 0)}")

    if inner.get("days_since_last_session", 0) > 0:
        parts.append(f"Days since last session: {inner['days_since_last_session']} (student may have forgotten material)")

    topics = inner.get("topic_scores", {})
    if topics:
        parts.append("\nTOPIC KNOWLEDGE:")
        for topic, score in topics.items():
            bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            taught = "✓" if topic in inner.get("topics_taught", []) else " "
            quizzed = "Q" if topic in inner.get("topics_quizzed", []) else " "
            parts.append(f"  {topic:20s} [{bar}] {score:.2f} {taught}{quizzed}")

    available = inner.get("available_topics", [])
    parts.append(f"\nAVAILABLE TOPICS (prerequisites met): {available}")

    prereqs = inner.get("prerequisite_map", {})
    if prereqs:
        parts.append(f"PREREQUISITES: {json.dumps(prereqs)}")

    # Behavioral signals
    parts.append(f"\nBEHAVIORAL SIGNALS:")
    parts.append(f"  Last quiz score: {inner.get('last_quiz_score', 'N/A')}")
    parts.append(f"  Consecutive failures: {inner.get('consecutive_failures', 0)}")
    parts.append(f"  Consecutive successes: {inner.get('consecutive_successes', 0)}")

    last_result = inner.get("last_action_result", {})
    if last_result:
        parts.append(f"\nLAST ACTION RESULT: {json.dumps(last_result, default=str)}")

    parts.append("\nRespond with your next action as a JSON object.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------


def run_task(
    client: OpenAI,
    base_url: str,
    task_id: str,
    model: str,
) -> Dict[str, Any]:
    """Run the LLM agent against a single EduXRL task."""
    reset_resp = requests.post(f"{base_url}/reset", json={"task_id": task_id}, timeout=30)
    reset_resp.raise_for_status()
    obs = reset_resp.json()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    rewards: List[float] = []
    steps = 0
    done = False

    log_start(task=task_id, env=BENCHMARK, model=model)

    try:
        while not done and steps < MAX_STEPS:
            obs_text = format_observation(obs)
            messages.append({"role": "user", "content": obs_text})

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    response_format={"type": "json_object"},
                )
                action_text = response.choices[0].message.content or ""
                messages.append({"role": "assistant", "content": action_text})
            except Exception as e:
                print(f"[DEBUG] LLM API error: {e}", flush=True)
                action_text = '{"action_type": "end_session", "topic": "", "format": "", "difficulty": ""}'

            try:
                action = json.loads(action_text)
            except json.JSONDecodeError:
                messages.append({
                    "role": "user",
                    "content": "Invalid JSON. Respond with only a JSON action object.",
                })
                continue

            try:
                step_resp = requests.post(
                    f"{base_url}/step", json={"action": action}, timeout=30,
                )
                step_resp.raise_for_status()
                obs = step_resp.json()
            except Exception as e:
                print(f"[DEBUG] Step error: {e}", flush=True)
                break

            reward = obs.get("reward", 0.0) or 0.0
            done = obs.get("done", False)
            rewards.append(reward)
            steps += 1

            action_str = f"{action.get('action_type', '?')}({action.get('topic', '')})"
            log_step(step=steps, action=action_str, reward=reward, done=done, error=None)

        # Get final score
        final_meta = obs.get("observation", obs).get("metadata", {})
        final_score = final_meta.get("final_score", {})
        score = final_score.get("score", 0.0)

        if score == 0.0:
            try:
                grader_resp = requests.post(f"{base_url}/grader", json={"task_id": task_id}, timeout=30)
                if grader_resp.ok:
                    score = grader_resp.json().get("score", score)
            except Exception:
                pass

        success = score >= 0.3
        log_end(success=success, steps=steps, score=score, rewards=rewards)

        return {
            "score": round(score, 4),
            "steps": steps,
            "total_reward": round(sum(rewards), 4),
            "status": "completed" if done else "max_steps_reached",
        }

    except Exception as e:
        log_end(success=False, steps=steps, score=0.0, rewards=rewards)
        return {"score": 0.0, "steps": steps, "status": "error", "error": str(e)}


def main() -> None:
    if not API_KEY:
        print("Error: HF_TOKEN or API_KEY environment variable not set.", flush=True)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    model = MODEL_NAME

    task_ids = ["task1", "task2", "task3"]

    print(f"EduXRL Inference | Model: {model} | Env: {ENV_BASE_URL}", flush=True)

    results = {}
    for task_id in task_ids:
        result = run_task(client=client, base_url=ENV_BASE_URL, task_id=task_id, model=model)
        results[task_id] = result

    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
