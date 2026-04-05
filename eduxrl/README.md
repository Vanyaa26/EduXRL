---
title: EduXRL - Adaptive Learning Path RL Environment
emoji: "🎓"
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---

# EduXRL — Adaptive Learning Path RL Environment

**Most online learners never finish what they start.**

Not because the content is bad. Because the *path* is wrong. A fixed sequence of lessons treats every student the same — the beginner who needs more time on basics, the experienced developer who just wants to skip ahead, the student who learns best by doing rather than reading. The result: frustration, boredom, and dropout.

Adaptive learning platforms attempt to fix this with rule-based logic — "if the student failed, repeat the lesson." But that's reactive, not strategic. It doesn't plan ahead, doesn't manage motivation, doesn't account for forgetting, and doesn't discover how a student actually learns best.

**EduXRL** is an OpenEnv-compliant RL environment where AI agents learn to *teach* — not by following rules, but by discovering optimal teaching strategies through interaction with simulated students whose cognition follows real, published scientific models.

## How It Works

The environment simulates a student learning Python programming. The student isn't a random number generator — their learning, forgetting, and motivation follow established cognitive science:

| Cognitive Model | What It Governs | Research Basis |
|----------------|----------------|----------------|
| **Power Law of Learning** | How knowledge grows with practice (diminishing returns) | Newell & Rosenbloom (1981) |
| **Ebbinghaus Forgetting Curve** | How knowledge decays over time without review | Ebbinghaus (1885), widely replicated |
| **Spacing Effect** | Why spaced review beats cramming | Cepeda et al. (2006), meta-analysis of 254 studies |
| **Zone of Proximal Development** | Why difficulty must match current ability | Vygotsky (1978) |

The AI agent controls the teaching path — deciding what topic to present, in what format, at what difficulty, when to review old material, and when to end the session before the student burns out.

The reward function measures **observable learning outcomes**: knowledge gained, material retained, session completed, time not wasted. No human labels. No opinion about what the "right" teaching decision is.

## Why This Is a Genuine RL Problem

A fixed teaching strategy fails here. "Always teach the next topic in sequence" doesn't work because:

- **Students have hidden gaps** — they might know loops but not the prerequisite (conditionals) properly. Pushing forward leads to cascading failure.
- **Motivation is fragile** — two consecutive failures can cause the student to disengage. The agent must notice and adapt.
- **Forgetting is real** — what was learned on Day 1 fades by Day 3. The agent must decide: teach new material or review old?
- **Learning styles differ** — some students learn faster from exercises, others from explanations. The agent doesn't know this upfront — it must explore to discover.
- **Attention is finite** — each session has limited steps. The agent can't do everything. It must prioritize.

Every action changes the student's state. Every decision constrains what's possible next. There is no lookup table. The agent must adapt.

## Tasks

| Task | Difficulty | Scenario | Sessions | Topics |
|------|-----------|----------|----------|--------|
| `task1` | Easy | Fresh student, stable motivation, no surprises | 1 session (20 steps) | 5 |
| `task2` | Medium | Student with knowledge gaps, sensitive motivation, hidden format preference | 1 session (25 steps) | 7 |
| `task3` | Hard | Multi-session with forgetting between days, increasing fatigue | 3 sessions (Day 1, 3, 7) | 10 |

### Task 1: The Steady Learner
Teach a fresh student 5 Python topics in one session. Tests whether the agent can sequence lessons properly, match difficulty to current knowledge, and avoid wasting time on already-mastered content.

### Task 2: The Struggling Student
A student who already knows some topics but has hidden gaps. Motivation drops quickly on failure. Strongly prefers one content format over others — the agent must discover this through trial and observation. Tests adaptivity, gap detection, and motivation management.

### Task 3: The Forgetting Student
Three sessions spread across a simulated week. Knowledge decays between sessions following the Ebbinghaus curve. Agent must balance teaching new content vs reviewing forgotten material. Must plan ahead — prerequisites taught in Session 1 are needed for advanced topics in Session 3.

## Action Space

The agent sends one action per step:

```json
{
  "action_type": "teach",
  "topic": "loops",
  "format": "exercise",
  "difficulty": "medium"
}
```

| Action Type | Description |
|-------------|------------|
| `teach` | Present new content on a topic (specify format + difficulty) |
| `quiz` | Test the student on a topic — reveals their current knowledge level |
| `review` | Re-teach a previously covered topic to combat forgetting |
| `end_session` | End the session (good when student is fatigued, bad when they still have capacity) |

Content formats: `explanation`, `worked_example`, `exercise`
Difficulty levels: `easy`, `medium`, `hard`

## Observation Space

After each action, the agent receives:

| Field | Type | Description |
|-------|------|-------------|
| `topic_scores` | Dict[str, float] | Last known quiz score per topic (0.0 if untested) |
| `topics_taught` | List[str] | Topics presented at least once |
| `topics_quizzed` | List[str] | Topics tested at least once |
| `available_topics` | List[str] | Topics whose prerequisites are satisfied |
| `prerequisite_map` | Dict | Prerequisite dependency graph |
| `last_quiz_score` | float | Most recent quiz result |
| `consecutive_failures` | int | Sequential quiz scores below 0.4 |
| `consecutive_successes` | int | Sequential quiz scores above 0.7 |
| `step_number` | int | Current step in the session |
| `steps_remaining` | int | Steps left before session ends |
| `session_number` | int | Current session (1, 2, or 3) |
| `days_since_last_session` | int | Simulated days since previous session |

The agent does **not** see the student's internal motivation or fatigue. It must infer these from behavioral signals — just as a real teaching system would.

## Reward Design

Rewards are **dense and per-step** — the agent gets feedback after every action, not just at the end.

**Positive signals:**
- Knowledge gained from teaching (proportional to actual improvement)
- Knowledge recovered from timely review
- Diagnostic value from first-time quizzing
- Ending the session at an appropriate time (student was fatigued)

**Penalties:**
- Teaching content the student already mastered (wasted time)
- Teaching a topic whose prerequisites aren't met (student can't learn it)
- Redundant quizzing without teaching in between
- Ending the session while the student still had capacity
- Causing repeated failure by pushing content that's too hard

### Grading Breakdown

| Component | Weight | What It Measures |
|-----------|--------|-----------------|
| Knowledge Acquisition | 30% | How much the student learned across all topics |
| Retention | 25% | How much knowledge persists after forgetting (Task 3) |
| Engagement | 20% | Session completion rate and motivation preserved |
| Efficiency | 15% | Proportion of actions that produced useful outcomes |
| Adaptivity | 10% | Whether the agent changed strategy when the student struggled |

All scores normalized to [0.0, 1.0]. Deterministic given the same seed.

## Curriculum

10 Python topics with prerequisite dependencies:

```
variables ──► conditionals ──► loops ──► functions ──► error_handling
    │              │             │           │
    ▼              │             ▼           ▼
data_types         │           lists      file_io
    │              │             │
    ▼              │             ▼
 strings ──────────┘        dictionaries
```

Each topic has content in 3 formats and quizzes at 3 difficulty levels. The curriculum structure is configurable — while this version uses Python, the environment engine is subject-agnostic.

## Setup & Usage

### Local Development

```bash
cd eduxrl

# Install dependencies
uv sync

# Start the server
uv run server

# In another terminal — run the LLM baseline
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-token"
python inference.py
```

### Docker

```bash
docker build -t eduxrl:latest -f server/Dockerfile .
docker run -d -p 8000:8000 eduxrl:latest

curl http://localhost:8000/health
```

### Deploy to HF Spaces

```bash
openenv push --repo-id your-username/eduxrl
```

### API

```python
import requests

# Start a new episode
resp = requests.post("http://localhost:8000/reset", json={"task_id": "task1"})
obs = resp.json()

# Take an action
action = {
    "action_type": "teach",
    "topic": "variables",
    "format": "exercise",
    "difficulty": "easy"
}
resp = requests.post("http://localhost:8000/step", json={"action": action})
result = resp.json()
print(f"Reward: {result['reward']}, Done: {result['done']}")
```

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start a new episode (optional `task_id`, `seed`) |
| `/step` | POST | Execute a TeachingAction |
| `/state` | GET | Current session state |
| `/health` | GET | Health check |
| `/tasks` | GET | List tasks and action schema |
| `/grader` | POST | Score after episode completion |
| `/baseline` | POST | Run heuristic baseline across all tasks |

## Baseline Scores

Expected baseline performance with an LLM agent:

| Task | Expected Score | Steps | Notes |
|------|---------------|-------|-------|
| task1 (Easy) | 0.55 - 0.70 | ~18 | Straightforward sequencing |
| task2 (Medium) | 0.40 - 0.55 | ~22 | Gap detection and style discovery are challenging |
| task3 (Hard) | 0.25 - 0.40 | ~50 | Long-term planning across sessions is genuinely difficult |

## Project Structure

```
eduxrl/
├── __init__.py                     # Package exports
├── models.py                       # EduxrlAction, EduxrlObservation (Pydantic)
├── client.py                       # EduxrlEnv HTTP/WebSocket client
├── inference.py                    # LLM baseline agent (submission script)
├── openenv.yaml                    # OpenEnv manifest
├── pyproject.toml                  # Dependencies
└── server/
    ├── app.py                      # FastAPI application (create_app + custom endpoints)
    ├── eduxrl_environment.py       # EduxrlEnvironment — core reset/step/state
    ├── student_model.py            # SimulatedStudent (cognitive science models)
    ├── curriculum.py               # 10 Python topics with prerequisites
    ├── learning_grader.py          # 5-dimension grading + per-step rewards
    ├── task_definitions.py         # Task registry
    ├── task1_steady_learner.py     # Easy: fresh student, 5 topics
    ├── task2_struggling_student.py # Medium: gaps + motivation + style discovery
    ├── task3_forgetting_student.py # Hard: multi-session with forgetting
    ├── dashboard.py                # Landing page HTML
    └── Dockerfile                  # Container build
```
