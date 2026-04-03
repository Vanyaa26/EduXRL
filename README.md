# EduXRL — Adaptive Learning Path RL Environment

An OpenEnv-compliant RL environment where AI agents learn to teach. Simulated students follow real cognitive science models — Ebbinghaus forgetting curves, power law of learning, motivation dynamics. The agent controls the teaching path.

**See [`eduxrl/README.md`](eduxrl/README.md) for full documentation.**

## Quick Start

```bash
cd eduxrl
pip install -e ".[dev]"
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

## Tasks

| Task | Difficulty | Challenge |
|------|-----------|-----------|
| task1 | Easy | Fresh student, basic sequencing |
| task2 | Medium | Knowledge gaps + motivation management + style discovery |
| task3 | Hard | Multi-session with forgetting between days |
