"""
Landing page for EduXRL.
Minimal, clean HTML that explains the project and links to the playground.
"""


def get_dashboard_html() -> str:
    return '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EduXRL — Adaptive Learning Path RL Environment</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
:root { --bg: #0f1117; --card: #1a1d27; --border: #2a2d37; --text: #e0e0e0; --muted: #888; --accent: #6c63ff; --green: #4caf50; --yellow: #ffc107; --red: #f44336; --mono: 'SF Mono', 'Cascadia Code', Consolas, monospace; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); line-height: 1.6; }
.container { max-width: 860px; margin: 0 auto; padding: 40px 24px; }

h1 { font-size: 2rem; margin-bottom: 8px; }
h1 span { color: var(--accent); }
.subtitle { color: var(--muted); font-size: 1.05rem; margin-bottom: 32px; }

.hero { margin-bottom: 36px; }
.hero-hook { font-size: 1.5rem; font-weight: 700; line-height: 1.3; margin-bottom: 16px; }
.hero-hook em { color: var(--accent); font-style: normal; }
.hero p { color: var(--muted); font-size: 0.95rem; margin-bottom: 10px; max-width: 640px; }
.hero .highlight { color: var(--text); font-weight: 500; }

.card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 24px; margin-bottom: 20px; }
.card h2 { font-size: 1.1rem; margin-bottom: 12px; color: var(--accent); }

.badge { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; margin-right: 6px; }
.badge-green { background: #1b3d1b; color: var(--green); }
.badge-yellow { background: #3d3a1b; color: var(--yellow); }
.badge-red { background: #3d1b1b; color: var(--red); }
.badge-accent { background: #1b1b3d; color: var(--accent); }

table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 0.9rem; }
th { text-align: left; color: var(--muted); font-weight: 500; padding: 8px 12px; border-bottom: 1px solid var(--border); }
td { padding: 8px 12px; border-bottom: 1px solid var(--border); }

.science-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 12px 0; }
.science-item { background: var(--bg); border-radius: 8px; padding: 14px; }
.science-item strong { display: block; font-size: 0.85rem; margin-bottom: 4px; }
.science-item span { color: var(--muted); font-size: 0.8rem; }

code { background: var(--bg); padding: 2px 6px; border-radius: 4px; font-family: var(--mono); font-size: 0.85rem; }

.actions { display: flex; gap: 12px; margin: 24px 0; flex-wrap: wrap; }
.btn { display: inline-block; padding: 10px 24px; border-radius: 8px; font-weight: 600; text-decoration: none; font-size: 0.9rem; }
.btn-primary { background: var(--accent); color: white; }
.btn-secondary { background: var(--card); color: var(--text); border: 1px solid var(--border); }
.btn:hover { opacity: 0.9; }

.endpoint-grid { display: grid; grid-template-columns: auto 1fr; gap: 4px 16px; font-size: 0.85rem; font-family: var(--mono); }
.method { color: var(--green); font-weight: 600; }

.flow { display: flex; align-items: center; gap: 8px; margin: 16px 0; flex-wrap: wrap; }
.flow-step { background: var(--bg); padding: 8px 14px; border-radius: 8px; font-size: 0.85rem; }
.flow-arrow { color: var(--muted); }

@media (max-width: 600px) { .science-grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div class="container">

<div class="hero">
    <h1>🎓 <span>EduXRL</span></h1>
    <div class="hero-hook">You study for 3 hours.<br>Two days later, you remember <em>almost nothing.</em></div>
    <p>That's not a flaw in your brain — it's how memory works. Ebbinghaus proved in 1885 that we forget most of what we learn within 48 hours without review. The problem isn't studying harder. It's studying <span class="highlight">smarter</span> — the right topic, at the right time, at the right difficulty.</p>
    <p>Most learning platforms ignore this. They give everyone the same fixed path. The result: most learners never finish what they start.</p>
    <p><span class="highlight">EduXRL is an RL environment where AI agents learn to solve this.</span> They teach a simulated student whose memory, motivation, and fatigue follow real cognitive science — and discover the optimal teaching strategy through trial and error.</p>
</div>

<div class="actions">
    <a href="/web/" class="btn btn-primary">Open Playground</a>
    <a href="/docs" class="btn btn-secondary">API Docs</a>
    <a href="/tasks" class="btn btn-secondary">View Tasks</a>
</div>

<div class="card">
    <h2>How it works</h2>
    <div class="flow">
        <div class="flow-step">Agent picks action<br><code>teach loops, exercise, medium</code></div>
        <div class="flow-arrow">→</div>
        <div class="flow-step">Student model computes<br><code>ΔK = f(ZPD, format, motivation)</code></div>
        <div class="flow-arrow">→</div>
        <div class="flow-step">Returns reward<br><code>+0.24 (knowledge gained)</code></div>
    </div>
    <p style="color:var(--muted);font-size:0.85rem;margin-top:8px;">The student model is the "physics engine." It computes how much learning happens based on the power law of learning, Ebbinghaus forgetting curve, and zone of proximal development.</p>
</div>

<div class="card">
    <h2>Scientific grounding</h2>
    <div class="science-grid">
        <div class="science-item">
            <strong>Power Law of Learning</strong>
            <span>Knowledge grows with diminishing returns. Newell & Rosenbloom, 1981.</span>
        </div>
        <div class="science-item">
            <strong>Ebbinghaus Forgetting Curve</strong>
            <span>Memory decays exponentially: R = e<sup>−t/S</sup>. Ebbinghaus, 1885.</span>
        </div>
        <div class="science-item">
            <strong>Zone of Proximal Development</strong>
            <span>Optimal difficulty is just above current ability. Vygotsky, 1978.</span>
        </div>
        <div class="science-item">
            <strong>Spacing Effect</strong>
            <span>Spaced review beats cramming. Cepeda et al., 2006 (254 studies).</span>
        </div>
    </div>
</div>

<div class="card">
    <h2>3 Tasks</h2>
    <table>
        <tr><th>Task</th><th>Difficulty</th><th>Challenge</th><th>Baseline</th></tr>
        <tr>
            <td><strong>Steady Learner</strong></td>
            <td><span class="badge badge-green">Easy</span></td>
            <td>5 topics, fresh student, single session</td>
            <td>0.62</td>
        </tr>
        <tr>
            <td><strong>Struggling Student</strong></td>
            <td><span class="badge badge-yellow">Medium</span></td>
            <td>Knowledge gaps + fragile motivation + hidden format preference</td>
            <td>0.56</td>
        </tr>
        <tr>
            <td><strong>Forgetting Student</strong></td>
            <td><span class="badge badge-red">Hard</span></td>
            <td>3 sessions over a week, Ebbinghaus decay between sessions</td>
            <td>0.49</td>
        </tr>
    </table>
</div>

<div class="card">
    <h2>Action space</h2>
    <table>
        <tr><th>Action</th><th>What it does</th></tr>
        <tr><td><code>teach</code></td><td>Present new content (topic + format + difficulty)</td></tr>
        <tr><td><code>quiz</code></td><td>Test student — reveals knowledge level</td></tr>
        <tr><td><code>review</code></td><td>Re-teach a topic to fight forgetting</td></tr>
        <tr><td><code>end_session</code></td><td>End session (good if fatigued, bad if student still fresh)</td></tr>
    </table>
    <p style="margin-top:10px;font-size:0.85rem;color:var(--muted);">
        Formats: <code>explanation</code> <code>worked_example</code> <code>exercise</code> &nbsp;|&nbsp;
        Difficulties: <code>easy</code> <code>medium</code> <code>hard</code>
    </p>
</div>

<div class="card">
    <h2>Reward design</h2>
    <table>
        <tr><th>Signal</th><th>Source</th></tr>
        <tr><td style="color:var(--green);">+ knowledge gained</td><td>Proportional to actual learning (power law)</td></tr>
        <tr><td style="color:var(--green);">+ knowledge recovered</td><td>Review recovers forgotten material (spacing effect)</td></tr>
        <tr><td style="color:var(--green);">+ good session end</td><td>Ending when student is fatigued</td></tr>
        <tr><td style="color:var(--red);">− wasted time</td><td>Teaching mastered content</td></tr>
        <tr><td style="color:var(--red);">− prerequisite violation</td><td>Teaching topic student can't learn yet</td></tr>
        <tr><td style="color:var(--red);">− frustration</td><td>Pushing content that's too hard</td></tr>
    </table>
    <p style="margin-top:10px;font-size:0.85rem;color:var(--muted);">All rewards derived from measurable student outcomes. No human labels.</p>
</div>

<div class="card">
    <h2>Grading (5 dimensions)</h2>
    <table>
        <tr><th>Dimension</th><th>Weight</th><th>What it measures</th></tr>
        <tr><td>Knowledge Acquisition</td><td>30%</td><td>How much the student learned</td></tr>
        <tr><td>Retention</td><td>25%</td><td>Knowledge that persists after forgetting</td></tr>
        <tr><td>Engagement</td><td>20%</td><td>Session completion + motivation preserved</td></tr>
        <tr><td>Efficiency</td><td>15%</td><td>Useful actions / total actions</td></tr>
        <tr><td>Adaptivity</td><td>10%</td><td>Strategy changes when student struggles</td></tr>
    </table>
</div>

<div class="card">
    <h2>API endpoints</h2>
    <div class="endpoint-grid">
        <span class="method">POST</span><span>/reset — start a new episode</span>
        <span class="method">POST</span><span>/step — execute a teaching action</span>
        <span class="method">GET</span><span>/state — current session state</span>
        <span class="method">GET</span><span>/health — health check</span>
        <span class="method">GET</span><span>/tasks — list tasks + action schema</span>
        <span class="method">POST</span><span>/grader — score after episode</span>
        <span class="method">POST</span><span>/baseline — heuristic baseline scores</span>
    </div>
</div>

<div class="card">
    <h2>Try it</h2>
    <p style="margin-bottom:12px;">Use the <a href="/web/" style="color:var(--accent);">Playground</a> or call the API directly:</p>
    <pre style="background:var(--bg);padding:14px;border-radius:8px;font-size:0.82rem;overflow-x:auto;"><code># Reset to task1
curl -X POST /reset -d '{"task_id":"task1"}'

# Teach variables
curl -X POST /step -d '{"action":{"action_type":"teach","topic":"variables","format":"exercise","difficulty":"easy"}}'

# Check score
curl -X POST /grader -d '{"task_id":"task1"}'</code></pre>
</div>

<p style="text-align:center;color:var(--muted);font-size:0.8rem;margin-top:32px;">
    EduXRL — Built for the OpenEnv Hackathon
</p>

</div>
</body>
</html>'''
