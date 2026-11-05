QUANTUM HVAC LAB — HOSTED WEB VERSION

The hosted version runs Raphael Langer's original Qiskit functions through
a Flask/Gunicorn web backend. It supports 1–5 QAOA layers. The original
development scripts, report, README, and original requirements are preserved.

FILES
web/app.py              Flask routes, PDF report, run endpoint, bounded cache
web/index.html          Public-facing GUI with a visible five-layer selector
web/gunicorn.conf.py    Production process configuration
requirements-web.txt   Hosting dependencies
render.yaml            Render Free web-service configuration
.python-version        Python 3.12 selection

RUN THE WEB VERSION LOCALLY
From the project folder:
  python -m pip install -r requirements-web.txt
  python -m gunicorn --config web/gunicorn.conf.py web.app:app
Then open http://127.0.0.1:8765
Gunicorn is intended for Linux/macOS; Windows users can continue to use
the existing local GUI launcher instead.

DEPLOY TO RENDER
The source, including web/, gui/, requirements-web.txt, render.yaml,
and .python-version, must first be pushed to the connected Git repository.
The deployment files are maintained on the quantum-hvac-web-demo branch.

Create a Python web service from the branch containing the new files:
  Build command: pip install -r requirements-web.txt
  Start command: gunicorn --config web/gunicorn.conf.py web.app:app
  Health check: /health
  Instance type: Free
  Environment: OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1
Alternatively, use the included render.yaml as a Blueprint.

Render assigns the actual public URL after service creation and deployment.
Verify the live URL and /health after deployment succeeds. No IBM account, hardware
access, database, or secret API keys are required by this app.

The prepared service uses Render's Free instance type. Render's free instances
sleep after 15 minutes of inactivity and can take about a minute to wake up.
The included usage limits still apply; see https://render.com/docs/free

PUBLIC DEMO BEHAVIOR
- All results use ideal Qiskit statevectors and seeded finite-shot sampling.
- Historical hardware/noise results are in a separate, clearly labeled tab.
- One compute job runs at a time; another visitor gets a retry message.
- The most recent 16 identical-input results may be reused from memory.
- Input and result JSON can be downloaded without storing visitor data.
- /health returns build web-5-layers-v1 and max_layers=5, which helps identify
  whether an old or new build is being served.
- Responses disable browser caching so an old three-layer page is not reused.

AUTHORSHIP
Original modeling, QUBO and QAOA solver functions: Raphael Langer.
GUI, web backend, and integration: AI-assisted.
