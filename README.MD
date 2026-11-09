# AgentSwarm

AI-powered autonomous browser testing and self-healing platform.

AgentSwarm uses AI agents and Playwright to test web applications, detect failures, collect evidence, and suggest repairs for broken test scripts.

## Features

- AI-generated test scenarios and user personas
- Browser automation with Playwright
- Multi-agent workflow orchestration
- Bug detection with screenshots, console logs, and network evidence
- Self-healing suggestions for broken test locators
- Optional repository analysis and GitHub issue/PR integration

> **Status:** Hackathon project in development. Features may be incomplete.

## Architecture

- **Frontend:** HTML, CSS, JavaScript
- **Backend:** Python, FastAPI
- **AI:** Google Gemini API
- **Agent orchestration:** LangGraph
- **Browser automation:** Playwright
- **Database:** PostgreSQL

## Repository Structure

```text
AgentSwarm/
├── frontend/           # HTML, CSS, JavaScript dashboard
├── backend/            # API, agents, and browser runner
├── README.md
└── .gitignore
```

## Getting Started

### Prerequisites

- Python 3.11+
- Git
- A Google Gemini API key for AI-powered features
- A modern web browser

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd AgentSwarm
```

### 2. Set Up the Backend

```bash
cd backend
python -m venv .venv
```

Activate the virtual environment:

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
source .venv/bin/activate
```

Install dependencies once `requirements.txt` is available:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the backend directory:

```env
GOOGLE_API_KEY=your_gemini_api_key
DATABASE_URL=your_postgresql_connection_string
```

Run the API when `app/main.py` is implemented:

```bash
uvicorn app.main:app --reload
```

### 3. Set Up the Frontend

The frontend is built using standard HTML, CSS, and JavaScript.

Navigate to the frontend directory:

```bash
cd ../frontend
```

Open `index.html` in your browser or use the Live Server extension in Visual Studio Code to run the frontend locally.

## Development Roadmap

1. Build the dashboard and FastAPI foundation.
2. Add a Playwright worker for real browser tests.
3. Generate test scenarios with Gemini.
4. Coordinate planning, execution, and triage with LangGraph.
5. Collect evidence and produce structured bug reports.
6. Add self-healing locator suggestions.
7. Add repository-aware diagnosis and optional GitHub integration.

## Security Notes

- Only test websites you own or are explicitly authorized to test.
- Use staging environments and synthetic test accounts.
- Keep API keys and credentials in environment variables; never commit `.env` files.
- Run browser sessions and repository code in isolated environments.
- Require human approval before creating pull requests or applying application-code changes.

## Contributing

This project is currently being developed for a hackathon.

Contributions, ideas, and bug reports are welcome.
