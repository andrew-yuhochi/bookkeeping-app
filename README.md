# Bookkeeping App

Household transaction tracking and financial planning automation — PDF statement ingestion, ML-powered categorization, and per-person expense splitting for a two-person Canadian household.

> Built with [Claude Code](https://claude.ai/code)

## Status

Phase 1 PoC — scaffolding complete, implementation in progress.

## Getting Started

### Prerequisites

- Python 3.11+
- Git

### Setup

```bash
cd projects/bookkeeping-app
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your configuration
```

## Project Structure

```
src/
├── config.py          # Configuration (Pydantic BaseSettings)
├── models/            # Pydantic data models
├── services/          # Business logic
├── integrations/      # External API clients
└── utils/             # Shared utilities
tests/
├── unit/
└── integration/
data/                  # Local data (gitignored)
notebooks/             # Exploration notebooks
```

## Documentation

- [Discovery Notes](../../docs/bookkeeping-app/poc/DISCOVERY-NOTES.md)
- [PRD](../../docs/bookkeeping-app/poc/PRD.md)
- [TDD](../../docs/bookkeeping-app/poc/TDD.md)
- [Data Sources](../../docs/bookkeeping-app/poc/DATA-SOURCES.md)
- [Tasks](../../docs/bookkeeping-app/poc/TASKS.md)
