# Backend Changes

This file tracks all changes made to the backend (`backend/` directory).

## Changelog

### [Unreleased]
- Fixed static file serving: replaced manual route handlers with StaticFiles middleware
- Updated pyproject.toml with UV configuration and better description
- Updated run.sh to auto-sync dependencies with `uv sync` before starting
- Initial tracking file created
