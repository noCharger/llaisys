# Backend Refactoring & Multi-Tenancy Architecture

## Overview
This document describes the refactoring of the Llaisys Chatbot backend from a monolithic `server.py` to a modular, layered architecture. The primary goal was to improve maintainability, testability, and introduce multi-tenancy support.

## Architecture

The backend is now organized into the following layers under `src/chatbot/backend/app/`:

### 1. Routers (`app/routers/`)
-   **Role**: Define API endpoints, handle request validation, and delegate to services.
-   **Files**:
    -   `chat.py`: `/v1/chat/completions` endpoint.
    -   `context.py`: `/v1/context/*` endpoints.
    -   `health.py`: Health checks and configuration.

### 2. Services (`app/services/`)
-   **Role**: Encapsulate business logic and orchestrate operations.
-   **Files**:
    -   `chat_service.py`: Handles chat completion logic, message pruning, and session management.
    -   `model_service.py`: Wraps the `llaisys` model interaction (loading, inference, session creation/destruction).

### 3. Data Access Object (DAO) (`app/dao/`)
-   **Role**: Abstract data storage and retrieval. **Enforces tenant isolation.**
-   **Files**:
    -   `session_dao.py`: Manages active inference sessions.
    -   `context_dao.py`: Manages context items (knowledge base).

### 4. Models (`app/models/`)
-   **Role**: Define data structures and schemas.
-   **Files**:
    -   `chat.py`: Pydantic models for chat API.
    -   `context.py`: Pydantic models for context API and `TaggedItem`.
    -   `session.py`: `SessionMeta` data class.

### 5. Middleware (`app/middleware/`)
-   **Role**: Handle cross-cutting concerns.
-   **Files**:
    -   `security.py`: Adds security headers.
    -   `observability.py`: Prometheus metrics and OpenTelemetry tracing.

### 6. Configuration & Dependencies
-   `config.py`: Centralized configuration management.
-   `dependencies.py`: Dependency injection setup (singletons for services and DAOs).
-   `main.py`: Application factory and lifespan management.

## Multi-Tenancy Implementation

Multi-tenancy is implemented at the **DAO layer** and **API layer**:

1.  **API Layer**: Requests can include a `tenant_id` (optional, defaults to "default").
2.  **Models**:
    -   `SessionMeta` includes `tenant_id`.
    -   `TaggedItem` includes `tenant_id`.
3.  **DAO Layer**:
    -   `SessionDAO.find_best_prefix_match`: Only reuses sessions belonging to the requesting tenant.
    -   `ContextDAO`: Stores items in a dictionary keyed by `tenant_id` (e.g., `self.items[tenant_id][item_id]`).
    -   All DAO methods require `tenant_id` as an argument.

## Key Changes from `server.py`

-   **Modularization**: `server.py` is now a thin entry point that bootstraps the application defined in `app/main.py`.
-   **State Management**: Global `ServerState` is replaced by injected `ModelService`, `SessionDAO`, etc.
-   **Testing**: Unit tests added for DAO and Service layers, covering tenant isolation.

## Running the Server

The entry point remains the same:

```bash
python src/chatbot/backend/server.py --model <model_path> [options]
```

## Running Tests

```bash
PYTHONPATH=src pytest test/chatbot/backend/
```
