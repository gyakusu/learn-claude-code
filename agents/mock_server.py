#!/usr/bin/env python3
"""
Mock Anthropic API server for offline demos.

Simulates the Messages API (/v1/messages) with scripted responses
that demonstrate autonomous agent behavior (task creation, spawning,
auto-claiming, idle cycles).

Usage:
    # Terminal 1: start the mock server
    python agents/mock_server.py

    # Terminal 2: run s11 against it
    ANTHROPIC_BASE_URL=http://localhost:18088 MODEL_ID=mock ANTHROPIC_API_KEY=mock \
        python agents/s11_autonomous_agents.py

    # Or simply:
    python agents/s11_autonomous_agents.py --mock
"""

import json
import re
import threading
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler

PORT = 18088

# Per-agent conversation state
_agent_state: dict[str, dict] = {}
_state_lock = threading.Lock()


def _msg_id() -> str:
    return f"msg_{uuid.uuid4().hex[:12]}"


def _tool_id() -> str:
    return f"toolu_{uuid.uuid4().hex[:12]}"


def _text(t: str) -> dict:
    return {"type": "text", "text": t}


def _tool(name: str, inp: dict) -> dict:
    return {"type": "tool_use", "id": _tool_id(), "name": name, "input": inp}


def _response(content: list, stop_reason: str = "end_turn") -> dict:
    return {
        "id": _msg_id(),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": "mock-offline",
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 100, "output_tokens": 50},
    }


# ---------------------------------------------------------------------------
# Teammate (autonomous worker) response logic
# ---------------------------------------------------------------------------

def _get_agent_name(system: str) -> str:
    m = re.search(r"You are '(\w+)'", system)
    return m.group(1) if m else "worker"


def _last_user_text(messages: list) -> str:
    """Extract the last plain-text user message."""
    for m in reversed(messages):
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            return m["content"]
    return ""


def _handle_teammate(system: str, messages: list) -> dict:
    name = _get_agent_name(system)
    with _state_lock:
        if name not in _agent_state:
            _agent_state[name] = {"turn": 0}
        _agent_state[name]["turn"] += 1
        turn = _agent_state[name]["turn"]

    last = _last_user_text(messages)
    has_task = "<auto-claimed>" in last

    # Woken up with an auto-claimed task: work then idle
    if has_task:
        if turn % 2 == 1:
            return _response([
                _text(f"Working on the claimed task."),
                _tool("bash", {"command": f"echo '[{name}] Task completed successfully'"}),
            ], "tool_use")
        return _response([
            _text("Task done. Going idle to look for more work."),
            _tool("idle", {}),
        ], "tool_use")

    # Normal first-spawn cycle
    if turn == 1:
        return _response([
            _text(f"I'm {name}. Starting work."),
            _tool("bash", {"command": f"echo '[{name}] Initialized and ready'"}),
        ], "tool_use")
    if turn <= 3:
        return _response([
            _text("No more immediate work. Going idle to poll for tasks."),
            _tool("idle", {}),
        ], "tool_use")
    # Fallback after being woken
    return _response([
        _text("Continuing work."),
        _tool("bash", {"command": f"echo '[{name}] Processing...'"}),
    ], "tool_use")


# ---------------------------------------------------------------------------
# Lead (orchestrator) response logic
# ---------------------------------------------------------------------------

_SAMPLE_TASKS = [
    {"id": 1, "subject": "Set up database schema",
     "description": "Create tables for users and orders", "status": "pending"},
    {"id": 2, "subject": "Build REST API endpoints",
     "description": "CRUD endpoints for /users and /orders", "status": "pending"},
    {"id": 3, "subject": "Write unit tests",
     "description": "Test coverage for API endpoints", "status": "pending"},
]


def _handle_lead(messages: list) -> dict:
    last_msg = messages[-1] if messages else {}

    # --- continuation after tool results ---
    if last_msg.get("role") == "user" and isinstance(last_msg.get("content"), list):
        ctx = _last_user_text(messages).lower()
        tool_rounds = sum(
            1 for m in messages
            if m.get("role") == "user" and isinstance(m.get("content"), list)
        )
        if "task" in ctx and "spawn" in ctx:
            if tool_rounds == 1:
                return _response([
                    _text("Tasks created. Now spawning teammates."),
                    _tool("spawn_teammate", {
                        "name": "alice", "role": "backend",
                        "prompt": "You are alice. Check the task board and work on available tasks."}),
                    _tool("spawn_teammate", {
                        "name": "bob", "role": "frontend",
                        "prompt": "You are bob. Check the task board and work on available tasks."}),
                ], "tool_use")
            return _response([_text(
                "Done! Created 3 tasks and spawned alice and bob. "
                "They will autonomously scan the task board, claim unclaimed tasks, "
                "work on them, and go idle when done.\n\n"
                "Use `/tasks` to see task status and `/team` to monitor teammates."
            )])
        if "spawn" in ctx:
            return _response([_text(
                "Teammate spawned. They will autonomously find and claim tasks."
            )])
        if "task" in ctx:
            return _response([_text(
                "Tasks created on the board. Teammates will auto-claim them when idle."
            )])
        return _response([_text("Done.")])

    # --- initial user message ---
    user = _last_user_text(messages).lower()

    if "task" in user and "spawn" in user:
        content = [_text("Creating 3 tasks and spawning teammates.")]
        for t in _SAMPLE_TASKS:
            content.append(_tool("write_file", {
                "path": f".tasks/task_{t['id']}.json",
                "content": json.dumps(t, indent=2),
            }))
        return _response(content, "tool_use")

    if "spawn" in user:
        m = re.search(r"spawn\s+(\w+)", user)
        name = m.group(1) if m else "coder"
        return _response([
            _text(f"Spawning {name}."),
            _tool("spawn_teammate", {
                "name": name, "role": "developer",
                "prompt": f"You are {name}. Check the task board for available tasks and work on them."}),
        ], "tool_use")

    if "task" in user:
        content = [_text("Creating tasks on the board.")]
        for t in _SAMPLE_TASKS:
            content.append(_tool("write_file", {
                "path": f".tasks/task_{t['id']}.json",
                "content": json.dumps(t, indent=2),
            }))
        return _response(content, "tool_use")

    if "depend" in user or "block" in user:
        tasks = [
            {"id": 10, "subject": "Design API schema", "status": "pending"},
            {"id": 11, "subject": "Implement API", "status": "pending", "blockedBy": [10]},
            {"id": 12, "subject": "Write API tests", "status": "pending", "blockedBy": [11]},
        ]
        content = [_text("Creating tasks with dependencies.")]
        for t in tasks:
            content.append(_tool("write_file", {
                "path": f".tasks/task_{t['id']}.json",
                "content": json.dumps(t, indent=2),
            }))
        return _response(content, "tool_use")

    return _response([_text(
        "I'm the team lead. Teammates are autonomous -- they find work themselves.\n"
        "Try: 'Create 3 tasks on the board, then spawn alice and bob.'"
    )])


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        system = body.get("system", "")
        messages = body.get("messages", [])

        if "You are '" in system and "team:" in system:
            resp = _handle_teammate(system, messages)
        else:
            resp = _handle_lead(messages)

        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        pass  # suppress per-request logs


def start_background(port: int = PORT) -> HTTPServer:
    """Start the mock server in a daemon thread. Returns the server instance."""
    server = HTTPServer(("127.0.0.1", port), _Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


if __name__ == "__main__":
    print(f"Mock Anthropic API server on http://localhost:{PORT}")
    print()
    print("Usage:")
    print(f"  ANTHROPIC_BASE_URL=http://localhost:{PORT} \\")
    print(f"  MODEL_ID=mock ANTHROPIC_API_KEY=mock \\")
    print(f"    python agents/s11_autonomous_agents.py")
    print()
    print("Or simply:  python agents/s11_autonomous_agents.py --mock")
    print()
    server = HTTPServer(("", PORT), _Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
