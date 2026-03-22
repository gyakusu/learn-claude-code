#!/usr/bin/env python3
"""
mock_api.py - Offline mock for the Anthropic Messages API.

Responds to POST /v1/messages with scripted tool-use sequences,
so s01–s02 (and beyond) can run without a real API key.

Usage:
    python tests/mock_api.py              # starts on port 5123
    python tests/mock_api.py --port 5124  # custom port

Then point the agent at it:
    ANTHROPIC_BASE_URL=http://localhost:5123 MODEL_ID=mock python agents/s02_tool_use.py
"""

import argparse
import json
import re
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler

PORT = 5123


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _id():
    return "toolu_" + uuid.uuid4().hex[:24]


def _msg_id():
    return "msg_" + uuid.uuid4().hex[:24]


def _text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def _tool_block(name: str, inp: dict) -> dict:
    return {"type": "tool_use", "id": _id(), "name": name, "input": inp}


def _response(content: list, stop: str = "end_turn") -> dict:
    return {
        "id": _msg_id(),
        "type": "message",
        "role": "assistant",
        "model": "mock",
        "content": content,
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {"input_tokens": 42, "output_tokens": 42},
    }


# ---------------------------------------------------------------------------
# Scenario engine — keyword-based dispatch
# ---------------------------------------------------------------------------

# Each scenario is (compiled_regex, handler_function).
# handler receives (user_text, tool_results, turn_number) and returns a
# response dict.  turn_number counts how many assistant messages have been
# sent for this scenario so far (tracked via tool_results presence).

SCENARIOS: list[tuple[re.Pattern, callable]] = []


def scenario(pattern: str):
    """Decorator: register a keyword regex for a scenario."""
    compiled = re.compile(pattern, re.IGNORECASE)
    def deco(fn):
        SCENARIOS.append((compiled, fn))
        return fn
    return deco


# ---- s01 / s02 demo scenarios ----

@scenario(r"read.*requirements|requirements.*read")
def _read_requirements(text, tool_results, turn):
    if tool_results:
        return _response([_text_block(
            "The file `requirements.txt` contains the project's Python "
            "dependencies: anthropic and python-dotenv."
        )])
    return _response([
        _tool_block("read_file", {"path": "requirements.txt"}),
    ], stop="tool_use")


@scenario(r"create.*greet|greet.*create|write.*greet")
def _create_greet(text, tool_results, turn):
    if tool_results:
        return _response([_text_block(
            "Created `greet.py` with a `greet(name)` function."
        )])
    code = 'def greet(name):\n    return f"Hello, {name}!"\n'
    return _response([
        _tool_block("write_file", {"path": "greet.py", "content": code}),
    ], stop="tool_use")


@scenario(r"edit.*greet.*docstring|docstring.*greet|add.*docstring")
def _edit_greet(text, tool_results, turn):
    if tool_results:
        return _response([_text_block(
            "Added a docstring to the `greet` function in `greet.py`."
        )])
    return _response([
        _tool_block("edit_file", {
            "path": "greet.py",
            "old_text": "def greet(name):\n",
            "new_text": 'def greet(name):\n    """Return a greeting for the given name."""\n',
        }),
    ], stop="tool_use")


@scenario(r"read.*greet|verify.*greet|show.*greet|cat.*greet")
def _read_greet(text, tool_results, turn):
    if tool_results:
        return _response([_text_block(
            "The edit worked — `greet.py` now has a docstring."
        )])
    return _response([
        _tool_block("read_file", {"path": "greet.py"}),
    ], stop="tool_use")


@scenario(r"(ls|list|dir|files)")
def _ls(text, tool_results, turn):
    if tool_results:
        return _response([_text_block("Here are the files in the workspace.")])
    return _response([
        _tool_block("bash", {"command": "ls -la"}),
    ], stop="tool_use")


# ---- fallback ----

def _fallback(text, tool_results, turn):
    """Echo back as a text response — no tool use."""
    return _response([_text_block(
        f"(mock) I received your message. In offline mode I only handle "
        f"the scripted demo scenarios from the tutorial. "
        f"Try: 'Read the file requirements.txt'"
    )])


# ---------------------------------------------------------------------------
# Route a request to the right scenario
# ---------------------------------------------------------------------------

def route(messages: list) -> dict:
    """Pick a scenario based on the latest user text."""
    # Find the last user message text
    user_text = ""
    tool_results = []
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                user_text = content
                break
            elif isinstance(content, list):
                # Could be tool_results or text blocks
                texts = [b.get("text", "") for b in content
                         if isinstance(b, dict) and b.get("type") == "text"]
                results = [b for b in content
                           if isinstance(b, dict) and b.get("type") == "tool_result"]
                if texts:
                    user_text = " ".join(texts)
                    break
                if results:
                    tool_results = results
                    # Keep looking for the original user text
                    continue
        # If we hit an assistant message while looking through tool_results,
        # keep going to find the original user text
        continue

    # If we only found tool_results, search further back for original text
    if not user_text and tool_results:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    user_text = content
                    break

    has_results = any(
        isinstance(m.get("content"), list) and
        any(isinstance(b, dict) and b.get("type") == "tool_result"
            for b in m["content"])
        for m in messages
        if m.get("role") == "user"
    )

    for pattern, handler in SCENARIOS:
        if pattern.search(user_text):
            return handler(user_text, has_results, 0)

    return _fallback(user_text, has_results, 0)


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class MockHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/v1/messages":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        messages = body.get("messages", [])
        resp = route(messages)

        payload = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        # Prefix with [mock] for clarity
        print(f"[mock] {fmt % args}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Mock Anthropic API server")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    server = HTTPServer(("127.0.0.1", args.port), MockHandler)
    print(f"[mock] Anthropic API mock running on http://127.0.0.1:{args.port}")
    print(f"[mock] Usage: ANTHROPIC_BASE_URL=http://localhost:{args.port} "
          f"MODEL_ID=mock python agents/s02_tool_use.py")
    print(f"[mock] Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[mock] Stopped.")


if __name__ == "__main__":
    main()
