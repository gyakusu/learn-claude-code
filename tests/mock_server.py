#!/usr/bin/env python3
"""
Mock Anthropic Messages API server for offline testing.

Provides a lightweight HTTP server that mimics the Anthropic /v1/messages
endpoint. Parses the conversation context and returns scripted tool_use or
text responses so that agent tutorials (s01-s12) can run without a real API
key or network connection.

Usage:
    # Terminal 1: start the mock server
    python tests/mock_server.py              # default port 9800

    # Terminal 2: run any agent against it
    ANTHROPIC_BASE_URL=http://localhost:9800 MODEL_ID=mock-model \
        python agents/s09_agent_teams.py
"""

import json
import re
import uuid
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 9800


def _id():
    return "msg_" + uuid.uuid4().hex[:16]


def _tool_id():
    return "toolu_" + uuid.uuid4().hex[:16]


def _text_response(text, model="mock-model"):
    return {
        "id": _id(),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 10},
    }


def _tool_response(tool_calls, model="mock-model"):
    content = []
    for name, inp in tool_calls:
        content.append({
            "type": "tool_use",
            "id": _tool_id(),
            "name": name,
            "input": inp,
        })
    return {
        "id": _id(),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 10},
    }


def _last_user_text(messages):
    """Extract the latest user text from the messages list."""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            return content.lower()
        if isinstance(content, list):
            # Could be tool_results or text blocks
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "tool_result":
                        return ""  # tool result turn, skip
                    if block.get("type") == "text":
                        return block.get("text", "").lower()
                elif isinstance(block, str):
                    return block.lower()
    return ""


def _has_tool(tools, name):
    return any(t.get("name") == name for t in (tools or []))


def _is_tool_result_turn(messages):
    """Check if the last user message is a tool_result."""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict) and first.get("type") == "tool_result":
                return True
        break
    return False


def _count_tool_result_turns(messages):
    """Count consecutive tool_result round-trips so far."""
    count = 0
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict) and first.get("type") == "tool_result":
                count += 1
                continue
        break
    return count


# ---------------------------------------------------------------------------
# Routing: decide what the mock "LLM" should do based on context
# ---------------------------------------------------------------------------

def decide(body):
    messages = body.get("messages", [])
    tools = body.get("tools", [])
    model = body.get("model", "mock-model")
    text = _last_user_text(messages)
    is_tool_result = _is_tool_result_turn(messages)
    tool_rounds = _count_tool_result_turns(messages)

    # After processing a tool result, usually finish with text
    if is_tool_result and tool_rounds >= 2:
        return _text_response("Done. All tasks completed successfully.", model)

    # ---- s09 teammate agent (has send_message but NOT spawn_teammate) ----
    if _has_tool(tools, "send_message") and not _has_tool(tools, "spawn_teammate"):
        if is_tool_result:
            return _text_response("Task complete. Sent results to lead.", model)
        # First turn: do some work then send a message back
        if "fix" in text or "cod" in text or "implement" in text:
            return _tool_response([
                ("bash", {"command": "echo 'Working on code task...'"}),
            ], model)
        if "test" in text or "verif" in text or "check" in text:
            return _tool_response([
                ("bash", {"command": "echo 'Running tests... all passed'"}),
            ], model)
        # Default teammate: send a status message to lead
        return _tool_response([
            ("send_message", {"to": "lead", "content": "Ready for instructions."}),
        ], model)

    # ---- s09 lead agent (has spawn_teammate) ----
    if _has_tool(tools, "spawn_teammate"):
        if is_tool_result and tool_rounds >= 1:
            # After first tool round, maybe do one more action then stop
            if "spawn" in str(messages).lower() and tool_rounds == 1:
                return _tool_response([
                    ("send_message", {
                        "to": "alice",
                        "content": "Please read the README and summarize it.",
                    }),
                ], model)
            return _text_response(
                "Team is set up. Alice (coder) and Bob (tester) are spawned. "
                "I've asked Alice to read the README. Use /team to check status "
                "and /inbox to read messages.", model)

        # Keyword-based routing
        if "spawn" in text and ("alice" in text or "bob" in text):
            calls = []
            if "alice" in text:
                calls.append(("spawn_teammate", {
                    "name": "alice",
                    "role": "coder",
                    "prompt": "You are alice, a coder. Write clean code.",
                }))
            if "bob" in text:
                calls.append(("spawn_teammate", {
                    "name": "bob",
                    "role": "tester",
                    "prompt": "You are bob, a tester. Verify code quality.",
                }))
            return _tool_response(calls, model)

        if "broadcast" in text:
            return _tool_response([
                ("broadcast", {"content": re.sub(r'.*broadcast\s*', '', text).strip() or "Status update from lead."}),
            ], model)

        if "send" in text:
            target = "alice"
            for name in ("alice", "bob"):
                if name in text:
                    target = name
                    break
            return _tool_response([
                ("send_message", {"to": target, "content": "Hello from lead!"}),
            ], model)

        if "inbox" in text or "check" in text or "read" in text:
            return _tool_response([
                ("read_inbox", {}),
            ], model)

        if "team" in text or "list" in text or "status" in text:
            return _tool_response([
                ("list_teammates", {}),
            ], model)

        # Default: suggest spawning
        return _text_response(
            "I'm the team lead. Try asking me to:\n"
            "- Spawn alice (coder) and bob (tester)\n"
            "- Send a message to a teammate\n"
            "- Broadcast a status update\n"
            "- Check my inbox", model)

    # ---- Generic agent (s01-s08 etc.) ----
    if _has_tool(tools, "bash"):
        if is_tool_result:
            return _text_response("Done.", model)
        if any(kw in text for kw in ("list", "ls", "file", "dir")):
            return _tool_response([("bash", {"command": "ls"})], model)
        if any(kw in text for kw in ("read", "cat", "show")):
            return _tool_response([("read_file", {"path": "README.md"})], model)
        return _text_response(f"I understand: '{text}'. What would you like me to do?", model)

    # Fallback
    return _text_response("Hello! I'm the mock assistant. How can I help?", model)


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = {}

        response = decide(body)
        payload = json.dumps(response).encode()

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        # Health check
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def log_message(self, format, *args):
        print(f"  [mock] {args[0]}")


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Mock Anthropic API server on http://127.0.0.1:{PORT}")
    print(f"Usage: ANTHROPIC_BASE_URL=http://127.0.0.1:{PORT} MODEL_ID=mock-model python agents/s09_agent_teams.py")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutdown.")
