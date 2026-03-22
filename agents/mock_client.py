"""
mock_client.py - Offline mock for Anthropic client.

Provides MockAnthropic that mimics client.messages.create() with scripted
responses driven by keyword matching. No API key or network required.

Usage:
    USE_MOCK=1 python agents/s10_team_protocols.py
"""

import re
import uuid
from dataclasses import dataclass, field


@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ToolUseBlock:
    name: str
    input: dict
    id: str = field(default_factory=lambda: f"toolu_{uuid.uuid4().hex[:12]}")
    type: str = "tool_use"


@dataclass
class MockResponse:
    content: list
    stop_reason: str = "end_turn"


class MockMessages:
    """Pattern-based mock that returns tool_use or text responses."""

    def create(self, *, model, system, messages, tools, max_tokens, **kw):
        last = self._last_user_text(messages)
        tool_names = {t["name"] for t in tools}

        # --- Teammate behaviour ---
        if "spawn_teammate" not in tool_names:
            return self._teammate_response(last, messages, tool_names)

        # --- Lead behaviour ---
        return self._lead_response(last, messages, tool_names)

    # ------------------------------------------------------------------ lead
    def _lead_response(self, text, messages, tool_names):
        low = text.lower()

        # spawn request
        m = re.search(r"spawn\s+(\w+)", low)
        if m:
            name = m.group(1)
            role = "coder"
            if "test" in low:
                role = "tester"
            prompt = text
            return MockResponse(
                content=[
                    TextBlock(f"Spawning {name} as {role}."),
                    ToolUseBlock("spawn_teammate", {
                        "name": name, "role": role,
                        "prompt": f"You are {name}. Task: {prompt}",
                    }),
                ],
                stop_reason="tool_use",
            )

        # shutdown request
        m = re.search(r"shutdown\s+(\w+)|(\w+).*shut\s*down", low)
        if m and "shutdown_request" in tool_names:
            name = m.group(1) or m.group(2)
            # filter out non-name words
            if name in ("request", "her", "his", "the", "please", "then"):
                name = self._extract_name_from_context(text, messages)
            return MockResponse(
                content=[
                    TextBlock(f"Requesting shutdown for {name}."),
                    ToolUseBlock("shutdown_request", {"teammate": name}),
                ],
                stop_reason="tool_use",
            )

        # list teammates
        if any(w in low for w in ("list", "team", "/team", "status")):
            return MockResponse(
                content=[ToolUseBlock("list_teammates", {})],
                stop_reason="tool_use",
            )

        # plan approval (approve)
        m = re.search(r"approv\w*.*plan|plan.*approv", low)
        if m and "plan_approval" in tool_names:
            req_id = self._find_plan_request_id(messages)
            if req_id:
                return MockResponse(
                    content=[
                        TextBlock(f"Approving plan {req_id}."),
                        ToolUseBlock("plan_approval", {
                            "request_id": req_id, "approve": True,
                            "feedback": "Looks good, proceed.",
                        }),
                    ],
                    stop_reason="tool_use",
                )

        # plan rejection
        m = re.search(r"reject.*plan|plan.*reject", low)
        if m and "plan_approval" in tool_names:
            req_id = self._find_plan_request_id(messages)
            if req_id:
                return MockResponse(
                    content=[
                        TextBlock(f"Rejecting plan {req_id}."),
                        ToolUseBlock("plan_approval", {
                            "request_id": req_id, "approve": False,
                            "feedback": "Too risky, please revise.",
                        }),
                    ],
                    stop_reason="tool_use",
                )

        # read inbox
        if "inbox" in low:
            return MockResponse(
                content=[ToolUseBlock("read_inbox", {})],
                stop_reason="tool_use",
            )

        # default: text reply
        return MockResponse(
            content=[TextBlock(f"[mock-lead] I received: {text[:120]}")],
        )

    # -------------------------------------------------------------- teammate
    def _teammate_response(self, text, messages, tool_names):
        low = text.lower()

        # Received shutdown_request → approve
        if "shutdown" in low or "shut down" in low:
            req_id = self._extract_field(text, "request_id")
            if req_id and "shutdown_response" in tool_names:
                return MockResponse(
                    content=[
                        TextBlock("Acknowledged. Shutting down gracefully."),
                        ToolUseBlock("shutdown_response", {
                            "request_id": req_id,
                            "approve": True,
                            "reason": "Work complete, ready to shut down.",
                        }),
                    ],
                    stop_reason="tool_use",
                )

        # Received plan approval/rejection response
        if "plan_approval_response" in low or "approve" in low:
            return MockResponse(
                content=[TextBlock("[mock-teammate] Plan response received. Proceeding.")],
            )

        # First message → submit a plan
        if "plan_approval" in tool_names and len(messages) <= 2:
            return MockResponse(
                content=[
                    TextBlock("I'll submit my plan for review."),
                    ToolUseBlock("plan_approval", {
                        "plan": f"Plan: 1) Analyze the task. 2) Implement changes. 3) Run tests. Task: {text[:80]}",
                    }),
                ],
                stop_reason="tool_use",
            )

        # default
        return MockResponse(
            content=[TextBlock(f"[mock-teammate] Working on: {text[:80]}")],
        )

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _last_user_text(messages):
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                # tool_result list or mixed content
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "tool_result":
                            return item.get("content", "")
                        if item.get("type") == "text":
                            return item.get("text", "")
                    if isinstance(item, str):
                        return item
        return ""

    @staticmethod
    def _extract_field(text, field_name):
        """Extract a JSON field value from text like '\"request_id\": \"abc\"'."""
        m = re.search(rf'"{field_name}"\s*:\s*"([^"]+)"', text)
        if m:
            return m.group(1)
        m = re.search(rf"'{field_name}'\s*:\s*'([^']+)'", text)
        if m:
            return m.group(1)
        return None

    @staticmethod
    def _extract_name_from_context(text, messages):
        """Try to find a teammate name from recent messages."""
        for msg in reversed(messages):
            content = msg.get("content", "")
            if isinstance(content, str):
                m = re.search(r"Spawned '(\w+)'|Spawning (\w+)", content)
                if m:
                    return m.group(1) or m.group(2)
            if isinstance(content, list):
                for block in content:
                    if hasattr(block, "text"):
                        m = re.search(r"Spawned '(\w+)'|Spawning (\w+)", block.text)
                        if m:
                            return m.group(1) or m.group(2)
        return "teammate"

    @staticmethod
    def _find_plan_request_id(messages):
        """Scan messages for a plan request_id."""
        for msg in reversed(messages):
            content = msg.get("content", "")
            text = ""
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        text += item.get("content", "") + " "
                    elif hasattr(item, "text"):
                        text += item.text + " "
            m = re.search(r"request_id[=:]\s*['\"]?(\w+)", text)
            if m:
                return m.group(1)
        return None


class MockAnthropic:
    """Drop-in replacement for Anthropic() client (offline mock)."""

    def __init__(self, **kw):
        self.messages = MockMessages()
