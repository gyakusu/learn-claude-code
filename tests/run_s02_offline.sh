#!/usr/bin/env bash
# Run s02_tool_use.py against the mock API server (no API key needed).
#
# Usage:
#   bash tests/run_s02_offline.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

MOCK_PORT=${MOCK_PORT:-5123}

# Start mock server in the background
python tests/mock_api.py --port "$MOCK_PORT" &
MOCK_PID=$!
trap "kill $MOCK_PID 2>/dev/null" EXIT

# Wait for mock server to be ready
for i in 1 2 3 4 5; do
    if curl -s "http://127.0.0.1:$MOCK_PORT/v1/messages" -X POST \
         -H "Content-Type: application/json" \
         -d '{"messages":[]}' >/dev/null 2>&1; then
        break
    fi
    sleep 0.3
done

echo ""
echo "=== s02 offline demo ==="
echo "Try these prompts:"
echo "  1. Read the file requirements.txt"
echo "  2. Create a file called greet.py with a greet(name) function"
echo "  3. Edit greet.py to add a docstring to the function"
echo "  4. Read greet.py to verify the edit worked"
echo ""

ANTHROPIC_BASE_URL="http://127.0.0.1:$MOCK_PORT" \
ANTHROPIC_API_KEY="mock-key" \
MODEL_ID="mock" \
python agents/s02_tool_use.py
