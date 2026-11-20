#!/usr/bin/env bash
set -euo pipefail

echo "Ollama models:"
ollama list
echo
echo "TreeMindDB backend:"
curl --fail --silent --show-error http://127.0.0.1:8000/api/v1/models/status
echo
curl --fail --silent --show-error http://127.0.0.1:8000/api/v1/chat-memory/status
echo
