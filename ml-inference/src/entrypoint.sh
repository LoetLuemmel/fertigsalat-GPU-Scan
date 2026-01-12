#!/bin/bash
# Entrypoint script for ml-inference container
# Starts scanner server if SCANNER_MODE=true, otherwise runs task watcher

if [ "$SCANNER_MODE" = "true" ]; then
    echo "Starting Scanner Server..."
    exec python3 -u /app/src/scanner_server.py
else
    echo "Starting Task Watcher..."
    exec python3 -u /app/src/main.py
fi
