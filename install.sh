#!/usr/bin/env bash
set -e
python3 -m pip install -e .
echo ""
echo "RouteZero installed! Try:"
echo "  routezero audit ."
echo "  routezero route \"Classify this ticket sentiment\""
echo "  routezero start"
