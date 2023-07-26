#! /usr/bin/env bash
set -e

python3 /home/CDG-NORD/florian-a/fastapi-base/test_pre_start.py

bash ./scripts/test.sh "$@"