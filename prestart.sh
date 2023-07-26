#! /usr/bin/env bash

# Let the DB start
python3 /home/CDG-NORD/florian-a/fastapi-base/backend_pre_start.py

# Run migrations
# PYTHONPATH=. alembic upgrade head

# python3 /home/CDG-NORD/florian-a/fastapi-base/src/db/init_db.py

# Create initial data in DB
python3 /home/CDG-NORD/florian-a/fastapi-base/initial_data.py