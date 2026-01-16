#!/bin/bash
# Delete nohup.out if it exists
rm -f nohup.out
# Run the Python script in the background with nohup
nohup python -u planner_stats.py &
