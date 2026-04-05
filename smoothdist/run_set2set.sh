#!/bin/bash
# Delete nohup.out if it exists
source ../../debug-py314/bin/activate
rm -f nohup.out
# Run the Python script in the background with nohup
nohup python -u set2set.py &
