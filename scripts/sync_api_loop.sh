#!/bin/bash
# Run sync 4 times per minute (every ~15s) 
while true; do
    /home/openclaw/distillation/scripts/sync_api_state.sh
    sleep 15
done
