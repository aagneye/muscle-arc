#!/usr/bin/env bash
cd /home/azureuser/muscle-arc
echo "=== processes ==="
ps aux | grep -E 'vm_run_v11|eval_gate|audit_depth|calibrate_predict' | grep -v grep || echo none
echo "=== v11 log tail ==="
tail -40 logs/v11_run.log 2>/dev/null || echo no log
echo "=== decisions ==="
ls -la experiments/gate*.json submissions/submission_v11.csv experiments/v11_submit_decision.txt 2>/dev/null || true
grep -E 'GATE|DECISION|DONE|Traceback|Error|Wrote submissions|DEPTH_SCALE|SUBMIT' logs/v11_run.log 2>/dev/null | tail -40
