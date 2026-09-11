#!/usr/bin/env bash
cd /home/azureuser/muscle-arc
echo "=== processes ==="
ps aux | grep -E 'vm_run_v10b|audit_depth|calibrate_predict' | grep -v grep || echo none
echo "=== v10b log tail ==="
tail -60 logs/v10b_scale_run.log 2>/dev/null || echo no log
echo "=== DECISION lines ==="
grep -E 'DECISION|DEPTH_SCALE|Scale assignment|Sample residual|Temporal|GATE_B|Wrote submissions|BEST|FL_std|DONE|Traceback|Error' logs/v10b_scale_run.log logs/depth_scale_audit_v10b.log logs/infer_v10b.log 2>/dev/null | tail -40
ls -la submissions/submission_v10b.csv experiments/depth_scale_table.csv 2>/dev/null || true
