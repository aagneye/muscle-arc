#!/usr/bin/env bash
cd /home/azureuser/muscle-arc
echo "=== processes ==="
ps aux | grep -E 'vm_run_v10|audit_depth|calibrate_predict|tesseract' | grep -v grep || echo none
echo "=== v10 log ==="
tail -50 logs/v10_scale_run.log 2>/dev/null || echo no log
echo "=== depth audit ==="
tail -30 logs/depth_scale_audit.log 2>/dev/null || echo no audit yet
echo "=== infer ==="
tail -20 logs/infer_v10.log 2>/dev/null || echo no infer yet
ls -la experiments/depth_scale_table.csv submissions/submission_v10.csv 2>/dev/null || true
