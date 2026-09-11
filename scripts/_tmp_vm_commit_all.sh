#!/usr/bin/env bash
cd /home/azureuser/muscle-arc || exit 1
BRANCH="vm-wip/20260911-100900"
git checkout "$BRANCH" 2>/dev/null || git checkout -b "$BRANCH"
git add -A
git rm -r --cached --quiet data/external 2>/dev/null
echo "STAGED=$(git diff --cached --name-only | wc -l)"
git -c user.name="rogii-gpu" -c user.email="rogii-gpu@vm.local" commit -m "chore(vm): snapshot uncommitted VM code/config/log experiment work" -m "Excludes large data/external benchmark binaries (gitignored)."
echo "COMMIT_EXIT=$?"
git log -1 --oneline
echo "=== PUSH ==="
git push -u origin "$BRANCH"
echo "PUSH_EXIT=$?"
