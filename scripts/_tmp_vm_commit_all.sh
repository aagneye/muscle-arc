#!/usr/bin/env bash
# Safely commit ALL VM working-tree changes onto a timestamped WIP branch and push.
# VM is 107 commits behind origin/main, so we DO NOT touch main. We branch off
# the current VM HEAD to preserve uncommitted experiment work without conflicts.
set -euo pipefail

cd /home/azureuser/muscle-arc

echo "=== Preview: untracked top-level dirs (sizes) ==="
git status --porcelain | grep '^??' | awk '{print $2}' | sed 's#/.*##' | sort -u \
  | while read -r d; do du -sh "$d" 2>/dev/null || true; done

BRANCH="vm-wip/$(date +%Y%m%d-%H%M%S)"
echo "=== Creating branch: ${BRANCH} (off current HEAD $(git rev-parse --short HEAD)) ==="
git checkout -b "${BRANCH}"

echo "=== Staging all (gitignore excludes data/artifacts/csv) ==="
git add -A

echo "=== Files staged: ==="
git diff --cached --name-only | sed -n '1,80p'
echo "... total staged: $(git diff --cached --name-only | wc -l)"

if git diff --cached --quiet; then
  echo "Nothing to commit."
else
  git -c user.name="${GIT_AUTHOR_NAME:-rogii-gpu}" \
      -c user.email="${GIT_AUTHOR_EMAIL:-rogii-gpu@vm.local}" \
      commit -m "chore(vm): snapshot uncommitted VM experiment work on ${BRANCH}" \
      -m "Auto-committed from rogii-gpu; VM was 107 commits behind origin/main. Preserved on WIP branch to avoid clobbering main."
fi

echo "=== Pushing ${BRANCH} to origin ==="
git push -u origin "${BRANCH}"

echo "=== DONE. VM HEAD now: ==="
git log -1 --format='%H %s'
