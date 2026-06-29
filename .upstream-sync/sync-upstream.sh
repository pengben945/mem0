#!/usr/bin/env bash
# sync-upstream.sh — Fetch upstream mem0ai/mem0 main and merge into current branch
set -euo pipefail

REPO_DIR="/Users/Edison/mem0"
SYNC_DIR="${REPO_DIR}/.upstream-sync"
MERGE_LOG="${SYNC_DIR}/merge-history.md"

cd "$REPO_DIR"

# Ensure upstream remote exists
if ! git remote get-url upstream &>/dev/null; then
  echo "➕ Adding upstream remote..."
  git remote add upstream https://github.com/mem0ai/mem0.git
fi

echo "🔄 Fetching upstream (mem0ai/mem0)..."
git fetch upstream main 2>&1

# Get commits that will be merged
NEW_COMMITS=$(git log HEAD..upstream/main --oneline --no-merges 2>/dev/null || true)

if [ -z "$NEW_COMMITS" ]; then
  echo "✅ Already up to date with upstream/main. No merge needed."
  exit 0
fi

echo "📋 New commits to be merged:"
echo "$NEW_COMMITS"

COMMIT_COUNT=$(echo "$NEW_COMMITS" | grep -c . || true)

# Collect commit details (subject + hash, max 50)
COMMIT_DETAILS=$(git log HEAD..upstream/main --format="- %s (%h)" --no-merges | head -50)

# Do the merge
echo "🔀 Merging upstream/main..."
git merge upstream/main --no-edit -m "chore: sync upstream mem0ai/mem0 main $(date '+%Y-%m-%d %H:%M:%S')" 2>&1
MERGE_EXIT=$?

MERGE_TIME=$(date '+%Y-%m-%d %H:%M:%S %Z')
MERGE_DATE=$(date '+%Y-%m-%d')

if [ $MERGE_EXIT -ne 0 ]; then
  STATUS="❌ FAILED (冲突需手动处理)"
  CONFLICT_FILES=$(git diff --name-only --diff-filter=U 2>/dev/null | sed 's/^/  - /' || true)
  echo "❌ Merge failed. Conflicts may need manual resolution."
  git merge --abort 2>/dev/null || true
else
  STATUS="✅ SUCCESS"
  CONFLICT_FILES=""
  echo "✅ Merge succeeded."
fi

# Initialize log file if not exists
if [ ! -f "$MERGE_LOG" ]; then
  cat > "$MERGE_LOG" << 'EOF'
# Upstream 同步历史记录

本文件记录每次从上游开源仓库 [mem0ai/mem0](https://github.com/mem0ai/mem0) 合并代码的历史。

> **说明**：原始提交信息见"提交列表"，中文摘要由 AI Agent 在合并后自动补充。

---
EOF
fi

# Append raw entry (AI agent will later fill in the Chinese summary)
cat >> "$MERGE_LOG" << ENTRY

## ${MERGE_DATE} — ${MERGE_TIME}

**状态**: ${STATUS}
**合并提交数**: ${COMMIT_COUNT} 个提交

### 📝 中文摘要

> ⏳ 待 AI Agent 生成...

### 📋 提交列表（原文）

${COMMIT_DETAILS}
${CONFLICT_FILES:+
### ⚠️ 冲突文件

${CONFLICT_FILES}}
---
ENTRY

echo "📝 Raw log written: ${MERGE_LOG}"
echo "SYNC_DONE:${MERGE_DATE}:${COMMIT_COUNT}:${STATUS}"
