# Upstream 同步历史记录

本文件记录每次从上游开源仓库 [mem0ai/mem0](https://github.com/mem0ai/mem0) 合并代码的历史。

> **说明**：原始提交信息见"提交列表"，中文摘要由 AI Agent 在合并后自动补充。

---

## 2026-06-29 — 2026-06-29 10:33:27 CST

**状态**: ✅ SUCCESS
**合并提交数**: 6 个提交

### 📝 中文摘要

本次合并共包含 6 个提交，均为文档修复与版本更新，无功能性破坏变更：

- **版本升级**：Python SDK 升至 `2.0.10`，TypeScript SDK 升至 `3.0.12`，并更新了 Changelog
- **Cookbook 文档修复**：修正了多个 v3 版本示例中过时的过滤器写法（`filters`）、响应数据结构，以及失效的代码片段；同步修正了 `update()` 方法在同伴（companions）与基础示例中的用法
- **组件文档修复**：更正了 LLM 与 Embedding 模型 ID、修正了 TypeScript 支持列表的错误描述
- **平台文档对齐**：将 Platform 文档与 v3 SDK 的实际行为保持一致，消除了文档与 SDK 之间的描述偏差
- **向量数据库文档修复**：修正了各向量存储配置项的默认值与导入方式

### 📋 提交列表（原文）

- chore: update changelog, bump SDK versions to Python 2.0.10 and TypeScript 3.0.12 (#5927) (8d6b7c1d)
- docs(cookbooks): fix v3 filters, response shapes & dead snippet in cookbooks (#5841) (b44ce4dc)
- docs(components): fix LLM & embedder model IDs and TS support lists (#5838) (e9c930c4)
- docs(platform): align Platform docs with v3 SDK behavior (#5849) (4b39d01c)
- docs(cookbooks): fix v3 filters & update() in companions/essentials (#5844) (49061718)
- docs(vectordbs): correct vector store config defaults & imports (#5843) (7e7682a0)

---
