# Mem0 开源版（v3 OSS）vs 企业版（Platform）能力缺口详解

> 本文基于代码实测 + 官方迁移文档整理。重点聚焦"智能记忆层的核心能力差异"，即记忆的存、取、管三个维度，不讨论 SaaS 运维/多租户等平台层能力。
>
> 仓库基准：`pengben945/mem0` = fork 自上游 v3（version 2.0.9）

---

## 一、先理解 v3 的大背景

### v2 → v3 是一次大重构，整体效果是增强

官方基准测试数据：

| 基准 | v2 OSS | v3 OSS | 提升 |
|---|---|---|---|
| LoCoMo | 71.4 | 91.6 | **+20 分** |
| LongMemEval | 67.8 | 93.4 | **+26 分** |
| 抽取延迟 | baseline | ~减半 | 快约 2x |

v3 做了三件核心事：
1. **抽取从两次 LLM → 单次 ADD-only**：原来要"提取候选 + 决定ADD/UPDATE/DELETE"两次调用，v3 合并成一次，模型专注理解输入，效果更好、延迟更低。
2. **检索从纯语义 → 三路混合**：语义向量 + BM25 关键词 + 实体匹配，融合打分。
3. **图记忆从外部图库 → 内置实体链接**：这是最大争议点，下面详细讲。

---

## 二、图记忆能力的变迁（最重要的变化）

### v2 的图：外部图库，可遍历关系

```
用户说："我在阿里巴巴工作，我的上司是张三"

v2 写入 Neo4j：
  (用户) --[WORKS_AT]--> (阿里巴巴)
  (用户) --[REPORTS_TO]--> (张三)
  (张三) --[WORKS_AT]--> (阿里巴巴)

检索 "用户的上司在哪里工作？"
  → 图遍历：用户 → 张三 → 阿里巴巴
  → 返回可遍历的 relations 结构
```

外部依赖：Neo4j / Memgraph / Kuzu / Apache AGE，代码约 4000 行。

### v3 OSS 的图：内置实体链接，关系隐式

```
用户说："我在阿里巴巴工作，我的上司是张三"

v3 写入向量库：
  主集合：{user_id}_memories
    - "用户在阿里巴巴工作"
    - "用户的上司是张三"
  实体集合：{user_id}_memories_entities
    - 实体: 阿里巴巴
    - 实体: 张三

检索 "用户的上司在哪里工作？"
  → query 中抽出实体：张三、阿里巴巴
  → 含这些实体的记忆得 entity_boost 加权
  → 分数高的排前面（而不是图遍历）
  → 不返回 relations 字段
```

### 关键区别一句话

> v2：你可以"走图"——从 A 找到 B 再找到 C（多跳关系推理）
> v3：你只能"找相关"——含相同实体的记忆排名更高（单跳实体加权）

### 对 CastrelAI 的影响

| 场景 | v3 OSS 是否够用 |
|---|---|
| "用户喜欢什么？" | ✅ 语义检索直接命中 |
| "用户和张三是什么关系？" | ✅ 实体加权能召回相关记忆 |
| "用户老板的老板是谁？"（多跳） | ⚠️ 无法图遍历，需多次 search 拼接 |
| 知识图谱式推理 | ❌ 需自接图库 |

**结论：如果 Agent 不需要多跳关系推理，v3 内置实体链接完全够用。**

### 自建图库的可行性（如需要）

v2 图库代码结构清晰，与向量管线完全独立，回合成本低：

| 文件 | 大小 | 内容 |
|---|---|---|
| `mem0/graphs/configs.py` | 2.5KB | Neo4j/Memgraph 配置类 |
| `mem0/graphs/tools.py` | 15KB | LLM tool call 定义（实体+关系提取） |
| `mem0/graphs/utils.py` | 5.7KB | 图提取/更新/删除 Prompts |
| `mem0/memory/graph_memory.py` | 21KB | MemoryGraph 主类（add/search/delete） |

接入方式（Parallel Sidecar，不改 v3 内核）：

```
Memory.add(text)
  ├─→ v3 向量管线（不动）
  └─→ MemoryGraph.add(text)   ← 旁路图写入

Memory.search(query)
  ├─→ v3 混合检索结果
  └─→ MemoryGraph.search(query) ← 旁路图查询
       ↓
    合并去重后返回
```

改动量：约 50-60 行 hook 代码 + 复制 4 个文件。**估计 1~3 天工作量。**
图库推荐：Neo4j（v2 原生支持）或 Apache AGE（基于 PostgreSQL，若已有 PG 则零增运维成本）。

---

## 三、Advanced Retrieval（高级检索）

### 是什么

在基础语义检索之上，提供 **Reranking（二次精排）** 能力——初次召回后，再用更深层的语义模型对结果重新排序，把最相关的结果排到最前面。

### OSS 支持情况

✅ **OSS 完全支持**，与企业版无差距：

```python
results = m.search(
    query="用户的出行计划",
    filters={"user_id": "u1"},
    rerank=True,   # 启用二次精排
    top_k=10,
)
# reranker 可配置：Cohere、HuggingFace、SentenceTransformer、ZeroEntropy、LLM-based
```

配置 reranker 示例：
```python
config = {
    "reranker": {
        "provider": "cohere",
        "config": {"api_key": "...", "model": "rerank-english-v3.0", "top_n": 5}
    }
}
```

**延迟参考**：开启 reranker 额外增加约 150-200ms，适合对结果精度要求高、对延迟不敏感的场景。

---

## 四、Advanced Memory Operations（高级记忆操作）

企业版文档里的"Advanced Memory Operations"对应三个写入控制功能：

### 4.1 Direct Import（直接导入，跳过 LLM 推断）

**是什么**：`add()` 时设置 `infer=False`，跳过 LLM 抽取阶段，**直接把消息内容原样存入记忆库**，不经过事实提取和去重。

**适用场景**：你已经有结构化的记忆数据（如从旧系统迁移），不需要 LLM 再推断，直接批量写入。

```python
# infer=True（默认）：LLM 从对话里抽取事实后存入
# infer=False：直接存原始内容，跳过推断
messages = [{"role": "user", "content": "Alice 喜欢打羽毛球"}]
client.add(messages, user_id="alice", infer=False)
# ⚠️ 注意：跳过了去重检测，重复导入会产生重复记忆
```

**OSS 支持情况**：✅ OSS 完全支持，`infer` 参数两个版本一致。

---

### 4.2 Selective Memory（选择性记忆）

**是什么**：在 `add()` 调用时，通过在 `custom_instructions` 里指定规则，让 LLM **选择性地只记某些内容、忽略其他内容**。

**与 Custom Instructions 的关系**：Selective Memory 是 Custom Instructions 的一种应用方式——通过写精确的 instructions 控制什么记、什么不记。

```python
# 只记工作相关，忽略闲聊
config = {
    "custom_instructions": "只提取与用户工作、技能、项目相关的信息。忽略日常闲聊和情感表达。"
}
```

**OSS 支持情况**：✅ OSS 完全支持，通过 `custom_instructions` 配置实现。

---

### 4.3 Group Chat（群聊记忆）

**是什么**：多人对话场景下，Mem0 自动识别每个说话人，**把记忆分别归属到不同参与者**，而不是混在一起。

```python
# 群聊消息加 name 字段，Mem0 自动按人拆分记忆
messages = [
    {"role": "user", "name": "Alice", "content": "我们前端用 React 吧"},
    {"role": "user", "name": "Bob",   "content": "我更倾向 Vue.js"},
    {"role": "user", "name": "Charlie", "content": "Angular 企业支持更好"},
]
client.add(messages, run_id="meeting-001", infer=True)

# 结果：三条记忆分别归属 Alice / Bob / Charlie 的 user_id
```

检索时可精确到某个人：
```python
client.search("技术偏好", filters={"AND": [{"user_id": "alice"}, {"run_id": "meeting-001"}]})
```

**OSS 支持情况**：✅ **OSS 完全支持**，`name` 字段识别逻辑在核心抽取管线里，两个版本一致。

---

## 五、Contextual Memory Creation（上下文记忆创建）

### 是什么

每次 `add()` **只传当前这轮新消息**，Mem0 自动基于 `user_id` / `run_id` 管理历史上下文关联，不需要你手动拼接完整对话历史。

```python
# 第1轮
m.add([{"role": "user", "content": "我叫 Sarah，来自纽约"}], user_id="sarah")

# 第2轮：只传新消息，Mem0 自己知道 Sarah 来自纽约
m.add([{"role": "user", "content": "下个月我要去意大利"}], user_id="sarah")
```

`run_id` 控制上下文隔离粒度：
```
user_id="sarah"                        → 跨所有会话的长期记忆（偏好、个人信息）
user_id="sarah", run_id="trip-italy"  → 这次旅行规划专属上下文
user_id="sarah", run_id="work-q4"     → 工作项目上下文，与旅行完全隔离
```

**OSS 支持情况**：✅ **OSS 完全支持**，这不是一个新算法特性，而是 `user_id` / `run_id` 机制的使用方式，两版本完全一致。

---

## 六、Custom Instructions（自定义抽取指令）

### 是什么

用自然语言告诉 Mem0 的抽取 LLM：**哪些信息要记，哪些信息不要记**。本质上是注入到 `add()` 抽取 LLM 调用里的额外 Prompt。

```python
# 企业版：项目级配置
client.project.update(custom_instructions="""
只提取工作相关信息：技能、项目、职位、工作偏好。
排除：密码、财务数据、银行卡号、身份证号等敏感信息。
""")

# OSS：配置里直接设置，效果完全一样
config = {
    "custom_instructions": """
    只提取工作相关信息：技能、项目、职位、工作偏好。
    排除：密码、财务数据、银行卡号、身份证号等敏感信息。
    """
}
m = Memory.from_config(config)
```

**典型场景**：

| 应用类型 | 配置方向 |
|---|---|
| 工作 Agent | 只记技术偏好、项目上下文、工作目标 |
| 客服 Agent | 只记问题类型、产品偏好；排除支付信息 |
| 教育 Agent | 只记学习进度、薄弱点、学习风格 |
| 医疗 Agent | 只记症状、用药、健康目标；严格排除身份信息 |

**OSS 支持情况**：✅ **OSS 完全支持**，两版本完全对等，无差距。

---

## 七、Criteria Retrieval（自定义标准检索）

### Platform 的 Criteria Retrieval 是什么

在检索时叠加"业务相关性标准"，不只看语义距离，还可按时间、来源、类型等自定义维度决定"什么记忆在当前场景最有用"。

### v3 OSS 实际实现到什么程度

**直接看代码，v3 OSS 的 `search()` 已实现：**

#### 三路融合打分（源自 `mem0/utils/scoring.py`）

```python
combined = (semantic_score + bm25_score + entity_boost) / max_possible
# max_possible 自适应：
#   仅语义 → 1.0
#   语义+BM25 → 2.0
#   语义+BM25+实体 → 2.5
```

#### 完整的 metadata 条件运算符（源自 `mem0/memory/main.py`）

```python
# 精确匹配
filters={"user_id": "u1", "category": "work"}

# 比较运算
filters={"user_id": "u1", "score": {"gt": 0.8}}
filters={"user_id": "u1", "created_at": {"gte": "2024-06-01", "lte": "2024-12-31"}}

# 列表
filters={"user_id": "u1", "tag": {"in": ["work", "tech"]}}
filters={"user_id": "u1", "source": {"nin": ["deprecated"]}}

# 文本包含
filters={"user_id": "u1", "source": {"contains": "slack"}}
filters={"user_id": "u1", "content": {"icontains": "python"}}  # 不区分大小写

# 逻辑组合
filters={
    "user_id": "u1",
    "AND": [
        {"tag": "work"},
        {"created_at": {"gte": "2024-06-01"}}
    ]
}
filters={"user_id": "u1", "OR": [{"tag": "work"}, {"tag": "tech"}]}
filters={"user_id": "u1", "NOT": [{"tag": "archived"}]}
```

#### 打分透明（explain 模式）

```python
results = m.search("用户喜欢什么语言", filters={"user_id": "u1"}, explain=True)
# 每条结果附带：
{
  "memory": "用户喜欢用 Go 开发后端服务",
  "score": 0.84,
  "score_details": {
    "semantic_score": 0.82,   # 语义向量相似度
    "bm25_score": 0.45,       # 关键词匹配分
    "entity_boost": 0.30,     # 实体关联加权
    "final_score": 0.84
  }
}
```

#### 其他参数

```python
m.search(
    query="...",
    filters={"user_id": "u1"},
    top_k=20,          # 返回条数
    threshold=0.1,     # 语义分下限，低于此直接过滤
    rerank=True,       # 启用 Cohere/HuggingFace 等 reranker 二次精排
    show_expired=True, # 是否返回已过期记忆
)
```

### OSS 与 Platform 的真实差距（仅剩两点）

| 能力 | v3 OSS | Platform |
|---|---|---|
| metadata 多条件过滤（eq/ne/gt/in/AND/OR/NOT） | ✅ | ✅ |
| 时间范围过滤 | ✅（`created_at` filter） | ✅ |
| 三路混合打分 | ✅ | ✅ |
| reranker 二次精排 | ✅（可配置） | ✅ |
| explain 打分透明 | ✅ | ✅ |
| **打分权重可配置** | ❌ 写死（ENTITY_BOOST_WEIGHT=0.5） | ✅ |
| **时序感知检索**（reference_date 偏置） | ❌（OSS 调用直接报错） | ✅ |

**实际结论：Criteria Retrieval 的缺口极小。**
- 权重不可配置 → 改一行代码（`ENTITY_BOOST_WEIGHT = 0.5`），5 分钟。
- 时序感知 → 用 `created_at gte/lte` filter 手动实现，效果相近。

---

## 八、Memory Decay（记忆衰减）

### 不做 Decay 会有什么问题

Agent 记忆库只增不减，长期运行后：

```
记忆库（用户用了1年）：
  - "用户在用 Python"       （5年前）
  - "用户在用 JavaScript"   （3年前）
  - "用户在用 Go"           （6个月前）
  - "用户最近在学 Rust"      （上周）
```

检索"用户用什么语言"时，四条都可能命中，旧信息干扰当前判断。用得越久，噪声越多，召回越混乱。

### Decay 做什么

模拟人类记忆遗忘机制——长时间未被访问/强化的记忆，权重自动降低（不是删除，而是"退场"）：

```
衰减后：
  - "用户在用 Python"       （5年前）→ 权重 0.05，基本不召回
  - "用户在用 JavaScript"   （3年前）→ 权重 0.15
  - "用户在用 Go"           （6个月前）→ 权重 0.65
  - "用户最近在学 Rust"      （上周）  → 权重 1.0，优先召回
```

### 时序推理（Temporal Reasoning）是什么

配套 Decay 的推理能力：
- 知道"上周说的"比"去年说的"更可信
- 知道"用户已换工作"意味着旧的工作记忆应降权
- 能处理时间相关问题："用户去年在哪工作？"

### v3 OSS 现状

- ✅ 每条记忆有 `created_at` / `updated_at` 时间戳
- ✅ 有 `history()` 方法查看单条记忆的变更历史
- ✅ 有 `expiration_date` 字段（硬性过期，非软性衰减）
- ❌ 无自动衰减权重计算
- ❌ `reference_date` 参数在 OSS 直接抛 `ValueError`（官方故意保留企业差异）

### OSS 如何自建 Decay

**实现思路**（改动量中等，约 3~5 天）：

```python
# 方案：在 metadata 里存衰减权重，定时任务更新
# 检索时把 decay_weight 融入最终打分

def compute_decay(created_at, last_accessed_at, access_count):
    age_days = (now - created_at).days
    recency_days = (now - last_accessed_at).days
    # 指数衰减：越久没访问，权重越低
    decay = exp(-0.01 * recency_days) * (1 + log(1 + access_count) * 0.1)
    return min(max(decay, 0.05), 1.0)  # 最低保留 5% 不彻底消失
```

定时任务每天/每周批量更新所有记忆的 `decay_weight` 字段，`search()` 的 `score_and_rank` 融入该字段即可。

---

## 九、Feedback / Scoring（反馈打分闭环）

### 没有反馈会怎样

记忆系统"盲飞"——不知道召回的记忆有没有帮到用户，无法自我优化。

### Feedback 做什么

收集每次记忆被使用后的反馈信号，持续更新记忆质量分：

```
场景：
  Agent 召回："用户喜欢喝美式咖啡"→ 回答"您喜欢美式"
  用户：不对，我现在喝拿铁了

  反馈信号：NEGATIVE（记忆过时）
  系统响应：
    - 降低该记忆 score
    - 触发记忆更新（add 新事实）
    - 该记忆在后续检索中排名下降
```

反馈不一定要用户显式点赞/踩：
- 用户接受了 Agent 建议 → 隐式正反馈
- 用户纠正了 Agent → 隐式负反馈
- 用户忽略回答 → 轻微负反馈

### Scoring 是什么

每条记忆维护一个质量分，随反馈累积更新。高分记忆优先召回，低分记忆逐渐淡出。这是记忆系统从"静态存储"到"自适应学习"的关键。

### v3 OSS 现状

- ❌ 无内置反馈接口
- ❌ 无自动 scoring 机制
- ✅ metadata 可存任意字段，**可以自建反馈接口写入 `quality_score`**，再在 filter 里利用

### OSS 如何自建

```python
# 1. 新增反馈接口
POST /feedback
{
  "memory_id": "xxx",
  "feedback": "positive" | "negative",
  "reason": "记忆已过时"
}

# 2. 更新 metadata
memory.update(memory_id, metadata={"quality_score": new_score})

# 3. 检索时利用（filter 过滤低分记忆）
m.search("...", filters={
    "user_id": "u1",
    "quality_score": {"gt": 0.3}  # 过滤掉低质量记忆
})
```

---

## 十、能力汇总与优先级建议

### 完整能力对比

| 能力维度 | 具体功能 | v3 OSS | Platform | 差距性质 |
|---|---|---|---|---|
| **记忆抽取** | ADD-only 单次抽取 | ✅ | ✅ | 无差距 |
| | 智能去重/冲突消解 | ✅ | ✅ | 无差距 |
| | 多模态输入 | ✅ | ✅ | 无差距 |
| | Custom Instructions（自定义抽取指令） | ✅ | ✅ | 无差距 |
| | Direct Import（infer=False，跳过推断） | ✅ | ✅ | 无差距 |
| | Selective Memory（选择性记忆） | ✅ | ✅ | 无差距 |
| | Group Chat（群聊多人归属） | ✅ | ✅ | 无差距 |
| **写入管理** | Contextual Memory Creation（上下文自动关联） | ✅ | ✅ | 无差距 |
| | Custom Categories（LLM 自动归类打标） | ⚠️ 需手动打标 | ✅ | 小，可自建 |
| **检索** | 语义向量检索 | ✅ | ✅ | 无差距 |
| | BM25 关键词检索 | ✅ | ✅ | 无差距 |
| | 实体链接加权 | ✅（内置） | ✅ | 无差距 |
| | Advanced Retrieval（reranker 精排） | ✅ | ✅ | 无差距 |
| | metadata 多条件过滤 | ✅（完整） | ✅ | 无差距 |
| | Memory Filters v2（categories/keywords 字段） | ❌（无自动归类） | ✅ | 小，依赖自建分类 |
| | 打分权重可配置 | ❌（改1行代码） | ✅ | **极小，自己改** |
| | 时序感知检索（reference_date） | ❌ | ✅ | 小，可用 filter 近似 |
| **图记忆** | 实体关联加权 | ✅（内置） | ✅ | 无差距 |
| | **可遍历图关系/多跳** | ❌ | ✅ | **中等，可二开** |
| **生命周期** | 过期日期（硬性） | ✅ | ✅ | 无差距 |
| | **Memory Decay（软性衰减）** | ❌ | ✅ | **中等，可二开** |
| | **时序推理** | ❌ | ✅ | 中等，可近似 |
| **质量优化** | **Feedback/Scoring 闭环** | ❌ | ✅ | **中等，可二开** |

### 对 CastrelAI 的优先级建议

```
第一阶段（现在）：直接用 v3 OSS
  ✅ 抽取、混合检索、过滤全部够用
  ✅ 先跑通 Agent 记忆读写主流程，积累真实数据
  ⚡ 记得安装 spaCy [nlp] 扩展，否则退化为纯语义检索

     pip install "mem0ai[nlp]"
     python -m spacy download en_core_web_sm

第二阶段（业务增长后）：按需补齐
  P1：Memory Decay
      → 优先级最高，Agent 上线后随时间积累记忆噪声会快速暴露
      → 3~5 天自建，指数衰减公式成熟

  P2：图关系（如需多跳推理）
      → 从 v2 回合代码，Sidecar 方式接 Neo4j
      → 1~3 天，代码已有，按需接

  P3：Feedback 闭环
      → 有足够用户量后再做，需要交互数据积累
      → 自建接口 + metadata 写入，成本低但价值需数据支撑
```

---

## 十一、关键结论

1. **v3 大重构让 OSS 与企业版同源**，核心抽取/检索管线一致，基准分提升 20~26 分，延迟减半。

2. **检索能力几乎对等**：filter 运算符完整，三路融合打分，explain 透明，reranker 可配。Criteria Retrieval 的缺口只剩"权重可配置"和"时序感知"，成本极低。

3. **真正需要评估的缺口只有两个**：
   - **多跳图关系**：取决于 Agent 是否需要，需要就补，代码成本低。
   - **Memory Decay**：时间越长越有必要，中期必做。

4. **Feedback 闭环**属于锦上添花，等有用户规模再做。

5. **Apache-2.0 许可**：完全允许闭源二开、商用、不强制开源衍生代码。

