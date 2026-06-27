# Mem0 源码架构深度解析

> 一份面向 Agent 记忆层开发者的快速入门文档

---

## 目录

1. [项目概览](#1-项目概览)
2. [仓库结构](#2-仓库结构)
3. [核心架构](#3-核心架构)
4. [Memory 记忆管道深度剖析](#4-memory-记忆管道深度剖析)
5. [五大组件详解](#5-五大组件详解)
6. [配置系统](#6-配置系统)
7. [混合检索与评分算法](#7-混合检索与评分算法)
8. [TypeScript SDK](#8-typescript-sdk)
9. [平台客户端 API](#9-平台客户端-api)
10. [关键数据流图](#10-关键数据流图)
11. [学习路径建议](#11-学习路径建议)

---

## 1. 项目概览

**Mem0**（"mem-zero"）是一个为 AI Agent 和助手设计的智能记忆层。它提供持久化、个性化的记忆能力，支持两种使用模式：

| 模式 | Python | TypeScript | 适用场景 |
|------|--------|------------|---------|
| **托管平台** | `MemoryClient` / `AsyncMemoryClient` | `MemoryClient` | 开箱即用，API Key 接入 |
| **自托管 OSS** | `Memory` / `AsyncMemory` | `Memory` (from `mem0ai/oss`) | 私有部署，完全可控 |

### 核心理念

Mem0 的记忆层不是简单的"存消息→查消息"，而是：

1. **从对话中抽取结构化事实**（LLM 驱动的记忆提取）
2. **混合检索**（语义搜索 + 关键词 BM25 + 实体增强）
3. **记忆去重与更新**（增量式记忆管理，而非覆盖式）
4. **实体关联**（通过实体图谱实现跨记忆的连接）

---

## 2. 仓库结构

```
mem0/
├── mem0/                    # Python SDK 核心
│   ├── memory/              # 记忆引擎主逻辑 (Memory/MemoryClient)
│   │   ├── main.py          # ★ 核心文件 (~3742 行): Memory + AsyncMemory
│   │   ├── base.py          # MemoryBase 抽象基类
│   │   ├── storage.py       # SQLite 管理 (历史记录 + 消息缓存)
│   │   ├── utils.py         # 消息解析、格式转换等工具函数
│   │   ├── setup.py         # 配置文件读写
│   │   ├── telemetry.py     # PostHog 遥测
│   │   └── notices.py       # 产品内提示
│   ├── client/              # 托管平台客户端
│   │   └── main.py          # MemoryClient + AsyncMemoryClient (~1813 行)
│   ├── llms/                # 19 个 LLM 提供商实现
│   ├── embeddings/          # 15 个 Embedding 提供商实现
│   ├── vector_stores/       # 27 个向量库提供商实现
│   ├── graphs/              # (不存在于 OSS 版本)
│   ├── reranker/            # 5 个重排序器实现
│   ├── configs/             # Pydantic 配置模型 + Prompts 模板
│   ├── utils/
│   │   ├── factory.py       # ★ 工厂模式: 统一创建所有组件
│   │   ├── scoring.py       # ★ 混合检索评分算法
│   │   └── entity_extraction.py  # spaCy 实体抽取
│   └── exceptions.py        # 结构化异常体系
│
├── mem0-ts/                 # TypeScript SDK
│   └── src/
│       ├── client/          # 托管客户端 (axios-based)
│       ├── oss/src/         # 自托管 Memory 类 (~2039 行)
│       └── common/          # 共享异常体系
│
├── server/                  # FastAPI 自托管服务 (Docker)
├── openmemory/              # 完整自托管平台 (API + Next.js UI)
├── cli/python/              # Python CLI (Typer)
├── cli/node/                # Node CLI (Commander)
├── integrations/            # 编辑器/框架集成
├── skills/                  # Claude Code 技能定义
├── docs/                    # Mintlify 文档站点
└── tests/                   # pytest 测试
```

---

## 3. 核心架构

### 3.1 Provider 抽象模式

Mem0 的核心设计模式是 **Provider 抽象**——每个组件领域都遵循相同的三层结构：

```
┌─────────────────────────────────────────────┐
│              抽象基类 (ABC)                   │  ← 定义契约
├─────────────────────────────────────────────┤
│              工厂类 (Factory)                 │  ← 按名称创建实例
├─────────────────────────────────────────────┤
│  具体实现 1  │  具体实现 2  │ ... │ 具体实现 N │  ← 提供商实现
└─────────────────────────────────────────────┘
```

**五大组件领域：**

| 领域 | 基类 | 工厂 | 实现数 | 职责 |
|------|------|------|--------|------|
| LLMs | `LLMBase` | `LlmFactory` | 18 | 从对话中提取记忆事实 |
| Embeddings | `EmbeddingBase` | `EmbedderFactory` | 11 | 文本向量化 |
| Vector Stores | `VectorStoreBase` | `VectorStoreFactory` | 23 | 向量存储与检索 |
| Reranker | `BaseReranker` | `RerankerFactory` | 5 | 搜索结果重排序 |
| Graph Stores | — | — | 0 | (OSS 版本不含图存储) |

### 3.2 两种使用模式的架构差异

```
【自托管 OSS 模式】                    【托管平台模式】
┌──────────────┐                      ┌──────────────┐
│   Memory     │                      │ MemoryClient │
│  (main.py)   │                      │  (client/)   │
│              │                      │              │
│  ┌────────┐  │                      │   HTTP/JSON  │
│  │  LLM   │  │  ← 本地调用           │     ↓        │
│  │Embedder│  │                      │  ┌────────┐  │
│  │VectorSt│  │                      │  │  Mem0  │  │
│  │Reranker│  │                      │  │ Cloud  │  │
│  │ SQLite │  │                      │  └────────┘  │
│  └────────┘  │                      │              │
└──────────────┘                      └──────────────┘
```

### 3.3 组件装配流程（`Memory.__init__`）

```python
class Memory(MemoryBase):
    def __init__(self, config: MemoryConfig):
        # 1. 通过工厂创建 Embedding 模型
        self.embedding_model = EmbedderFactory.create(
            config.embedder.provider, config.embedder.config
        )

        # 2. 通过工厂创建向量存储
        self.vector_store = VectorStoreFactory.create(
            config.vector_store.provider, config.vector_store.config
        )

        # 3. 通过工厂创建 LLM
        self.llm = LlmFactory.create(
            config.llm.provider, config.llm.config
        )

        # 4. 创建本地 SQLite 数据库
        self.db = SQLiteManager(config.history_db_path)

        # 5. 可选: 创建重排序器
        self.reranker = RerankerFactory.create(...) if config.reranker else None

        # 6. 懒加载: 实体存储 (独立的向量存储)
        self._entity_store = None  # @property 懒加载
```

---

## 4. Memory 记忆管道深度剖析

> 这是整个项目最核心的部分，理解 `add()` 和 `search()` 就理解了 Mem0 的 80%。

### 4.1 `add()` — V3 增量记忆提取管道

**方法签名：**
```python
def add(
    self,
    messages,                    # str | dict | list[dict]
    *,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    metadata: Optional[Dict] = None,
    infer: bool = True,          # True=LLM提取, False=原始存入
    memory_type: Optional[str] = None,  # "procedural_memory"
    prompt: Optional[str] = None,
) -> dict:  # {"results": [...]}
```

**V3 管道（`infer=True`）——9 个阶段：**

```
输入消息
   │
   ▼
┌──────────────────────────────────────────────────────────┐
│ Phase 0: 上下文收集                                        │
│  • 构建 session_scope (user_id + agent_id + run_id)       │
│  • 从 SQLite 获取最近 10 条消息 (作为上下文窗口)             │
│  • 解析消息为统一格式                                       │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│ Phase 1: 已有记忆检索                                      │
│  • Embed 解析后的消息                                       │
│  • 向量搜索 top_k=10 条相关已有记忆                          │
│  • 映射 UUID → 整数索引 (防 LLM 幻觉)                       │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│ Phase 2: LLM 记忆提取 (单次调用)                            │
│  • 组装 ADDITIVE_EXTRACTION_PROMPT (系统提示词)             │
│  • 输入: 新消息 + 最近记忆 + 已有记忆 + 观察日期             │
│  • LLM 输出 JSON: {"memory": [{"id": "0", "text": "...",  │
│      "attributed_to": "user", "linked_memory_ids": [...]}]}│
│  • 只提取 ADD 事件，不做 UPDATE/DELETE                       │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│ Phase 3: 批量 Embedding                                    │
│  • 将所有提取的 memory text 一次性 embed_batch()             │
│  • 失败时回退到逐条 embed                                    │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│ Phase 4-5: CPU 处理 + 哈希去重                              │
│  • 词形还原 (lemmatize) 用于 BM25 索引                      │
│  • 构建 payload 字典 (data, hash, created_at, ...)         │
│  • MD5 哈希去重: 排除与已有记忆重复或批次内重复的记忆           │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│ Phase 6: 批量持久化                                        │
│  • vector_store.insert(vectors, payloads, ids) — 单次调用  │
│  • SQLite batch_add_history() 批量写入历史记录              │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│ Phase 7: 批量实体链接                                      │
│  • extract_entities_batch() — spaCy 批量抽取实体           │
│  • 全局去重 → 批量 embed → 搜索已有实体                    │
│  • 精确文本匹配 OR 语义相似度 ≥ 0.95 → 更新已存在实体       │
│  • 否则 → 新建实体记录 (linked_memory_ids)                  │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│ Phase 8: 保存消息 + 返回结果                                │
│  • 将原始消息存入 SQLite messages 表                       │
│  • 自动淘汰旧消息 (每 session 最多保留 10 条)               │
│  • 返回: [{"id": uuid, "memory": text, "event": "ADD"}]   │
└──────────────────────────────────────────────────────────┘
```

**管道变体：**

| 条件 | 行为 |
|------|------|
| `infer=False` | 跳过 LLM 提取，每条消息直接 embedding + 存入 |
| `memory_type="procedural_memory"` | 调用 LLM 生成过程性记忆摘要 |
| 纯系统消息 | 跳过不处理 |

### 4.2 `search()` — 混合检索管道

**方法签名：**
```python
def search(
    self,
    query: str,
    *,
    top_k: int = 20,            # 返回结果数
    filters: Dict[str, Any],     # 过滤条件 (含 user_id/agent_id/run_id)
    threshold: float = 0.1,      # 语义分数最低阈值
    rerank: bool = False,        # 是否启用重排序
    explain: bool = False,       # 是否返回 score_details
) -> dict:  # {"results": [...]}
```

**检索流程——9 个步骤：**

```
查询字符串: "What does Alice like about hiking?"
   │
   ▼
┌──────────────────────────────────────────────────────────┐
│ Step 1: 查询预处理                                         │
│  • 词形还原 (lemmatize) → 用于 BM25                        │
│  • extract_entities(query) → ["Alice", "hiking"]          │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│ Step 2: Embed 查询                                        │
│  • embedding_model.embed(query, memory_action="search")   │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│ Step 3: 语义搜索 (Vector Search)                           │
│  • 过检索 (over-fetch): max(limit*4, 60) 个候选            │
│  • 余弦相似度 / 欧氏距离 → 归一化为 [0,1] 分数              │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│ Step 4: 关键词搜索 (BM25) — 可选                            │
│  • 仅当 vector_store 支持 keyword_search() 时启用          │
│  • 对 payload 中的 text_lemmatized 字段做 BM25 匹配        │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│ Step 5: BM25 分数归一化                                    │
│  • Sigmoid 归一化到 [0,1]                                  │
│  • 自适应参数: 短查询→低中点+高陡度 (更难拿高分)              │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│ Step 6: 实体增强 (Entity Boosting)                         │
│  • 对查询中的每个实体: embed → 搜索实体库→ 提取关联记忆 ID   │
│  • 相似度阈值 ≥ 0.5                                        │
│  • boost = ENTITY_BOOST_WEIGHT / (1 + 0.001*(n-1)²)      │
│    (ENTITY_BOOST_WEIGHT = 0.5, 记忆越多 boost 衰减越快)     │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│ Step 7: 构建候选集 + 过滤                                   │
│  • 过滤过期记忆 (除非 show_expired=True)                    │
│  • 应用元数据过滤器 (AND/OR/NOT, eq/ne/gt/lt/in/contains)  │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│ Step 8: 混合评分 (score_and_rank)                          │
│  • threshold 门控: 语义分数 < threshold → 直接丢弃          │
│  • max_possible = 1.0(sem) + 1.0(bm25) + 0.5(entity)     │
│  • combined = (sem + bm25 + entity) / max_possible        │
│  • 按 combined 降序排列                                     │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│ Step 9: 格式化结果                                         │
│  • 返回 MemoryItem 列表 (id, memory, score, metadata...)   │
│  • explain=True 时附加 score_details 字段                  │
└──────────────────────────────────────────────────────────┘
```

### 4.3 高级元数据过滤

Mem0 支持类 MongoDB 的高级过滤语法：

```python
filters = {
    "user_id": "alice",
    "AND": [
        {"category": {"eq": "personal"}},
        {"priority": {"gte": 3}}
    ],
    "OR": [
        {"tag": {"in": ["urgent", "important"]}},
        {"source": {"contains": "meeting"}}
    ],
    "NOT": [
        {"status": {"eq": "archived"}}
    ],
    "name": "*",  # 通配符: 存在即匹配
}
```

处理函数 `_process_metadata_filters()` 将其转换为向量库可识别的格式。

---

## 5. 五大组件详解

### 5.1 LLM 基类

**文件：** `mem0/llms/base.py`

```python
class LLMBase(ABC):
    def __init__(self, config: BaseLlmConfig):
        # 自动处理 dict / Pydantic model / None 三种配置格式
        # 验证 model 字段存在
        # 检测推理模型 (o1/o3/gpt-5 系列)

    def _is_reasoning_model(self, model: str) -> bool:
        # 检测 o1, o3, gpt-5 系列
        # 推理模型不支持 temperature/top_p，用 reasoning_effort 替代

    @abstractmethod
    def generate_response(
        self, messages: List[Dict], tools=None, tool_choice="auto"
    ) -> str | Dict:
        """生成 LLM 响应，子类必须实现"""
```

18 个实现中最重要的：
- **OpenAI** (`openai.py`) — 标准 OpenAI API
- **OpenAI Structured** (`openai_structured.py`) — 使用 `response_format` 实现结构化输出
- **Anthropic** (`anthropic.py`) — Claude 系列
- **Groq** (`groq.py`) — 快速推理
- **Ollama** (`ollama.py`) — 本地模型

### 5.2 Embedding 基类

**文件：** `mem0/embeddings/base.py`

```python
class EmbeddingBase(ABC):
    def __init__(self, config: BaseEmbedderConfig):
        # embedding_dims, model 等配置

    @abstractmethod
    def embed(self, text, memory_action: Literal["add","search","update"]) -> list:
        """单条文本向量化"""

    def embed_batch(self, texts, memory_action="add") -> list[list]:
        """批量向量化 (默认逐条循环，子类可覆盖用原生批量 API)"""
```

关键点：`memory_action` 参数允许不同操作使用不同的 embedding 策略（如 Jina AI 的 task-specific embeddings）。

### 5.3 Vector Store 基类

**文件：** `mem0/vector_stores/base.py`

```python
class VectorStoreBase(ABC):
    # 11 个抽象方法:
    create_col(name, vector_size, distance)  # 创建集合
    insert(vectors, payloads, ids)           # 插入向量
    search(query, vectors, top_k, filters)   # 语义搜索 (返回相似度分数)
    delete(vector_id)                        # 删除
    update(vector_id, vector, payload)        # 更新
    get(vector_id)                           # 获取
    list_cols()                              # 列出集合
    delete_col()                             # 删除集合
    col_info()                               # 集合信息
    list(filters, top_k)                     # 列出记忆
    reset()                                  # 重建集合

    # 可选方法:
    keyword_search(query, top_k, filters)    # BM25 全文搜索 (默认返回 None)
    search_batch(queries, vectors, top_k)    # 批量搜索
```

最重要的实现：
- **Qdrant** — 推荐默认，支持 embedded 模式（零依赖启动）
- **Chroma** — 轻量级，适合原型开发
- **pgvector** — PostgreSQL 原生向量扩展
- **Pinecone / Weaviate / Milvus** — 生产级向量数据库

### 5.4 Reranker 基类

**文件：** `mem0/reranker/base.py`

```python
class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, documents: List[Dict], top_k: int = None):
        """对搜索结果重排序，返回添加 rerank_score 的文档列表"""
```

5 个实现：Cohere、HuggingFace、LLM-based、Sentence Transformer、Zero Entropy。

### 5.5 实体抽取

**文件：** `mem0/utils/entity_extraction.py`

基于 **spaCy** 的 NER + 规则匹配，抽取 4 类实体：

| 类型 | 来源 | 示例 |
|------|------|------|
| PROPER | spaCy NER (PERSON, ORG, GPE, LOC...) + 大写模式 | "Alice", "Google", "New York" |
| QUOTED | 双引号/单引号内容 | "machine learning" |
| TOPIC | 带特定修饰词的名词短语 | "deep reinforcement learning" |
| IDENTIFIER | 点分隔的技术标识符 | "mem0.vector_stores.qdrant" |

实体存储在一个 **独立的向量库**（与记忆库同提供商，collection 名加 `_entities` 后缀），每条实体记录包含：
- `data`: 实体文本
- `linked_memory_ids`: 关联的记忆 ID 列表
- `entity_type`: PROPER / QUOTED / TOPIC / IDENTIFIER

---

## 6. 配置系统

### 6.1 配置层次

```
MemoryConfig (顶层)
├── vector_store: VectorStoreConfig
│   └── config: 具体向量库的配置 (如 QdrantConfig)
├── llm: LlmConfig
│   └── config: 具体 LLM 的配置 (如 OpenAIConfig)
├── embedder: EmbedderConfig
│   └── config: 具体 Embedder 的配置
├── reranker: RerankerConfig (可选)
│   └── config: 具体 Reranker 的配置
├── history_db_path: str = "~/.mem0/history.db"
├── version: str = "v1.1"
└── custom_instructions: Optional[str] = None
```

### 6.2 Prompt 模板系统

**文件：** `mem0/configs/prompts.py` (~1063 行)

最重要的 Prompt：

| Prompt | 用途 |
|--------|------|
| `ADDITIVE_EXTRACTION_PROMPT` | V3 add() 的核心系统提示词 (~475 行) |
| `USER_MEMORY_EXTRACTION_PROMPT` | 仅从用户消息提取事实 |
| `AGENT_MEMORY_EXTRACTION_PROMPT` | 仅从 Agent 消息提取事实 |
| `DEFAULT_UPDATE_MEMORY_PROMPT` | 记忆更新决策 (ADD/UPDATE/DELETE/NONE) |
| `PROCEDURAL_MEMORY_SYSTEM_PROMPT` | 过程性记忆摘要 |
| `FACT_RETRIEVAL_PROMPT` | 旧版事实提取 (已被 V3 替代) |

**V3 提取提示词的核心设计思想：**

1. **角色定义**：你是"增量记忆管理器"，只提取值得持久化的新信息
2. **输入结构**：新消息 + 摘要 + 最近提取的记忆 + 已有记忆 + 最后 k 条消息
3. **提取标准**：
   - **上下文丰富**：记忆必须自包含，脱离对话也能理解
   - **时序扎根**：包含相对时间锚点（"上个月"、"2024年3月"）
   - **数值精确**：保留具体数字而非模糊表述
   - **意义保持**：不改变原意
4. **完整性规则**：不编造、不推断、不回显提取
5. **记忆链接**：通过 `linked_memory_ids` 关联相关记忆
6. **10 个详细示例** + 完整自检清单

---

## 7. 混合检索与评分算法

**文件：** `mem0/utils/scoring.py`

### 7.1 评分公式

```
combined_score = min(
    (semantic_score + bm25_score + entity_boost) / max_possible,
    1.0
)
```

其中 `max_possible` 根据可用信号动态调整：

| 可用信号 | max_possible |
|---------|-------------|
| 仅语义搜索 | 1.0 |
| 语义 + BM25 | 2.0 |
| 语义 + 实体增强 | 1.5 |
| 全部三种 | 2.5 |

### 7.2 关键阈值

```
语义搜索最低阈值: threshold (默认 0.1)
  → 语义分数低于此值的候选项直接丢弃，即使 BM25/实体分数很高

实体匹配阈值: 0.5 (搜索实体库的相似度阈值)
实体链接阈值: 0.95 (判断是否更新已有实体)
实体增强权重上限: 0.5 (ENTITY_BOOST_WEIGHT)
```

### 7.3 BM25 归一化

```python
def get_bm25_params(query):
    # 自适应逻辑
    if num_terms <= 1:
        midpoint, steepness = (0.0, 12.0)   # 单字查询: 极难得高分
    elif num_terms <= 3:
        midpoint, steepness = (2.0, 2.0)    # 短查询: 较难
    elif num_terms <= 6:
        midpoint, steepness = (5.0, 1.0)    # 中等长度
    else:
        midpoint, steepness = (20.0, 0.3)   # 长查询: 较容易

def normalize_bm25(raw_score, midpoint, steepness):
    # Sigmoid 归一化
    return 1 / (1 + exp(-steepness * (raw_score - midpoint)))
```

### 7.4 实体增强衰减

```python
ENTITY_BOOST_WEIGHT = 0.5

# 实体关联的记忆越多，每个记忆获得的 boost 越少
entity_boost = ENTITY_BOOST_WEIGHT / (1 + 0.001 * (n - 1)²)

# 示例:
# n=1  → boost = 0.500
# n=5  → boost = 0.493
# n=20 → boost = 0.324
# n=100 → boost = 0.046
```

---

## 8. TypeScript SDK

### 8.1 结构对比

| 概念 | Python | TypeScript |
|------|--------|------------|
| 托管客户端类 | `MemoryClient` | `MemoryClient` (default export from `mem0ai`) |
| OSS 记忆类 | `Memory` | `Memory` (from `mem0ai/oss`) |
| HTTP 库 | `httpx` | `axios` |
| 配置验证 | Pydantic | Zod |
| OSS LLM | `mem0/llms/` | `src/oss/src/llms/` |
| OSS Embeddings | `mem0/embeddings/` | `src/oss/src/embeddings/` |
| OSS Vector Stores | `mem0/vector_stores/` | `src/oss/src/vector_stores/` |
| 包管理器 | Hatch | pnpm |

### 8.2 TypeScript OSS Memory 的关键差异

1. **自动初始化**：`_autoInitialize()` 方法在首次操作时自动检测 embedding 维度、创建集合
2. **Zod 验证**：用 Zod Schema 替代 Pydantic 做配置验证
3. **dummy history manager**：当历史记录功能关闭时使用 `DummyHistoryManager`
4. **大小写转换**：客户端在 `camelCase` 和 `snake_case` 之间自动转换（但保护 `metadata` 等用户定义的键）

---

## 9. 平台客户端 API

### 9.1 核心 API 速查

| 方法 | 端点 | 用途 |
|------|------|------|
| `add()` | POST `/v3/memories/add/` | 添加新记忆 |
| `search()` | POST `/v3/memories/search/` | 搜索记忆 |
| `get(memory_id)` | GET `/v1/memories/{id}/` | 获取单条记忆 |
| `get_all()` | POST `/v3/memories/` | 列出所有记忆（支持分页） |
| `update(memory_id)` | PUT `/v1/memories/{id}/` | 更新记忆 |
| `delete(memory_id)` | DELETE `/v1/memories/{id}/` | 删除记忆 |
| `delete_all()` | DELETE `/v1/memories/` | 删除所有记忆 |
| `history(memory_id)` | GET `/v1/memories/{id}/history/` | 记忆变更历史 |
| `users()` | GET `/v1/entities/` | 列出实体 |
| `delete_users()` | DELETE `/v2/entities/` | 删除实体 |
| `feedback()` | POST `/v1/feedback/` | 提交反馈 |

### 9.2 实体参数规范化（v3）

所有 `user_id`、`agent_id`、`run_id` 必须通过 `filters` 字典传递：

```python
# ✅ 正确
client.add(messages, user_id="alice")
client.search("query", filters={"user_id": "alice"})

# ❌ 错误 (v3 会报错)
client.search("query", user_id="alice")
```

---

## 10. 关键数据流图

### 10.1 记忆写入全流程

```
用户消息
    │
    ▼
┌──────────────────┐
│   Memory.add()   │
└────────┬─────────┘
         │
    ┌────▼────┐
    │ infer?  │── False ──→ 逐条 embed → vector_store.insert()
    └────┬────┘
         │ True
         ▼
┌─────────────────────┐
│ 获取上下文 (SQLite)   │  ← 最近 10 条消息
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ 检索已有记忆 (向量库) │  ← top_k=10 相关记忆
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ LLM 提取新事实       │  ← ADDITIVE_EXTRACTION_PROMPT
│ → {"memory": [...]} │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ 批量 Embed + 去重    │  ← embed_batch() + MD5 哈希
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ 持久化               │
│ ├ vector_store       │  ← 批量 insert
│ ├ SQLite history     │  ← batch_add_history
│ └ SQLite messages    │  ← 保存原始消息
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ 实体链接             │
│ ├ extract_entities   │  ← spaCy
│ ├ embed → search     │  ← 实体库语义匹配
│ └ upsert entities    │  ← 更新 linked_memory_ids
└─────────┬───────────┘
          │
          ▼
   返回结果列表
```

### 10.2 混合检索全流程

```
查询: "What does Alice like about hiking?"
    │
    ▼
┌──────────────────────┐
│ 查询预处理             │
│ ├ lemmatize           │  → ["what", "alice", "like", "about", "hiking"]
│ └ extract_entities    │  → ["Alice", "hiking"]
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ embed(query)          │  → [0.023, -0.451, ...]  (向量)
└──────────┬───────────┘
           │
           ├───────────────────────────────┐
           ▼                               ▼
┌─────────────────┐               ┌─────────────────┐
│ 语义搜索          │               │ 关键词搜索 (BM25) │
│ vector_store     │               │ keyword_search() │
│ .search()        │               │ (可选)           │
└────────┬────────┘               └────────┬────────┘
         │                                 │
         │ semantic_scores                  │ raw_bm25_scores
         │ [0.89, 0.76, ...]                │ [12.3, 5.1, ...]
         │                                 │
         │                          ┌──────▼────────┐
         │                          │ sigmoid 归一化  │
         │                          │ → [0.91, 0.67] │
         │                          └──────┬────────┘
         │                                 │
         │                                 │ bm25_scores (归一化)
         ▼                                 ▼
┌──────────────────────┐          ┌──────────────┐
│ 实体增强              │          │              │
│ embed("Alice")       │          │              │
│ → 搜索实体库          │          │              │
│ → 获取 linked_mems   │          │              │
│ → 计算 boost 权重     │          │              │
└──────────┬───────────┘          │              │
           │                      │              │
           │ entity_boosts         │              │
           │ {mem_1: 0.5, ...}     │              │
           ▼                      ▼              │
┌───────────────────────────────────────────────┘
│
▼
┌──────────────────────────────────────────────┐
│ score_and_rank()                              │
│                                               │
│ max_possible = 1.0 + 1.0 + 0.5 = 2.5         │
│                                               │
│ mem_1: (0.89 + 0.91 + 0.50) / 2.5 = 0.920    │ ★
│ mem_3: (0.76 + 0.67 + 0.00) / 2.5 = 0.572    │
│ mem_5: (0.65 + 0.80 + 0.30) / 2.5 = 0.700    │
│ ...                                           │
└───────────────────┬──────────────────────────┘
                    │
                    ▼
┌──────────────────────┐
│ 排序 + Top-K          │
│ rerank (可选)         │
└──────────────────────┘
                    │
                    ▼
            [MemoryItem, ...]
```

---

## 11. 学习路径建议

### 第一层：快速上手（1 天）

1. 安装 `pip install mem0ai`，用 5 行代码跑通 add → search
2. 阅读 `CLAUDE.md` 了解仓库结构
3. 理解两种模式的差异：`Memory` vs `MemoryClient`

### 第二层：核心管道（2-3 天）

按以下顺序深入源码：

```
1. mem0/configs/base.py         ← 先看配置入口
2. mem0/memory/base.py          ← 理解抽象契约
3. mem0/memory/main.py          ← ★ 核心: add() 和 search() 方法
4. mem0/memory/storage.py       ← SQLite 存储层
5. mem0/configs/prompts.py      ← Prompt 工程
6. mem0/utils/scoring.py        ← 混合评分算法
7. mem0/utils/entity_extraction.py ← 实体抽取
8. mem0/utils/factory.py        ← 工厂模式
```

### 第三层：组件实现（3-5 天）

根据需要选择一个领域深入：
- **LLM 层**：读 `mem0/llms/base.py` → `openai.py` → `anthropic.py`
- **Embedding 层**：读 `mem0/embeddings/base.py` → `openai.py` → `fastembed.py`
- **向量库层**：读 `mem0/vector_stores/base.py` → `qdrant.py` → `chroma.py`
- **Reranker 层**：读 `mem0/reranker/base.py` → `cohere_reranker.py`

### 第四层：TypeScript 对照（2-3 天）

1. `mem0-ts/src/oss/src/memory/index.ts` — 对照 Python 的 Memory 类
2. `mem0-ts/src/client/mem0.ts` — 对照 Python 的 MemoryClient
3. 理解两种语言在异步处理、配置验证上的差异

### 第五层：生产部署（按需）

1. `server/` — FastAPI + PostgreSQL/pgvector + Neo4j
2. `openmemory/` — 完整自托管平台 (API + Web UI)
3. `integrations/` — 编辑器/框架集成模式

---

## 附录：关键设计决策与面试要点

### 为什么是"增量提取"而非"全量重写"？

Mem0 V3 的 `ADDITIVE_EXTRACTION_PROMPT` 只做 ADD，不做 UPDATE/DELETE。
- **优点**：避免每次对话都重写全部记忆，减少 LLM 调用成本，保持记忆稳定性
- **代价**：需要依赖外部去重（哈希 + 语义相似度）来避免重复记忆

### 为什么混合检索要用"门控阈值"而非直接加权？

```python
if semantic_score < threshold:  # 默认 0.1
    discard  # 直接丢弃
```

设计理由：BM25 和实体增强是"补充信号"，不能替代语义相关性。如果一条记忆与查询在语义上完全无关（分数 < 0.1），即使碰巧包含了查询词，也不应该被返回。

### 为什么实体存储用独立向量库？

实体（Entity Store）存储的是"命名实体 → 关联记忆 ID"的映射，与主记忆库分离：
- **隔离性**：实体维度的 CRUD 不影响记忆主流程
- **可扩展**：可以独立调优实体匹配阈值、embedding 模型
- **避免污染**：实体 embedding 空间与记忆 embedding 空间不同

### 为什么 BM25 参数要自适应查询长度？

```python
# 单字查询: midpoint=0, steepness=12  → 极难触发高分
# 长查询:   midpoint=20, steepness=0.3 → 容易得高分
```

设计理由：短查询如 "Alice" 在 BM25 中会匹配大量文档，高 BM25 分数不应简单等同于高相关性。而长查询如 "what did Alice say about the project deadline last Friday" 中，多个词的匹配更能说明相关性。

---

> **版本信息**：本文档基于 mem0 仓库 `main` 分支 (2026-06-27) 分析生成。
> **许可证**：Apache-2.0
