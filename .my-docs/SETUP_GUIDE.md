# Mem0 本地调试环境搭建指南

> macOS + PGVector（Docker）+ DeepSeek/阿里云百炼 的开发环境配置

---

## 目录

1. [环境概览](#1-环境概览)
2. [第一步：PostgreSQL + pgvector 安装](#2-第一步postgresql--pgvector-安装)
3. [第二步：Hatch 环境搭建](#3-第二步hatch-环境搭建)
4. [第三步：配置环境变量](#4-第三步配置环境变量)
5. [第四步：验证 SDK 功能](#5-第四步验证-sdk-功能)
6. [第五步：Server REST API 调试](#6-第五步server-rest-api-调试)
7. [故障排查](#7-故障排查)

---

## 1. 环境概览

| 组件 | 用途 | 当前状态 |
|------|------|---------|
| Python 3.11 | SDK / Server 运行环境 | 已安装 ✓ |
| Homebrew | macOS 包管理器 | 已安装 ✓ |
| Hatch | Python 项目管理（**强制使用**） | 已安装 ✓ |
| Docker + pgvector/pgvector:pg16 | 向量数据库 | 已安装 ✓ |
| DeepSeek API Key | LLM 记忆提取 | 已配置 ✓ |
| 阿里云百炼 API Key | Embedding 向量化 | 已配置 ✓ |

> **本项目使用 Hatch 管理所有 Python 环境，不要手动创建 venv。**

---

## 2. 第一步：PostgreSQL + pgvector 安装

### 方案 A：Docker（推荐，已在用）

```bash
docker run -d \
  --name mem0-pgvector \
  -e POSTGRES_USER=mem0 \
  -e POSTGRES_PASSWORD=mem0pass \
  -e POSTGRES_DB=mem0_dev \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

Server 还需要一个独立的应用数据库（存用户/API Key/配置）：

```bash
# 在 mem0_dev 容器里创建 mem0_app 库
docker exec -it mem0-pgvector psql -U mem0 -d mem0_dev \
  -c "CREATE DATABASE mem0_app;"
```

### 方案 B：Homebrew（本地原生）

```bash
brew install postgresql@16
brew services start postgresql@16

# 创建向量数据库
createdb mem0_dev
psql mem0_dev -c "CREATE EXTENSION IF NOT EXISTS vector;"
psql mem0_dev -c "CREATE USER mem0 WITH PASSWORD 'mem0pass';"
psql mem0_dev -c "GRANT ALL PRIVILEGES ON DATABASE mem0_dev TO mem0;"
psql mem0_dev -c "GRANT ALL ON SCHEMA public TO mem0;"

# 创建应用数据库
createdb mem0_app
psql mem0_app -c "CREATE USER mem0 WITH PASSWORD 'mem0pass';" 2>/dev/null || true
psql mem0_app -c "GRANT ALL PRIVILEGES ON DATABASE mem0_app TO mem0;"
```

---

## 3. 第二步：Hatch 环境搭建

### 3.1 可用的 Hatch 环境

```bash
cd /Users/Edison/mem0

# 查看所有环境
hatch env show
```

| 环境名 | Python | 用途 |
|--------|--------|------|
| `dev_py_3_11` | 3.11 | SDK 开发、跑测试 |
| `dev_py_3_12` | 3.12 | SDK 开发 (备选) |
| `server_py_3_11` | 3.11 | Server REST API (含 server 额外依赖) |

### 3.2 创建并进入环境

```bash
# SDK 开发环境
hatch env create dev_py_3_11
hatch shell dev_py_3_11       # 进入 shell，所有依赖已装好

# Server 开发环境
hatch env create server_py_3_11
hatch shell server_py_3_11
```

> `hatch shell` 之后你在虚拟环境中，所有 pip 包都在隔离环境里，不污染系统 Python。

### 3.3 运行测试

```bash
hatch run dev_py_3_11:test                            # 跑全部测试
hatch run dev_py_3_11:test -- tests/vector_stores/  # 只跑向量库测试
hatch run dev_py_3_11:lint                            # ruff 检查
```

---

## 4. 第三步：配置环境变量

根目录 `/Users/Edison/mem0/.env` 是唯一的配置文件，`load_dotenv()` 会从 `server/` 目录向上查找并自动加载它。确认以下关键变量都已存在：

```ini
# ---- LLM（DeepSeek，兼容 OpenAI 格式）----
OPENAI_API_KEY=sk-xxx               # 实际传给 DeepSeek，server 读 OPENAI_API_KEY
MEM0_DEFAULT_LLM_MODEL=deepseek-v4-flash
MEM0_DEFAULT_LLM_BASE_URL=https://api.deepseek.com

# ---- Embedder（阿里云百炼）----
MEM0_DEFAULT_EMBEDDER_MODEL=text-embedding-v3
MEM0_DEFAULT_EMBEDDER_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MEM0_DEFAULT_EMBEDDER_API_KEY=sk-xxx   # DashScope key 与 LLM key 不同，单独设置

# ---- PostgreSQL（向量记忆库）----
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=mem0_dev
POSTGRES_USER=mem0
POSTGRES_PASSWORD=mem0pass
POSTGRES_COLLECTION_NAME=mem0_memories

# ---- Server 专用 ----
AUTH_DISABLED=true                  # 本地开发关闭鉴权
JWT_SECRET=dev-secret-do-not-use-in-production
ADMIN_API_KEY=mem0-admin-dev-key-32characters
HISTORY_DB_PATH=/Users/Edison/mem0/server/data/history.db   # ← 必须！见下方说明

# ---- 关闭遥测 ----
MEM0_TELEMETRY=false
```

### 为什么需要 `HISTORY_DB_PATH`？

Server 用 SQLite 存储每条记忆的变更历史（ADD / UPDATE / DELETE 操作日志），供 `GET /memories/{id}/history` 接口使用。默认路径 `/app/history/history.db` 是 Docker 容器内路径，macOS 本地运行时该目录不存在且 `/app` 只读，**不设置该变量会导致启动即崩溃**。

**首次配置后需创建目录（只需一次）：**

```bash
mkdir -p /Users/Edison/mem0/server/data
```

### 为什么需要 `MEM0_DEFAULT_EMBEDDER_API_KEY`？

Server 默认用 `OPENAI_API_KEY` 同时作为 LLM 和 Embedder 的密钥。但本项目 LLM 用 DeepSeek、Embedding 用阿里云百炼，两者 API Key 不同，需要单独指定 Embedder 的 key。

> `server/main.py` 已修改为读取 `MEM0_DEFAULT_LLM_BASE_URL`、`MEM0_DEFAULT_EMBEDDER_BASE_URL`、`MEM0_DEFAULT_EMBEDDER_API_KEY` 这三个变量，否则自定义 base_url 无法生效。

---

## 5. 第四步：验证 SDK 功能

用 hatch 环境跑测试脚本：

```bash
cd /Users/Edison/mem0
hatch shell dev_py_3_11
python debug/run_test.py
```

脚本示例（`debug/run_test.py`）：

```python
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mem0 import Memory
from mem0.configs.base import MemoryConfig

config = MemoryConfig(
    vector_store={
        "provider": "pgvector",
        "config": {
            "dbname": os.environ.get("PGDATABASE", "mem0_dev"),
            "collection_name": "mem0_memories",
            "embedding_model_dims": 1024,           # text-embedding-v3
            "user": os.environ.get("PGUSER", "mem0"),
            "password": os.environ.get("PGPASSWORD", "mem0pass"),
            "host": "localhost",
            "port": 5432,
        },
    },
    llm={
        "provider": "openai",
        "config": {
            "model": os.environ.get("DEEPSEEK_MODEL"),
            "api_key": os.environ.get("DEEPSEEK_API_KEY"),
            "base_url": os.environ.get("DEEPSEEK_API_BASE"),
        },
    },
    embedder={
        "provider": "openai",
        "config": {
            "model": os.environ.get("DASHSCOPE_EMBEDDING_MODEL"),
            "api_key": os.environ.get("DASHSCOPE_API_KEY"),
            "base_url": os.environ.get("DASHSCOPE_EMBEDDING_BASE_URL"),
        },
    },
    version="v1.1",
)

m = Memory(config)
print("✅ Memory 初始化成功")

r = m.add("My name is Alice and I love Python programming.", user_id="alice")
print(f"✅ add: {r}")

r = m.search("programming languages", filters={"user_id": "alice"})
for item in r.get("results", []):
    print(f"  [{item['score']:.3f}] {item['memory']}")
print("✅ search done")
```

---

## 6. 第五步：Server REST API 调试

### 6.1 数据库说明

Server 使用两个独立数据库：

| 数据库 | 用途 | 说明 |
|--------|------|------|
| `mem0_dev` | 向量记忆（pgvector） | 由 mem0 SDK 自动建表 |
| `mem0_app` | 用户/API Key/设置/请求日志 | 由 Alembic 迁移建表 |

`mem0_app` 已在 [第一步](#2-第一步postgresql--pgvector-安装) 中随 Docker 一起创建。

### 6.2 建表（Alembic 迁移）

`server/alembic.ini` 默认连接字符串指向 Docker 内网地址 `postgres`，本地开发需改为 `localhost`：

```ini
# server/alembic.ini 第 4 行
sqlalchemy.url = postgresql+psycopg://mem0:mem0pass@localhost:5432/mem0_app
```

执行迁移建表：

```bash
cd /Users/Edison/mem0
hatch run server_py_3_11:python -m alembic -c server/alembic.ini upgrade head
```

验证建表成功：

```bash
docker exec mem0-pgvector psql -U mem0 -d mem0_app -c "\dt"
```

应输出：`users`、`api_keys`、`refresh_token_jtis`、`request_logs`、`settings`

### 6.3 PyCharm 调试配置

`main.py` 是 FastAPI 模块，**不能直接作为 Script 运行**，必须通过 uvicorn 启动。

打开 **Run → Edit Configurations** → `+` → **Python**，填写：

| 字段 | 值 |
|------|-----|
| **Name** | `Mem0 Server` |
| ~~Script path~~ → 改选 **Module name** | `uvicorn` |
| **Parameters** | `main:app --host 127.0.0.1 --port 8000` |
| **Working directory** | `/Users/Edison/mem0/server` |
| **Python interpreter** | Type 选 **Python**，路径见下方 |

**获取解释器路径：**

```bash
hatch env find server_py_3_11
# 输出类似：/Users/edy/Library/Application Support/hatch/env/virtual/mem0ai/b0weGXj4/server_py_3_11
```

解释器路径 = 上述输出 + `/bin/python`

> ⚠️ **不要加 `--reload`**：uvicorn 的 `--reload` 会 fork 子进程，导致 PyCharm 断点失效。
>
> ⚠️ **解释器 Type 选 Python，不要选 Hatch**：PyCharm 的 Hatch 集成运行 `hatch env show` 时不带项目路径会报错。

**Environment variables**：点右侧文件夹图标 → 加载 `/Users/Edison/mem0/.env`（需安装 EnvFile 插件），或手动把关键变量复制进去。

### 6.4 首次启动后：注册管理员账户（必须，只做一次）

即使设置了 `AUTH_DISABLED=true`，Server 的用户类接口（`/auth/me`、`/api-keys` 等）在数据库没有任何用户时仍会返回 `401`。**首次启动后必须注册一个管理员账户：**

```bash
curl -X POST http://127.0.0.1:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"name": "Admin", "email": "admin@local.dev", "password": "admin123"}'
```

注册成功后返回 `access_token` 和 `refresh_token`，之后所有接口（包括有 `require_auth` 依赖的）均正常响应。

> 该接口只有在数据库**没有任何用户**时才允许调用，之后会返回 `403`，防止重复注册。

**本地默认账户：**

| 字段 | 值 |
|------|-----|
| Email | `admin@local.dev` |
| Password | `admin123` |

### 6.5 启动验证

```bash
# 访问 Swagger 文档（正常返回 200 即成功）
curl -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/docs

# 测试记忆接口
curl -s http://127.0.0.1:8000/memories?user_id=test
# 预期：{"results": []}
```

### 6.6 命令行启动（非 PyCharm）

```bash
cd /Users/Edison/mem0
hatch run server_py_3_11:run
```

---

## 7. 故障排查

### 7.1 hatch 命令不存在

```bash
pipx ensurepath
source ~/.zshrc
```

### 7.2 Docker PostgreSQL 未启动

```bash
docker ps | grep mem0-pgvector
docker start mem0-pgvector    # 如果容器已停止
```

### 7.3 pgvector 扩展不存在

```bash
docker exec mem0-pgvector psql -U mem0 -d mem0_dev \
  -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### 7.4 `ModuleNotFoundError: No module named 'jose'`

PyCharm 的解释器没选对。Run Configuration → Python interpreter → Type 选 **Python**，路径指向 `hatch env find server_py_3_11` 输出的目录下的 `bin/python`。

### 7.5 Server 启动即退出（exit code 0）

Run Configuration 里用了 **Script path** 指向 `main.py`。`main.py` 是模块，直接运行会导入后退出。必须改为 **Module name: uvicorn**，参见 [6.3节](#63-pycharm-调试配置)。

### 7.6 `sqlite3.OperationalError: unable to open database file`

`.env` 缺少 `HISTORY_DB_PATH` 或目录不存在：

```bash
mkdir -p /Users/Edison/mem0/server/data
# 确认 .env 中有：
# HISTORY_DB_PATH=/Users/Edison/mem0/server/data/history.db
```

### 7.7 SQLAlchemy 连接错误

```text
connection to server at "postgres" failed
```

`server/alembic.ini` 主机名是 Docker 内网的 `postgres`，本地需改为 `localhost`，参见 [6.2节](#62-建表alembic-迁移)。

### 7.8 `AUTH_DISABLED=true` 但部分接口仍返回 401

数据库里没有任何用户。`require_auth` 即使在禁用鉴权时也需要从数据库找到一个真实用户。解决方法：调用 `/auth/register` 注册管理员，参见 [6.4节](#64-首次启动后注册管理员账户必须只做一次)。

### 7.9 DeepSeek / 阿里云百炼 API 调用失败

Server 的 `main.py` 已修改为读取 `MEM0_DEFAULT_LLM_BASE_URL`、`MEM0_DEFAULT_EMBEDDER_BASE_URL`、`MEM0_DEFAULT_EMBEDDER_API_KEY`。确认这三个变量在 `.env` 中已正确设置，否则请求会打到 OpenAI 官方地址而报认证错误。

### 7.10 零依赖快速体验（不用 PostgreSQL）

```python
from mem0 import Memory
m = Memory()   # 默认 Qdrant 本地模式，数据在 ~/.mem0/
```
