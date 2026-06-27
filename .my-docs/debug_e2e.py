#!/usr/bin/env python3
"""
Mem0 端到端调试脚本
LLM: DeepSeek (deepseek-chat)
Embedding: 阿里云百炼 (OpenAI-compatible API)
Vector Store: PGVector (Docker)
"""

import os
import sys

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass

from mem0 import Memory
from mem0.configs.base import MemoryConfig


def build_config():
    """根据环境变量构建 MemoryConfig"""

    # -- PGVector --
    pg_user = os.environ.get("PGUSER", "mem0")
    pg_password = os.environ.get("PGPASSWORD", "mem0pass")
    pg_host = os.environ.get("PGHOST", "localhost")
    pg_port = int(os.environ.get("PGPORT", "5432"))
    pg_database = os.environ.get("PGDATABASE", "mem0_dev")

    # -- LLM: DeepSeek --
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    deepseek_base = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com")
    deepseek_model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

    if not deepseek_key:
        print("⚠️  DEEPSEEK_API_KEY 未设置！请在 .env 文件中填入你的 DeepSeek API Key")
        print("   然后执行: source .env")
        sys.exit(1)

    # -- Embedding: 阿里云百炼 (OpenAI-compatible) --
    dashscope_key = os.environ.get("DASHSCOPE_API_KEY")
    dashscope_base = os.environ.get("DASHSCOPE_EMBEDDING_BASE_URL",
                                     "https://dashscope.aliyuncs.com/compatible-mode/v1")
    dashscope_model = os.environ.get("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v3")

    if not dashscope_key:
        print("⚠️  DASHSCOPE_API_KEY 未设置！请在 .env 文件中填入你的阿里云百炼 API Key")
        print("   然后执行: source .env")
        sys.exit(1)

    # Build config
    config = MemoryConfig(
        vector_store={
            "provider": "pgvector",
            "config": {
                "dbname": pg_database,
                "collection_name": "mem0_memories",
                "embedding_model_dims": 1024,  # text-embedding-v3 默认 1024 维
                "user": pg_user,
                "password": pg_password,
                "host": pg_host,
                "port": pg_port,
                "hnsw": True,
            },
        },
        llm={
            "provider": "deepseek",
            "config": {
                "model": deepseek_model,
                "api_key": deepseek_key,
                "deepseek_base_url": deepseek_base,
                "temperature": 0.1,
                "max_tokens": 2000,
            },
        },
        embedder={
            "provider": "openai",  # 阿里云百炼兼容 OpenAI 格式
            "config": {
                "model": dashscope_model,
                "api_key": dashscope_key,
                "openai_base_url": dashscope_base,
                "embedding_dims": 1024,
            },
        },
        version="v1.1",
    )
    return config


def main():
    print("=" * 60)
    print("Mem0 端到端验证 — PGVector + DeepSeek + 阿里云百炼")
    print("=" * 60)

    # ---- 0. Build config ----
    print("\n[0] 初始化 Memory ...")
    config = build_config()
    m = Memory(config)
    print("✅ Memory 初始化成功")

    # ---- 1. Reset (clean start) ----
    print("\n[1] 重置数据库 ...")
    m.reset()
    print("✅ 数据库已重置")

    # ---- 2. Add memories ----
    print("\n[2] 添加记忆 ...")
    messages = [
        {"role": "user", "content": "Hi, my name is Alice. I'm a software engineer working at Google in San Francisco."},
        {"role": "assistant", "content": "Nice to meet you Alice! What kind of projects do you work on?"},
        {"role": "user", "content": "I work on the search infrastructure team. I love hiking and my favorite trail is in Yosemite. I also have a dog named Max."},
        {"role": "assistant", "content": "That sounds amazing! How long have you been at Google?"},
        {"role": "user", "content": "I've been at Google for about 4 years now. Before that I worked at a startup called DataStack."},
    ]
    result = m.add(messages, user_id="alice", agent_id="chat_agent")
    extracted = result.get("results", [])
    print(f"✅ 从 {len(messages)} 条消息中提取了 {len(extracted)} 条记忆:")
    for mem in extracted:
        print(f"   [{mem.get('event')}] {mem.get('memory', '')[:80]}{'...' if len(mem.get('memory', '')) > 80 else ''}")

    # ---- 3. Search ----
    print("\n[3] 搜索记忆 ...")
    queries = [
        "What does Alice do for work?",
        "where does Alice like hiking?",
        "What is Alice's dog's name?",
    ]
    for q in queries:
        print(f"   查询: \"{q}\"")
        sr = m.search(q, filters={"user_id": "alice"}, top_k=3, explain=True)
        for item in sr.get("results", []):
            print(f"     [{item.get('score', 0):.3f}] {item.get('memory', '')}")
            if "score_details" in item:
                sd = item["score_details"]
                print(f"       semantic={sd.get('semantic_score', 0):.3f}, "
                      f"bm25={sd.get('bm25_score', 0):.3f}, "
                      f"entity_boost={sd.get('entity_boost', 0):.3f}")
        print()

    # ---- 4. Get all ----
    print("[4] 列出所有记忆 ...")
    ga = m.get_all(filters={"user_id": "alice"})
    print(f"✅ 共 {len(ga.get('results', []))} 条记忆")

    # ---- 5. Entity Store verification ----
    print("\n[5] 检查 Entity Store ...")
    try:
        es = m.entity_store
        listed = es.list(filters={"user_id": "alice"}, top_k=100)
        # listed is list[list[OutputData]]
        entities = []
        for row in listed[0] if listed else []:
            p = getattr(row, "payload", {}) or {}
            name = p.get("data", "?")
            mems = p.get("linked_memory_ids", [])
            entities.append((name, len(mems)))
        print(f"✅ 发现 {len(entities)} 个实体:")
        for name, count in entities:
            print(f"   {name}: 关联 {count} 条记忆")
    except Exception as e:
        print(f"⚠️  Entity Store 检查失败 (可能 spaCy 模型未下载): {e}")

    print("\n" + "=" * 60)
    print("✅ 端到端验证完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
