"""
pytest 公共配置与 fixtures
所有测试不依赖真实 LLM API、FAISS 索引或本地模型文件。
"""
import sys
import os
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# 确保项目根目录在 sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Mock chunk 数据
# ---------------------------------------------------------------------------
MOCK_CHUNKS = [
    {
        "chunk_id": "chunk_001",
        "book": "陈嘉庚传",
        "chapter": "第一章 早年",
        "text": "陈嘉庚于1874年出生于福建集美，少年随父赴南洋经商。",
        "char_count": 30,
    },
    {
        "chunk_id": "chunk_002",
        "book": "陈嘉庚传",
        "chapter": "第三章 兴学",
        "text": "1921年陈嘉庚创办厦门大学，亲自选定校址并承担常年经费。",
        "char_count": 30,
    },
    {
        "chunk_id": "chunk_003",
        "book": "南侨回忆录",
        "chapter": "教育",
        "text": "陈嘉庚认为教育为立国之本，先后创办集美小学、师范、水产、航海等学校。",
        "char_count": 38,
    },
    {
        "chunk_id": "chunk_004",
        "book": "陈嘉庚传",
        "chapter": "第五章 实业",
        "text": "陈嘉庚在新加坡经营橡胶种植与菠萝罐头加工，成为南洋华侨实业家。",
        "char_count": 34,
    },
    {
        "chunk_id": "chunk_005",
        "book": "南侨回忆录",
        "chapter": "抗日",
        "text": "抗日战争期间，陈嘉庚组织南洋华侨筹赈祖国难民总会，积极筹款救国。",
        "char_count": 36,
    },
]


@pytest.fixture
def mock_chunks():
    """返回一份 mock chunk 列表的深拷贝"""
    return json.loads(json.dumps(MOCK_CHUNKS))


# ---------------------------------------------------------------------------
# Mock LLM Client
# ---------------------------------------------------------------------------
class MockLLMClient:
    """可控的 mock LLM 客户端，按预设返回值响应"""

    def __init__(self):
        self.json_response = {}
        self.chat_response = "这是mock生成的答案。"
        self.stream_chunks = ["这是", "mock", "生成", "的", "答案", "。"]
        self.chat_calls = []
        self.extract_json_calls = []

    def chat(self, messages, **kwargs):
        self.chat_calls.append({"messages": messages, "kwargs": kwargs})
        return self.chat_response

    def extract_json(self, prompt, system_prompt=""):
        self.extract_json_calls.append({"prompt": prompt, "system_prompt": system_prompt})
        return self.json_response

    def chat_stream(self, messages, **kwargs):
        self.chat_calls.append({"messages": messages, "kwargs": kwargs})
        for chunk in self.stream_chunks:
            yield chunk


@pytest.fixture
def mock_llm():
    return MockLLMClient()


# ---------------------------------------------------------------------------
# Mock 检索结果
# ---------------------------------------------------------------------------
def make_retrieval_result(chunk_id, score, source="vector", **extra):
    """构造一条检索结果"""
    chunk = next((c for c in MOCK_CHUNKS if c["chunk_id"] == chunk_id), None)
    if chunk is None:
        chunk = {"chunk_id": chunk_id, "book": "mock", "chapter": "mock",
                 "text": f"mock text {chunk_id}", "char_count": 20}
    result = dict(chunk)
    result["score"] = score
    result["source"] = source
    result.update(extra)
    return result


@pytest.fixture
def dense_results():
    """模拟向量检索结果：A=chunk_001 rank1, B=chunk_002 rank2"""
    return [
        make_retrieval_result("chunk_001", 0.95, "vector"),
        make_retrieval_result("chunk_002", 0.88, "vector"),
        make_retrieval_result("chunk_003", 0.70, "vector"),
    ]


@pytest.fixture
def bm25_results():
    """模拟BM25检索结果：B=chunk_002 rank1, C=chunk_003 rank2"""
    return [
        make_retrieval_result("chunk_002", 12.5, "bm25"),
        make_retrieval_result("chunk_003", 10.2, "bm25"),
        make_retrieval_result("chunk_004", 7.8, "bm25"),
    ]
