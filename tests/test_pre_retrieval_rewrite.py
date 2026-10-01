import pytest
from unittest.mock import MagicMock
from src.query_rewriter import PreRetrievalRewriter
from src.rag_pipeline import RAGPipeline


def test_pre_rewrite_fallback():
    """验证改写失败时安全回退 original_query"""
    mock_client = MagicMock()
    # 模拟大模型提取 JSON 失败
    mock_client.extract_json.side_effect = Exception("API Timeout")
    
    rewriter = PreRetrievalRewriter(client=mock_client)
    
    # 因为存在 history，会尝试改写，但最终由于异常必须安全回退
    history = [{"role": "user", "content": "他哪一年出生的"}, {"role": "assistant", "content": "陈嘉庚出生于1874年"}]
    rewritten = rewriter.rewrite("那他在哪里上学", history=history)
    
    assert rewritten == "那他在哪里上学", "改写异常时没有回退到原查询"


def test_pre_rewrite_empty_fallback():
    """验证大模型返回空字符串时安全回退"""
    mock_client = MagicMock()
    mock_client.extract_json.return_value = {"rewritten": "   "}
    rewriter = PreRetrievalRewriter(client=mock_client)
    
    history = [{"role": "user", "content": "陈嘉庚"}]
    rewritten = rewriter.rewrite("是谁", history=history)
    
    assert rewritten == "是谁", "返回空字符串时没有回退到原查询"


def test_pre_rewrite_uses_history():
    """验证透传 History，并在大模型正常时成功应用改写结果"""
    mock_client = MagicMock()
    mock_client.extract_json.return_value = {"rewritten": "陈嘉庚是谁", "reason": "补全主语"}
    rewriter = PreRetrievalRewriter(client=mock_client)
    
    history = [{"role": "user", "content": "陈嘉庚"}]
    rewritten = rewriter.rewrite("是谁", history=history)
    
    assert rewritten == "陈嘉庚是谁", "改写结果未被使用"
    
    # 验证传递给了大模型正确的历史文本
    call_args = mock_client.extract_json.call_args[0][0]
    assert "用户: 陈嘉庚" in call_args


def test_pre_rewrite_no_history_still_rewrites():
    """验证没有 history 或 history 为空时也应该调用改写（为了提升口语化/不完整句子）"""
    mock_client = MagicMock()
    mock_client.extract_json.return_value = {"rewritten": "陈嘉庚创办了哪些学校？", "reason": "规范表达"}
    rewriter = PreRetrievalRewriter(client=mock_client)

    assert rewriter.rewrite("创办了哪些学校") == "陈嘉庚创办了哪些学校？"
    assert rewriter.rewrite("创办了哪些学校", history=[]) == "陈嘉庚创办了哪些学校？"

    # 验证确实调用了大模型
    assert mock_client.extract_json.call_count == 2

def test_pre_rewrite_bad_history_fallback():
    """验证 history 格式损坏时不会崩溃，仍能继续执行改写或回退"""
    mock_client = MagicMock()
    mock_client.extract_json.return_value = {"rewritten": "安全通过"}
    rewriter = PreRetrievalRewriter(client=mock_client)

    bad_history = [{"wrong_key": "user"}] # 缺失 content 和 role
    rewritten = rewriter.rewrite("测一下", history=bad_history)

    assert rewritten == "安全通过"


def test_pipeline_preserves_original_query(monkeypatch):
    """验证改写后的 query 确实进入了分析和检索链路，但 Judge 和 Generation 仍围绕 original_query"""
    # Mock 掉重型检索器，避免 RAGPipeline.__init__ 加载真实 FAISS 索引 / sentence-transformers
    # （CI 环境无 data/vector_store/index.faiss，直连会抛 FileNotFoundError）
    import src.rag_pipeline as _rp
    _mock_hybrid = MagicMock()
    _mock_hybrid.vector_retriever = MagicMock()
    _mock_hybrid.bm25_retriever = MagicMock()
    monkeypatch.setattr(_rp, "HybridRetriever", MagicMock(return_value=_mock_hybrid))

    # Mock Pipeline 依赖
    pipeline = RAGPipeline(use_graph=False, use_rerank=False, enable_judge=True, enable_routing=False)
    
    # 拦截各个模块以追踪参数
    # 1. 拦截 Analyzer (确认收到改写后 query)
    mock_analyze = MagicMock(return_value={"query_mode": "hybrid"})
    monkeypatch.setattr(pipeline.query_analyzer, "analyze", mock_analyze)
    
    # 2. 拦截 Retriever (确认收到改写后 query)
    mock_retrieve = MagicMock(return_value=([{"chunk_id": "c1", "text": "陈嘉庚"}], [], []))
    monkeypatch.setattr(pipeline, "_retrieve_fuse_rerank", mock_retrieve)
    
    # 3. 拦截 Evidence Judge (确认收到 original query)
    from src.evidence_judge import EvidenceJudgement
    mock_judge = MagicMock(return_value=EvidenceJudgement(sufficient=True))
    monkeypatch.setattr(pipeline.evidence_judge, "judge", mock_judge)
    
    # 4. 拦截 Generator (确认收到 original query)
    mock_generate = MagicMock(return_value={"answer": "mock ans", "references": []})
    monkeypatch.setattr(pipeline.generator, "generate", mock_generate)
    
    # 5. 直接注入一个明确的改写结果
    mock_rewriter = MagicMock()
    mock_rewriter.rewrite.return_value = "陈嘉庚在哪里上学？"
    pipeline.pre_retrieval_rewriter = mock_rewriter

    # 执行 Query (模拟带历史)
    history = [{"role": "user", "content": "陈嘉庚"}]
    pipeline.query("在哪里上学？", history=history)

    # 验证 Analyzer 收到的是【改写后】的 Query
    mock_analyze.assert_called_once_with("陈嘉庚在哪里上学？")
    
    # 验证 Retriever 收到的是【改写后】的 Query
    assert mock_retrieve.call_args[0][0] == "陈嘉庚在哪里上学？"

    # 验证 Evidence Judge 收到的是【原】Query
    assert mock_judge.call_args[0][0] == "在哪里上学？"
    
    # 验证 Generator 收到的是【原】Query
    assert mock_generate.call_args[0][0] == "在哪里上学？"
