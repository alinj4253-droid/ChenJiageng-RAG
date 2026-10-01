"""
查询改写器（Query Rewriter）—— 服务于有限 Retry

仅在 Evidence Judge 判定证据不足时调用：
根据缺失信息点把当前检索查询改写为更可能召回缺失证据的新查询。

输出 RewrittenQuery(query, reason)；改写失败返回 None（调用方据此停止重试）。
"""
from dataclasses import dataclass
from typing import List, Dict, Optional
import logging

from src.llm_client import get_client
from src.prompts import (
    QUERY_REWRITER_SYSTEM, QUERY_REWRITER_USER,
    PRE_RETRIEVAL_REWRITE_SYSTEM, PRE_RETRIEVAL_REWRITE_USER
)
from src.evidence_judge import EvidenceJudge

logger = logging.getLogger(__name__)


@dataclass
class RewrittenQuery:
    query: str
    reason: str = ""


class PreRetrievalRewriter:
    """检索前查询改写器：补全代词和缺失语境"""

    def __init__(self, client=None):
        self.client = client if client is not None else get_client()

    def rewrite(self, query: str, history: Optional[List[Dict]] = None) -> str:
        """
        根据历史上下文对原查询进行轻量级改写。
        如果改写失败，必须安全返回原查询 (fallback to original)。
        """
        # 没有历史或者全是空白，退化为不需要改写
        if not history:
            return query.strip()

        history_text = "\n".join(
            f"{'用户' if msg['role'] == 'user' else '助手'}: {msg['content']}"
            for msg in history[-4:]  # 仅带入最近几轮
        )
        if not history_text.strip():
            return query.strip()

        prompt = PRE_RETRIEVAL_REWRITE_USER.format(
            history=history_text,
            query=query,
        )

        try:
            result = self.client.extract_json(
                prompt, system_prompt=PRE_RETRIEVAL_REWRITE_SYSTEM, max_tokens=512
            )
            if result and result.get("rewritten"):
                rewritten = result["rewritten"].strip()
                if rewritten:
                    logger.info(f"检索前改写: '{query}' -> '{rewritten}' (原因: {result.get('reason', '')})")
                    return rewritten
        except Exception as e:
            logger.warning(f"检索前改写失败，回退原查询 '{query}': {e}")

        return query.strip()


class QueryRewriter:
    """证据不足时的检索查询改写器"""

    def __init__(self, client=None):
        self.client = client if client is not None else get_client()

    def rewrite(self, original_query: str, current_query: str,
                missing: Optional[List[str]],
                contexts: Optional[List[Dict]]) -> Optional[RewrittenQuery]:
        """
        Args:
            original_query: 用户原始问题
            current_query: 上一轮检索实际使用的查询（首轮等于原始问题）
            missing: Evidence Judge 给出的缺失信息点
            contexts: 上一轮检索到的证据（避免重复同一表述）
        Returns:
            RewrittenQuery 或 None（改写失败 / 无有效输出）
        """
        missing = missing or []
        missing_text = "\n".join(f"- {m}" for m in missing if m) or "（未明确列出）"
        evidence_text = EvidenceJudge._format_evidence(contexts)

        prompt = QUERY_REWRITER_USER.format(
            original_query=original_query,
            current_query=current_query,
            missing=missing_text,
            evidence=evidence_text,
        )
        result = self.client.extract_json(prompt, system_prompt=QUERY_REWRITER_SYSTEM)
        return self._parse(result)

    @staticmethod
    def _parse(result: Dict) -> Optional[RewrittenQuery]:
        if not result:
            return None
        rewritten = (result.get("rewritten_query") or "").strip()
        if not rewritten:
            return None
        return RewrittenQuery(
            query=rewritten,
            reason=str(result.get("reason", "")),
        )
