"""
查询改写器（Query Rewriter）—— 服务于有限 Retry

仅在 Evidence Judge 判定证据不足时调用：
根据缺失信息点把当前检索查询改写为更可能召回缺失证据的新查询。

输出 RewrittenQuery(query, reason)；改写失败返回 None（调用方据此停止重试）。
"""
from dataclasses import dataclass
from typing import List, Dict, Optional

from src.llm_client import get_client
from src.prompts import QUERY_REWRITER_SYSTEM, QUERY_REWRITER_USER
from src.evidence_judge import EvidenceJudge


@dataclass
class RewrittenQuery:
    query: str
    reason: str = ""


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
