"""
证据充分性判断（Evidence Judge）—— Agentic 决策点之二

职责严格单一：只回答“当前检索证据是否足以回答用户问题”。
不重新回答问题、不规划流程、不决定工具。

输出 EvidenceJudgement(sufficient, missing, reason)。

失败策略：
- 证据为空：直接判 insufficient，不浪费一次 LLM 调用；
- Judge LLM 调用失败 / 返回非法 JSON：fail-open，判 sufficient，
  避免因为裁判故障而无限重试或阻塞生成。
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from src.llm_client import get_client
from src.prompts import EVIDENCE_JUDGE_SYSTEM, EVIDENCE_JUDGE_USER
from src.config import FINAL_TOP_K


@dataclass
class EvidenceJudgement:
    """证据判断结果"""
    sufficient: bool
    missing: List[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> Dict:
        return {
            "sufficient": self.sufficient,
            "missing": self.missing,
            "reason": self.reason,
        }


class EvidenceJudge:
    """证据充分性裁判"""

    def __init__(self, client=None):
        self.client = client if client is not None else get_client()

    @staticmethod
    def _format_evidence(contexts: List[Dict], max_chunks: int = FINAL_TOP_K) -> str:
        """把检索 chunk 拼成编号证据文本"""
        if not contexts:
            return "（未检索到任何证据）"
        parts = []
        for i, c in enumerate(contexts[:max_chunks]):
            book = c.get("book", "")
            chapter = c.get("chapter", "")
            source = f"《{book}》{chapter}".strip("《》 ")
            header = f"[证据{i + 1}] {source}" if source else f"[证据{i + 1}]"
            parts.append(f"{header}\n{c.get('text', '')}")
        return "\n\n".join(parts)

    def judge(self, question: str, contexts: Optional[List[Dict]]) -> EvidenceJudgement:
        """
        判断证据是否足以回答 question。

        Args:
            question: 用户原始问题
            contexts: 检索 / 精排后的证据 chunk 列表
        """
        # 空证据：确定不充分，无需调用 LLM
        if not contexts:
            return EvidenceJudgement(
                sufficient=False,
                missing=["未检索到任何相关资料"],
                reason="检索结果为空",
            )

        evidence_text = self._format_evidence(contexts)
        prompt = EVIDENCE_JUDGE_USER.format(question=question, evidence=evidence_text)
        result = self.client.extract_json(prompt, system_prompt=EVIDENCE_JUDGE_SYSTEM)

        return self._parse(result)

    @staticmethod
    def _parse(result: Dict) -> EvidenceJudgement:
        """解析 LLM 输出，异常时 fail-open"""
        if not result:
            # 裁判不可用不应阻塞主流程，默认证据充分、直接生成
            return EvidenceJudgement(
                sufficient=True,
                missing=[],
                reason="证据裁判不可用，默认充分",
            )

        sufficient = bool(result.get("sufficient", True))
        missing = result.get("missing", [])
        if not isinstance(missing, list):
            missing = [str(missing)] if missing else []
        missing = [str(m) for m in missing if m]
        reason = str(result.get("reason", ""))

        return EvidenceJudgement(
            sufficient=sufficient,
            missing=missing,
            reason=reason,
        )
