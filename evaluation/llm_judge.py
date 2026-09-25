"""
LLM-as-Judge 评估指标

用于在有标准答案（ground_truth answer）和检索上下文（contexts）时，
对生成答案做更细粒度的质量评判，弥补 keyword coverage 的不足。

指标：
  - Faithfulness（忠实度）：答案中的每个事实声明是否都能被检索上下文支撑。
    1.0 = 完全无幻觉，0.0 = 完全编造。
  - Answer Correctness（答案正确性）：与标准答案对比，语义层面的正确程度。
    1.0 = 完全正确，0.0 = 完全错误。

设计原则：
  - Judge 输出结构化 JSON（score + reason），便于统计与溯源；
  - 失败时返回 None，不阻塞评估主流程；
  - 所有评判 prompt 明确要求只依据给定材料，不引入外部知识。
"""
import json
from typing import List, Dict, Optional

from src.llm_client import get_client


FAITHFULNESS_SYSTEM = """你是一个严谨的 RAG 忠实度评判专家。你的任务是判断模型生成的答案是否完全基于给定的检索上下文，不包含上下文中无法支撑的事实（幻觉）。

评分标准（0-1分）：
- 1.0：答案中所有事实声明都能在上下文中找到明确依据
- 0.7：绝大部分内容有依据，仅有少量无关或模糊表述
- 0.4：部分内容有依据，但存在明显无法支撑的声明
- 0.0：答案主要由编造或上下文中不存在的信息构成

只输出JSON：{"score": 0.0-1.0, "reason": "简短说明"}"""


FAITHFULNESS_USER = """【检索上下文】
{context}

【模型答案】
{answer}

请评判该答案相对于检索上下文的忠实度。只输出JSON。"""


CORRECTNESS_SYSTEM = """你是一个严谨的 RAG 答案正确性评判专家。你的任务是将模型生成的答案与标准答案进行语义对比，判断其正确程度。

评分标准（0-1分）：
- 1.0：答案完全正确，包含标准答案的所有关键信息点，无错误
- 0.7：答案基本正确，覆盖大部分关键信息点，仅有次要遗漏
- 0.4：答案部分正确，覆盖少量关键信息，或存在明显错误
- 0.0：答案完全错误或与问题无关

注意：评判基于语义而非字面匹配；答案措辞不同但意思相同应视为正确。

只输出JSON：{"score": 0.0-1.0, "reason": "简短说明"}"""


CORRECTNESS_USER = """【问题】
{question}

【标准答案】
{ground_truth}

【模型答案】
{answer}

请评判模型答案相对于标准答案的正确程度。只输出JSON。"""


class LLMJudge:
    """LLM-as-Judge 评估器（Faithfulness + Answer Correctness）"""

    def __init__(self, client=None):
        self.client = client if client is not None else get_client()

    @staticmethod
    def _format_context(contexts: List[Dict], max_chunks: int = 5) -> str:
        if not contexts:
            return "（无检索上下文）"
        parts = []
        for i, c in enumerate(contexts[:max_chunks]):
            book = c.get("book", "")
            chapter = c.get("chapter", "")
            header = f"[{i+1}] 《{book}》{chapter}"
            parts.append(f"{header}\n{c.get('text', '')[:500]}")
        return "\n\n".join(parts)

    def faithfulness(self, answer: str, contexts: List[Dict]) -> Optional[Dict]:
        """
        评判答案的忠实度（是否有幻觉）。

        Returns:
            {"score": float, "reason": str} 或 None（评判失败）
        """
        if not answer or not contexts:
            return None
        context_text = self._format_context(contexts)
        prompt = FAITHFULNESS_USER.format(context=context_text, answer=answer[:1000])
        result = self.client.extract_json(prompt, system_prompt=FAITHFULNESS_SYSTEM)
        if not result:
            return None
        score = result.get("score")
        if score is None:
            return None
        try:
            score = float(score)
            score = max(0.0, min(1.0, score))
        except (ValueError, TypeError):
            return None
        return {"score": score, "reason": str(result.get("reason", ""))}

    def answer_correctness(self, question: str, answer: str,
                            ground_truth: str) -> Optional[Dict]:
        """
        评判答案的正确性（与标准答案语义对比）。

        Returns:
            {"score": float, "reason": str} 或 None（评判失败）
        """
        if not answer or not ground_truth:
            return None
        prompt = CORRECTNESS_USER.format(
            question=question,
            ground_truth=ground_truth[:1000],
            answer=answer[:1000],
        )
        result = self.client.extract_json(prompt, system_prompt=CORRECTNESS_SYSTEM)
        if not result:
            return None
        score = result.get("score")
        if score is None:
            return None
        try:
            score = float(score)
            score = max(0.0, min(1.0, score))
        except (ValueError, TypeError):
            return None
        return {"score": score, "reason": str(result.get("reason", ""))}
