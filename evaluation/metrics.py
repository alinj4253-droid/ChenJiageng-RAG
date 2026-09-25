"""
评估指标（纯函数，便于单测）

检索指标：
  - 若数据集条目含 ``relevant_chunks``（相关 chunk_id 列表），采用标准 chunk 级
    Recall@5 / Recall@10 / MRR；
  - 否则回退到 keyword-proxy 口径：用 ``answer_keywords`` 是否出现在 top-K chunk
    文本中作为弱监督信号（Recall = 被覆盖关键词比例，MRR = 首个命中关键词 chunk
    的排名倒数）。该口径无需人工 chunk 标注、确定性可复现，但不等同于人工标注，
    结果中以 ``ground_truth`` 字段标明。

生成指标（确定性，不依赖 LLM 裁判）：
  - 非拒答题：answer_keywords 在答案中的命中比例（Answer Keyword Accuracy）；
  - 拒答题：答案是否包含拒答信号（Refusal Success Rate）。
"""
from typing import List, Dict, Optional, Tuple, Set
import math

# 拒答信号词（与语料无证据时的谨慎回答保持一致）
REFUSAL_SIGNALS = [
    "无法回答", "没有相关", "语料中没有", "未找到", "没有找到", "找不到",
    "无法确定", "没有记载", "未提及", "缺少相关", "抱歉", "没有提及",
]


def _blob(chunks: List[Dict]) -> str:
    return "\n".join((c.get("text") or "") for c in chunks)


def _covered_keywords(chunks: List[Dict], keywords: List[str]) -> Set[str]:
    """返回被给定 chunks 文本覆盖到的关键词集合"""
    blob = _blob(chunks)
    return {kw for kw in keywords if kw and kw in blob}


def _chunk_level_metrics(retrieved: List[Dict],
                         relevant_ids: Set[str]) -> Dict[str, float]:
    ids = [c.get("chunk_id") for c in retrieved]

    def recall(k: int) -> float:
        if not relevant_ids:
            return 0.0
        return len(relevant_ids & set(ids[:k])) / len(relevant_ids)

    mrr = 0.0
    for i, cid in enumerate(ids):
        if cid in relevant_ids:
            mrr = 1.0 / (i + 1)
            break

    # nDCG@K：二分类相关性（relevant=1, irrelevant=0）
    def dcg(k: int) -> float:
        return sum(
            1.0 / math.log2(i + 2)
            for i, cid in enumerate(ids[:k])
            if cid in relevant_ids
        )

    def ndcg(k: int) -> float:
        if not relevant_ids:
            return 0.0
        ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_ids), k)))
        return dcg(k) / ideal if ideal > 0 else 0.0

    return {
        "recall@5": recall(5),
        "recall@10": recall(10),
        "mrr": mrr,
        "ndcg@5": ndcg(5),
        "ndcg@10": ndcg(10),
        "ground_truth": "chunk_id",
    }


def _keyword_proxy_metrics(retrieved: List[Dict],
                           keywords: List[str]) -> Dict[str, float]:
    if not keywords:
        return {"recall@5": 0.0, "recall@10": 0.0, "mrr": 0.0,
                "ndcg@5": 0.0, "ndcg@10": 0.0,
                "ground_truth": "keyword_proxy"}

    def coverage(k: int) -> float:
        covered = _covered_keywords(retrieved[:k], keywords)
        return len(covered) / len(keywords)

    # MRR：第一个命中任意答案关键词的 chunk 的排名倒数
    mrr = 0.0
    for i, chunk in enumerate(retrieved):
        if _covered_keywords([chunk], keywords):
            mrr = 1.0 / (i + 1)
            break

    # nDCG proxy：chunk 包含任意答案关键词视为相关（二分类）
    def dcg(k: int) -> float:
        return sum(
            1.0 / math.log2(i + 2)
            for i, chunk in enumerate(retrieved[:k])
            if _covered_keywords([chunk], keywords)
        )

    def ndcg(k: int) -> float:
        ideal = sum(1.0 / math.log2(i + 2) for i in range(k))  # 理想：前k全相关
        return dcg(k) / ideal if ideal > 0 else 0.0

    return {
        "recall@5": coverage(5),
        "recall@10": coverage(10),
        "mrr": mrr,
        "ndcg@5": round(ndcg(5), 4),
        "ndcg@10": round(ndcg(10), 4),
        "ground_truth": "keyword_proxy",
    }


def retrieval_metrics(retrieved_chunks: List[Dict],
                      item: Dict) -> Optional[Dict[str, float]]:
    """
    计算单题检索指标。

    relevant_chunks 存在 → 标准 chunk 级口径；否则 → keyword-proxy 口径。
    拒答题（无关键词、无标注）返回 None，调用方应将其排除出检索指标聚合。
    """
    relevant = item.get("relevant_chunks")
    if relevant:
        return _chunk_level_metrics(retrieved_chunks, set(relevant))

    keywords = [k for k in item.get("answer_keywords", []) if k]
    if not keywords:
        return None
    return _keyword_proxy_metrics(retrieved_chunks, keywords)


def answer_accuracy(answer: str, item: Dict) -> Tuple[Optional[float], Dict]:
    """
    计算单题生成得分。

    返回 (score, detail)：
      - refusal 题：正确拒答=1，否则=0；
      - 其余题：命中关键词数 / 关键词总数；
      - 无任何判定信号（非 refusal 且无关键词）→ score=None。
    """
    answer = answer or ""
    if item.get("type") == "refusal":
        ok = any(sig in answer for sig in REFUSAL_SIGNALS)
        return (1.0 if ok else 0.0), {"refusal_correct": ok}

    keywords = [k for k in item.get("answer_keywords", []) if k]
    if not keywords:
        return None, {}
    hits = sum(1 for kw in keywords if kw in answer)
    return hits / len(keywords), {
        "keyword_hits": hits,
        "keyword_total": len(keywords),
    }
