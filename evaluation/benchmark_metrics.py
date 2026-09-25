"""Phase-one metrics; no additional LLM scoring calls."""
import re
import string
from collections import Counter


def normalize_evidence(text):
    """Match the official retrieval evaluator's space/newline normalization."""
    return text.replace(" ", "").replace("\n", "")


def recall_at_ten(chunks, item):
    """Fraction of gold facts found verbatim in individual top-ten chunks.

    Matching an article alone is insufficient. null_query has no retrieval
    target and is excluded, as in the official retrieval evaluator.
    """
    if item.get("category") == "null_query":
        return None
    gold = {normalize_evidence(e["fact"]) for e in item.get("supporting_evidence", []) if e.get("fact")}
    if not gold:
        return None
    retrieved = [normalize_evidence(c.get("text", "")) for c in chunks[:10]]
    return sum(any(fact in text for text in retrieved) for fact in gold) / len(gold)


def answer_f1(answer, ground_truth):
    """English normalized token-overlap F1, using multiset intersection.

    This is explicitly token F1, not the official repository's looser
    any-word-overlap success rate which its script also calls F1.
    """
    def tokens(text):
        text = text.lower().translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        return text.split()
    predicted, gold = tokens(answer), tokens(ground_truth)
    if not predicted or not gold:
        return float(predicted == gold)
    overlap = sum((Counter(predicted) & Counter(gold)).values())
    return 2 * overlap / (len(predicted) + len(gold))
