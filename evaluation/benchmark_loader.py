"""
公开 Benchmark 数据集加载器

支持将 CRUD-RAG、MultiHop-RAG 等公开 RAG 评测集转换为内部统一格式，
使 evaluation.run_ablation 可以直接在公开数据集上运行。

内部统一格式（与 evaluation/datasets/qa_eval.json 对齐）：
    {
        "id": str/int,
        "question": str,
        "type": "fact" | "relation" | "overview" | "multihop" | "refusal",
        "answer_keywords": [str],        # 可选，用于 keyword-proxy
        "ground_truth": str,              # 标准答案，用于 LLM-as-Judge Correctness
        "relevant_chunks": [str],         # 可选，相关 chunk_id（用于标准 Recall/MRR/nDCG）
        "category": str,                   # 可选，问题分类标签
        "source": str,                     # 数据集来源标记
    }

支持的公开数据集：
  - CRUD-RAG: https://github.com/OSU-NLP-Group/CRUD-RAG
    格式：JSON，含 question / answer / relevant document IDs
  - MultiHop-RAG: https://github.com/yixuantt/MultiHop-RAG
    格式：JSON/JSONL，含 query / answer / supporting_facts
"""
import json
from pathlib import Path
from typing import List, Dict


def _ensure_id(item: Dict, idx: int) -> str:
    return str(item.get("id") or item.get("qid") or item.get("_id") or idx)


def load_crud_rag(path: Path) -> List[Dict]:
    """
    加载 CRUD-RAG 数据集（IAAR-Shanghai/CRUD_RAG）。

    数据格式：JSON 对象，包含 6 个任务键：
      - questanswer_1doc / questanswer_2docs / questanswer_3docs: QA 任务
        每个 item 含 questions(list) / answers(list) / news1~news3(源文档)
      - continuing_writing: 文本续写
      - hallu_modified: 幻觉修改
      - event_summary: 事件摘要

    本加载器仅提取 QA 任务（questanswer_*），将其展开为统一格式。
    源文档（news1~news3）的 ID 作为 relevant_chunks。
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        # 已经是扁平列表，按通用格式处理
        return load_generic_json(path)

    converted = []
    qa_keys = [k for k in raw.keys() if k.startswith("questanswer")]

    for task_key in qa_keys:
        items = raw[task_key]
        if not isinstance(items, list):
            continue
        # 从键名提取文档数（1doc / 2docs / 3docs）
        doc_count = 1
        if "2doc" in task_key:
            doc_count = 2
        elif "3doc" in task_key:
            doc_count = 3

        for item in items:
            questions = item.get("questions") or []
            answers = item.get("answers") or []
            doc_id = item.get("ID", "")
            # 源文档作为相关 chunk（用 ID 标识）
            relevant = [f"{doc_id}_news{i+1}" for i in range(doc_count)]

            for q_idx, question in enumerate(questions):
                if not question or not isinstance(question, str):
                    continue
                answer = answers[q_idx] if q_idx < len(answers) else ""
                converted.append({
                    "id": f"{doc_id}_q{q_idx}",
                    "question": question,
                    "type": "multihop" if doc_count > 1 else "fact",
                    "answer_keywords": [],
                    "ground_truth": answer if isinstance(answer, str) else str(answer),
                    "relevant_chunks": relevant,
                    "category": f"CRUD-QA-{doc_count}doc",
                    "source": "CRUD-RAG",
                })

    return converted


def load_multihop_rag(path: Path) -> List[Dict]:
    """
    加载 MultiHop-RAG 数据集（yixuantt/MultiHop-RAG）。

    数据格式：JSON 列表，每个 item 含：
        - query: 多跳问题
        - answer: 标准答案
        - question_type: 问题类型（comparison / inference_query / etc.）
        - evidence_list: 支撑证据文档列表，每个含 title / author / url / content
    """
    items = []
    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
    else:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
            if isinstance(raw, dict):
                items = raw.get("data") or raw.get("questions") or []
            else:
                items = raw

    converted = []
    for idx, item in enumerate(items):
        question = item.get("query") or item.get("question") or ""
        answer = item.get("answer") or item.get("ground_truth") or ""
        qtype = item.get("question_type") or item.get("type") or "multihop"

        # evidence_list → relevant_chunks（用 title 作为 doc 标识）
        evidence = (
            item.get("evidence_list")
            or item.get("supporting_facts")
            or item.get("supporting_contexts")
            or []
        )
        relevant = []
        if isinstance(evidence, list):
            for ev in evidence:
                if isinstance(ev, dict):
                    title = ev.get("title") or ev.get("doc_title") or ""
                    if title:
                        relevant.append(title)
                elif isinstance(ev, str):
                    relevant.append(ev)

        converted.append({
            "id": _ensure_id(item, idx),
            "question": question,
            "type": "multihop",
            "answer_keywords": [],
            "ground_truth": answer,
            "relevant_chunks": relevant,
            "category": qtype,
            "source": "MultiHop-RAG",
        })

    return converted


def load_generic_json(path: Path) -> List[Dict]:
    """
    通用 JSON/JSONL 加载器：尝试自动识别字段。
    适用于格式不标准的评测集。
    """
    items = []
    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
    else:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
            if isinstance(raw, dict):
                items = raw.get("data") or raw.get("questions") or raw.get("examples") or [raw]
            else:
                items = raw

    converted = []
    for idx, item in enumerate(items):
        question = (
            item.get("question") or item.get("query")
            or item.get("input") or item.get("text") or ""
        )
        answer = (
            item.get("answer") or item.get("ground_truth")
            or item.get("output") or item.get("response") or ""
        )
        keywords = item.get("answer_keywords") or item.get("keywords") or []
        relevant = item.get("relevant_chunks") or item.get("relevant_docs") or []

        converted.append({
            "id": _ensure_id(item, idx),
            "question": question,
            "type": item.get("type") or "fact",
            "answer_keywords": keywords if isinstance(keywords, list) else [],
            "ground_truth": answer,
            "relevant_chunks": [str(r) for r in relevant if r] if isinstance(relevant, list) else [],
            "category": item.get("category") or "",
            "source": path.stem,
        })

    return converted


def load_benchmark(path: str, fmt: str = "auto") -> List[Dict]:
    """
    加载公开 Benchmark 数据集，自动识别格式。

    Args:
        path: 数据集文件路径
        fmt: 格式 ("auto" | "crud-rag" | "multihop-rag" | "generic")

    Returns:
        内部统一格式的题目列表
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Benchmark 文件不存在: {path}")

    if fmt == "auto":
        name = p.name.lower()
        if "crud" in name:
            fmt = "crud-rag"
        elif "multihop" in name or "multi_hop" in name or "multi-hop" in name:
            fmt = "multihop-rag"
        else:
            fmt = "generic"

    if fmt == "crud-rag":
        return load_crud_rag(p)
    elif fmt == "multihop-rag":
        return load_multihop_rag(p)
    else:
        return load_generic_json(p)
