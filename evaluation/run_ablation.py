"""
一键端到端 RAG 消融实验：

    python -m evaluation.run_ablation

七组单变量递进方案（见 evaluation/profiles.py）：
    A Dense Only / B +BM25 / C +Entity Graph / D +Relation Retrieval /
    E +Rerank (Full Fixed RAG) / F Agent Router / G Agent Router + Retry

指标：
    Recall@5、Recall@10、MRR、Answer Accuracy、
    Avg Latency、Avg Retrieval Rounds、Avg Retrieval Calls、Retry Rate

结果写入 evaluation/results/<key>.json，并在 results/summary_table.json
汇总；控制台打印对比表。

注意：检索指标在缺少人工 relevant_chunks 标注时为 keyword-proxy 口径，
详见 evaluation/metrics.py 与 evaluation/datasets/README.md。

可选参数：
    --dataset PATH   指定评估数据集（默认 evaluation/datasets/qa_eval.json）
    --only KEY       只跑某一组（key 见 profiles.py）
    --limit N        只取前 N 题（调试用）
"""
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict

from evaluation.metrics import retrieval_metrics, answer_accuracy
from evaluation.profiles import PROFILES

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = EVAL_DIR / "datasets" / "qa_eval.json"
RESULTS_DIR = EVAL_DIR / "results"


def load_dataset(path: Path = None) -> List[Dict]:
    path = Path(path) if path else DEFAULT_DATASET
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def run_profile(profile: Dict, questions: List[Dict]) -> Dict:
    """构造一组 pipeline，跑全部问题，返回 summary + details"""
    from src.rag_pipeline import RAGPipeline

    print(f"\n{'=' * 72}")
    print(f"运行配置：{profile['name']}  (key={profile['key']})")
    print('=' * 72)

    # 每组独立 pipeline（组件齐全，由 plan/开关决定实际调用），避免缓存串组
    pipeline = RAGPipeline(use_graph=True, use_rerank=True, **profile["kwargs"])

    details = []
    for i, item in enumerate(questions):
        question = item["question"]
        t0 = time.time()
        result = pipeline.query(question, verbose=False)
        wall = time.time() - t0

        rm = retrieval_metrics(
            result.get("retrieval_candidates")
            or result.get("retrieved_chunks", []),
            item,
        )
        acc, acc_detail = answer_accuracy(result.get("answer", ""), item)

        latency = result.get("latency", {})
        row = {
            "id": item.get("id"),
            "type": item.get("type"),
            "question": question,
            "query_mode": result.get("query_mode"),
            "retry_count": result.get("retry_count", 0),
            "retrieval_rounds": result.get("retrieval_rounds", 1),
            "retrieval_calls": result.get("retrieval_calls", 0),
            "retrieval_call_detail": result.get("retrieval_call_detail", {}),
            "evidence_sufficient": result.get("evidence_sufficient"),
            "answer": (result.get("answer") or "")[:300],
            "latency_total": round(latency.get("total", wall), 3),
            "wall_time": round(wall, 3),
            "accuracy": None if acc is None else round(acc, 3),
            **acc_detail,
        }
        if rm is not None:
            row.update({
                "recall@5": round(rm["recall@5"], 3),
                "recall@10": round(rm["recall@10"], 3),
                "mrr": round(rm["mrr"], 3),
                "ground_truth": rm["ground_truth"],
            })
        details.append(row)
        print(f"  [{i + 1}/{len(questions)}] {item.get('type'):<9} "
              f"{question[:28]}...  {row['latency_total']}s  "
              f"retry={row['retry_count']} acc={row['accuracy']}")

    summary = {
        "config": profile["name"],
        "key": profile["key"],
        "total_questions": len(questions),
        "recall@5": round(_mean([r.get("recall@5") for r in details]), 3),
        "recall@10": round(_mean([r.get("recall@10") for r in details]), 3),
        "mrr": round(_mean([r.get("mrr") for r in details]), 3),
        "answer_accuracy": round(_mean([r.get("accuracy") for r in details]), 3),
        "refusal_success": round(_mean([
            1.0 if r.get("refusal_correct") else (0.0 if r["type"] == "refusal" else None)
            for r in details
        ]), 3),
        "avg_latency": round(_mean([r["latency_total"] for r in details]), 3),
        "avg_retrieval_rounds": round(
            _mean([r["retrieval_rounds"] for r in details]), 3
        ),
        "avg_retrieval_calls": round(
            _mean([r["retrieval_calls"] for r in details]), 3
        ),
        "retry_rate": round(
            sum(1 for r in details if r["retry_count"] > 0) / max(len(details), 1), 3
        ),
    }
    return {"summary": summary, "details": details}


def _print_table(summaries: List[Dict]):
    header = (f"{'Method':<24}{'Recall@5':>10}{'Recall@10':>10}{'MRR':>8}"
              f"{'Acc':>7}{'Latency':>9}{'Rounds':>8}{'Calls':>7}{'Retry%':>8}")
    print('\n' + '=' * len(header))
    print(header)
    print('-' * len(header))
    for s in summaries:
        print(f"{s['config']:<24}{s['recall@5']:>10.3f}{s['recall@10']:>10.3f}"
              f"{s['mrr']:>8.3f}{s['answer_accuracy']:>7.3f}"
              f"{s['avg_latency']:>8.2f}s{s['avg_retrieval_rounds']:>8.2f}"
              f"{s['avg_retrieval_calls']:>7.2f}"
              f"{s['retry_rate']:>8.1%}")
    print('=' * len(header))


def main():
    parser = argparse.ArgumentParser(description="RAG 六组消融评估")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--only", default=None, help="只跑指定 key 的一组")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    questions = load_dataset(args.dataset)
    if args.limit:
        questions = questions[: args.limit]
    print(f"加载 {len(questions)} 道评估题：{args.dataset}")

    profiles = [p for p in PROFILES if (args.only is None or p["key"] == args.only)]
    if args.only and not profiles:
        raise SystemExit(f"未知 profile key：{args.only}，可选 "
                         f"{[p['key'] for p in PROFILES]}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    for profile in profiles:
        outcome = run_profile(profile, questions)
        out_path = RESULTS_DIR / f"{profile['key']}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(outcome, f, ensure_ascii=False, indent=2)
        summaries.append(outcome["summary"])
        print(f"已保存：{out_path}")

    with open(RESULTS_DIR / "summary_table.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    _print_table(summaries)
    print(f"\n汇总已保存：{RESULTS_DIR / 'summary_table.json'}")


if __name__ == "__main__":
    main()
