"""
一键端到端 RAG 消融实验：

    python -m evaluation.run_ablation

七组单变量递进方案（见 evaluation/profiles.py）：
    A Dense Only / B +BM25 / C +Entity Graph / D +Relation Retrieval /
    E +Rerank (Full Fixed RAG) / F Agent Router / G Agent Router + Retry

指标：
    检索侧：Recall@5、Recall@10、MRR、nDCG@5、nDCG@10
    生成侧：Answer Keyword Accuracy、Refusal Success Rate
    LLM-Judge（可选）：Faithfulness、Answer Correctness
    系统侧：Avg Latency、Avg Retrieval Rounds、Avg Retrieval Calls、Retry Rate
    Evidence Judge 校准（可选）：FP / FN 统计

结果写入 evaluation/results/<key>.json，并在 results/summary_table.json
汇总；控制台打印对比表。

注意：检索指标在缺少人工 relevant_chunks 标注时为 keyword-proxy 口径，
详见 evaluation/metrics.py。

可选参数：
    --dataset PATH       指定评估数据集（默认 evaluation/datasets/qa_eval.json）
    --benchmark          以公开 Benchmark 格式加载数据集（使用 benchmark_loader）
    --benchmark-format   公开数据集格式 (auto|crud-rag|multihop-rag|generic)
    --only KEY           只跑某一组（key 见 profiles.py）
    --limit N            只取前 N 题（调试用）
    --llm-judge          启用 LLM-as-Judge（Faithfulness + Answer Correctness，需 ground_truth）
"""
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Optional

from evaluation.metrics import retrieval_metrics, answer_accuracy
from evaluation.profiles import PROFILES

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = EVAL_DIR / "datasets" / "qa_eval.json"
RESULTS_DIR = EVAL_DIR / "results"


def load_dataset(path: Path = None, benchmark: bool = False,
                 benchmark_format: str = "auto") -> List[Dict]:
    """加载评估数据集；benchmark=True 时使用公开 Benchmark 加载器"""
    path = Path(path) if path else DEFAULT_DATASET
    if benchmark:
        from evaluation.benchmark_loader import load_benchmark
        return load_benchmark(str(path), fmt=benchmark_format)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def _compute_judge_fp_fn(details: List[Dict]) -> Dict[str, float]:
    """
    统计 Evidence Judge 的误判率。

    判定逻辑（需要 ground_truth 或 answer_keywords）：
      - TP: judge=sufficient 且 答案正确（accuracy>0.5 或 refusal 正确）
      - FP: judge=sufficient 但 答案错误/不完整（accuracy<=0.5）
      - FN: judge=insufficient 但 检索结果实际包含答案（recall@5>0）
      - TN: judge=insufficient 且 检索结果确实不包含答案

    注意：这是基于 proxy 指标的近似校准，非严格人工标注。
    """
    tp = fp = fn = tn = 0
    for r in details:
        sufficient = r.get("evidence_sufficient")
        if sufficient is None:
            continue  # 该组未启用 Evidence Judge

        accuracy = r.get("accuracy")
        recall5 = r.get("recall@5", 0)
        is_refusal = r.get("type") == "refusal"
        refusal_correct = r.get("refusal_correct")

        # 判定答案是否"实际可答"
        if is_refusal:
            answerable = refusal_correct is True  # 正确拒答 = 系统行为正确
        elif accuracy is not None:
            answerable = accuracy > 0.5
        else:
            answerable = recall5 > 0  # 无 accuracy 时用 recall proxy

        # 判定检索是否"实际有证据"
        has_evidence = recall5 > 0

        if sufficient:
            if answerable:
                tp += 1
            else:
                fp += 1  # 判充分但实际答不对 → 误报
        else:
            if has_evidence:
                fn += 1  # 判不足但实际有证据 → 漏报
            else:
                tn += 1

    total = tp + fp + fn + tn
    return {
        "judge_tp": tp,
        "judge_fp": fp,
        "judge_fn": fn,
        "judge_tn": tn,
        "judge_fp_rate": round(fp / max(tp + fp, 1), 3),
        "judge_fn_rate": round(fn / max(fn + tn, 1), 3),
        "judge_precision": round(tp / max(tp + fp, 1), 3),
        "judge_recall": round(tp / max(tp + fn, 1), 3),
        "judge_evaluated": total,
    }


def run_profile(profile: Dict, questions: List[Dict],
                use_llm_judge: bool = False) -> Dict:
    """构造一组 pipeline，跑全部问题，返回 summary + details"""
    from src.rag_pipeline import RAGPipeline

    print(f"\n{'=' * 72}")
    print(f"运行配置：{profile['name']}  (key={profile['key']})")
    print('=' * 72)

    # 每组独立 pipeline（组件齐全，由 plan/开关决定实际调用），避免缓存串组
    pipeline = RAGPipeline(use_graph=True, use_rerank=True, **profile["kwargs"])

    # LLM-Judge（可选）
    judge = None
    if use_llm_judge:
        from evaluation.llm_judge import LLMJudge
        judge = LLMJudge()
        print("  [LLM-Judge] 已启用 Faithfulness + Answer Correctness")

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

        # LLM-Judge 指标（需要 ground_truth）
        faith_score = None
        correctness_score = None
        if judge is not None:
            contexts = result.get("retrieved_chunks", [])
            ground_truth = item.get("ground_truth", "")
            if contexts:
                f_result = judge.faithfulness(result.get("answer", ""), contexts)
                if f_result:
                    faith_score = f_result["score"]
            if ground_truth:
                c_result = judge.answer_correctness(
                    question, result.get("answer", ""), ground_truth
                )
                if c_result:
                    correctness_score = c_result["score"]

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
            "faithfulness": faith_score,
            "answer_correctness": correctness_score,
            **acc_detail,
        }
        if rm is not None:
            row.update({
                "recall@5": round(rm["recall@5"], 3),
                "recall@10": round(rm["recall@10"], 3),
                "mrr": round(rm["mrr"], 3),
                "ndcg@5": round(rm.get("ndcg@5", 0), 3),
                "ndcg@10": round(rm.get("ndcg@10", 0), 3),
                "ground_truth": rm["ground_truth"],
            })
        details.append(row)
        print(f"  [{i + 1}/{len(questions)}] {item.get('type'):<9} "
              f"{question[:28]}...  {row['latency_total']}s  "
              f"retry={row['retry_count']} acc={row['accuracy']}"
              f"{' faith=' + str(faith_score) if faith_score is not None else ''}"
              f"{' correct=' + str(correctness_score) if correctness_score is not None else ''}")

    # Evidence Judge 误判统计（仅当该组启用了 Judge 时才有意义）
    judge_stats = _compute_judge_fp_fn(details)

    summary = {
        "config": profile["name"],
        "key": profile["key"],
        "total_questions": len(questions),
        "recall@5": round(_mean([r.get("recall@5") for r in details]), 3),
        "recall@10": round(_mean([r.get("recall@10") for r in details]), 3),
        "mrr": round(_mean([r.get("mrr") for r in details]), 3),
        "ndcg@5": round(_mean([r.get("ndcg@5") for r in details]), 3),
        "ndcg@10": round(_mean([r.get("ndcg@10") for r in details]), 3),
        "answer_accuracy": round(_mean([r.get("accuracy") for r in details]), 3),
        "faithfulness": round(_mean([r.get("faithfulness") for r in details]), 3),
        "answer_correctness": round(_mean([r.get("answer_correctness") for r in details]), 3),
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
        "evidence_judge_stats": judge_stats,
    }
    return {"summary": summary, "details": details}


def _print_table(summaries: List[Dict]):
    header = (f"{'Method':<24}{'R@5':>7}{'R@10':>7}{'MRR':>7}{'nDCG@5':>8}"
              f"{'Acc':>7}{'Latency':>9}{'Rounds':>8}{'Calls':>7}{'Retry%':>8}")
    print('\n' + '=' * len(header))
    print(header)
    print('-' * len(header))
    for s in summaries:
        print(f"{s['config']:<24}{s['recall@5']:>7.3f}{s['recall@10']:>7.3f}"
              f"{s['mrr']:>7.3f}{s['ndcg@5']:>8.3f}"
              f"{s['answer_accuracy']:>7.3f}"
              f"{s['avg_latency']:>8.2f}s{s['avg_retrieval_rounds']:>8.2f}"
              f"{s['avg_retrieval_calls']:>7.2f}"
              f"{s['retry_rate']:>8.1%}")
    print('=' * len(header))

    # LLM-Judge 指标表（如有非零值）
    has_judge = any(s.get("faithfulness") or s.get("answer_correctness") for s in summaries)
    if has_judge:
        print(f"\n{'LLM-Judge 指标':}")
        jheader = f"{'Method':<24}{'Faithfulness':>14}{'Ans Correctness':>16}"
        print('-' * len(jheader))
        print(jheader)
        print('-' * len(jheader))
        for s in summaries:
            print(f"{s['config']:<24}{s['faithfulness']:>14.3f}"
                  f"{s['answer_correctness']:>16.3f}")
        print('-' * len(jheader))

    # Evidence Judge 校准表（如有评估数据）
    has_judge_stats = any(s.get("evidence_judge_stats", {}).get("judge_evaluated", 0) > 0
                           for s in summaries)
    if has_judge_stats:
        print(f"\n{'Evidence Judge 校准（FP/FN 统计）':}")
        eheader = f"{'Method':<24}{'TP':>5}{'FP':>5}{'FN':>5}{'TN':>5}{'FP Rate':>9}{'FN Rate':>9}{'Precision':>10}"
        print('-' * len(eheader))
        print(eheader)
        print('-' * len(eheader))
        for s in summaries:
            js = s.get("evidence_judge_stats", {})
            if js.get("judge_evaluated", 0) == 0:
                continue
            print(f"{s['config']:<24}{js['judge_tp']:>5}{js['judge_fp']:>5}"
                  f"{js['judge_fn']:>5}{js['judge_tn']:>5}"
                  f"{js['judge_fp_rate']:>9.1%}{js['judge_fn_rate']:>9.1%}"
                  f"{js['judge_precision']:>10.3f}")
        print('-' * len(eheader))


def main():
    parser = argparse.ArgumentParser(description="RAG 七组单变量消融评估")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--benchmark", action="store_true",
                        help="以公开 Benchmark 格式加载数据集")
    parser.add_argument("--benchmark-format", default="auto",
                        choices=["auto", "crud-rag", "multihop-rag", "generic"],
                        help="公开数据集格式")
    parser.add_argument("--only", default=None, help="只跑指定 key 的一组")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--llm-judge", action="store_true",
                        help="启用 LLM-as-Judge（Faithfulness + Answer Correctness）")
    args = parser.parse_args()

    questions = load_dataset(args.dataset, benchmark=args.benchmark,
                             benchmark_format=args.benchmark_format)
    if args.limit:
        questions = questions[: args.limit]
    print(f"加载 {len(questions)} 道评估题：{args.dataset}"
          f"{' (benchmark format)' if args.benchmark else ''}")

    profiles = [p for p in PROFILES if (args.only is None or p["key"] == args.only)]
    if args.only and not profiles:
        raise SystemExit(f"未知 profile key：{args.only}，可选 "
                         f"{[p['key'] for p in PROFILES]}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    for profile in profiles:
        outcome = run_profile(profile, questions, use_llm_judge=args.llm_judge)
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
