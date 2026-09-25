"""Phase one: evaluate only the complete dynamic Router + Judge + Retry pipeline."""
import argparse
import json
import os
import time
from pathlib import Path

from evaluation.benchmark_metrics import answer_f1, recall_at_ten


def _mean(values):
    values = [x for x in values if x is not None]
    return sum(values) / len(values) if values else None


def validate_workspace(root, questions):
    from evaluation.prepare_benchmark import sha256
    root = Path(root).resolve()
    manifest = json.loads((root / "benchmark_manifest.json").read_text(encoding="utf-8"))
    for relative, field in [("chunks/chunks.jsonl", "chunks_sha256"),
                            ("questions.json", "questions_sha256"),
                            ("chunk_mapping.json", "mapping_sha256")]:
        if sha256(root / relative) != manifest[field]:
            raise ValueError(f"Benchmark manifest mismatch: {relative}")
    stored = json.loads((root / "questions.json").read_text(encoding="utf-8"))
    if any(q not in stored for q in questions):
        raise ValueError("Questions do not belong to this benchmark workspace")
    # Reject domain/stale dense indexes before any API calls or scoring.
    chunks = [json.loads(line) for line in (root / "chunks/chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    metadata = [json.loads(line) for line in (root / "vector_store/metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    if metadata != chunks:
        raise ValueError("Vector metadata does not match benchmark chunks; rebuild indexes")
    mapping = manifest["chunk_to_doc"]
    if set(mapping) != {c["chunk_id"] for c in chunks}:
        raise ValueError("Chunk mapping does not match corpus")
    needed = {d for q in questions for d in q.get("relevant_docs", [])}
    if needed - set(mapping.values()):
        raise ValueError("Supporting document IDs do not match corpus")
    return manifest



def evaluate(questions, pipeline, clock=time.perf_counter, completed=None, checkpoint=None):
    details = list(completed or [])
    if len(details) > len(questions) or any(
        row.get("id") != item.get("id") or row.get("question") != item["question"]
        or row.get("ground_truth") != item["ground_truth"]
        for row, item in zip(details, questions)
    ):
        raise ValueError("Checkpoint does not match this question sequence")
    for i, item in enumerate(questions[len(details):], start=len(details)):
        if not item.get("ground_truth"):
            raise ValueError("F1 requires a reference answer")
        pipeline.query_cache.clear()
        start = clock()
        result = pipeline.query(item["question"], verbose=False)
        latency = clock() - start
        answer = result.get("answer") or ""
        if not answer.strip():
            raise RuntimeError("Generation failed: empty answer; refusing to score an outage")
        # The actual workflow fuses ten candidates and retains five after rerank.
        # Recall@10 therefore refers to the final round's fusion candidates,
        # not an invented ten-item final generation context.
        candidates = result.get("retrieval_candidates", [])
        score = recall_at_ten(candidates, item)
        row = {"id": item.get("id"), "question": item["question"],
               "category": item.get("category"), "ground_truth": item["ground_truth"],
               "answer": answer, "recall@10": score,
               "f1": answer_f1(answer, item["ground_truth"]), "latency_seconds": latency,
               "retrieval_calls": result.get("retrieval_calls", 0),
               "retry_count": result.get("retry_count", 0),
               "retrieval_plan": result.get("retrieval_plan"),
               "top10_chunk_ids": [c["chunk_id"] for c in candidates[:10]],
               "final_context_ids": [c["chunk_id"] for c in result.get("retrieved_chunks", [])],
               "evidence_judgement": result.get("evidence_judgement")}
        details.append(row)
        if checkpoint is not None:
            checkpoint(details)
        print(f"[{i+1}/{len(questions)}] R@10={score} F1={row['f1']:.3f} latency={latency:.2f}s", flush=True)
    return {"summary": {"total_questions": len(questions),
                        "retrieval_scored_questions": sum(r["recall@10"] is not None for r in details),
                        "recall@10": _mean([r["recall@10"] for r in details]),
                        "f1": _mean([r["f1"] for r in details]),
                        "avg_latency_seconds": _mean([r["latency_seconds"] for r in details])},
            "diagnostics": {"avg_retrieval_calls": _mean([r["retrieval_calls"] for r in details]),
                            "retry_rate": _mean([float(r["retry_count"] > 0) for r in details])},
            "details": details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true", help="Resume validated per-question checkpoint")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    root = Path(args.data_dir).resolve()
    os.environ["RAG_DATA_DIR"] = str(root)
    questions = json.loads((root / "questions.json").read_text(encoding="utf-8"))
    manifest = validate_workspace(root, questions)
    if manifest["format"] != "multihop-rag":
        parser.error("Phase one supports MultiHop-RAG only")
    if args.limit:
        questions = questions[:args.limit]
    if not questions:
        parser.error("No evaluation questions")
    out = Path(args.output_dir) if args.output_dir else root / "results"
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "checkpoint.json"
    from evaluation.prepare_benchmark import sha256
    project = Path(__file__).resolve().parent.parent
    code_fingerprint = {str(p.relative_to(project)): sha256(p)
                        for folder in (project / "src", project / "evaluation")
                        for p in sorted(folder.glob("*.py"))}
    fingerprint = {"code_sha256": code_fingerprint, "questions_sha256": manifest["questions_sha256"],
                   "chunks_sha256": manifest["chunks_sha256"],
                   "metric_version": "evidence_recall10_token_f1_v1", "question_count": len(questions)}
    completed = []
    if args.resume and checkpoint_path.exists():
        saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if saved["fingerprint"] != fingerprint:
            parser.error("Checkpoint fingerprint mismatch")
        completed = saved["details"]

    def checkpoint(details):
        temporary = checkpoint_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"fingerprint": fingerprint, "details": details},
                                        ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(checkpoint_path)
    from src.config import KG_DIR
    required = ["entities.jsonl", "triples.jsonl", "entity_index.faiss", "relation_index.faiss"]
    if any(not (KG_DIR / f).exists() for f in required):
        parser.error("Graph indexes missing; build the complete benchmark knowledge base first")
    for filename in ("entities.jsonl", "triples.jsonl"):
        records = [json.loads(x) for x in (KG_DIR / filename).read_text(encoding="utf-8").splitlines()]
        if not records or any(set(r["source_chunks"]) - manifest["chunk_to_doc"].keys() for r in records):
            parser.error("Graph corpus mismatch or empty graph")
    from src.rag_pipeline import RAGPipeline
    pipeline = RAGPipeline(use_graph=True, use_rerank=True, enable_routing=True,
                           enable_judge=True, max_retry=1)
    if pipeline.reranker.model is None:
        raise RuntimeError("Reranker unavailable; refusing to label a degraded run as full pipeline")
    if pipeline.vector_retriever.index.ntotal != manifest["chunk_count"]:
        raise RuntimeError("Dense index count does not match corpus")
    graph = pipeline.graph_retriever
    if graph.entity_index.ntotal != len(graph.entity_meta) or graph.relation_index.ntotal != len(graph.relations):
        raise RuntimeError("Graph index count does not match metadata")
    pipeline.vector_retriever._get_model()
    pipeline.graph_retriever._get_embedding_model()
    outcome = evaluate(questions, pipeline, completed=completed, checkpoint=checkpoint)
    outcome["benchmark"] = {k: v for k, v in manifest.items() if k != "chunk_to_doc"}
    outcome["configuration"] = {"pipeline": "full_dynamic", "max_retry": 1,
                                "retrieval_metric_stage": "final_round_rrf_top10",
                                "generation_metric": "normalized_token_f1",
                                "generation_model": pipeline.generator.client.last_model}
    (out / "full_pipeline.json").write_text(json.dumps(outcome, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": outcome["summary"], "diagnostics": outcome["diagnostics"]}, indent=2))


if __name__ == "__main__":
    main()
