"""Prepare an isolated corpus and QA labels; never index answers or evidence facts.

CRUD uses all QA source articles embedded in the release (reduced corpus, not
the official 80k distractor corpus). MultiHop uses the separately released corpus.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

from evaluation.benchmark_loader import document_id, load_benchmark
from evaluation.benchmark_metrics import normalize_evidence


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_documents(path, fmt, corpus=None):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    docs = {}
    if fmt == "crud-rag":
        for task, rows in raw.items():
            if not task.startswith("questanswer"):
                continue
            for row in rows:
                for key in ("news1", "news2", "news3"):
                    text = row.get(key)
                    if isinstance(text, str) and text.strip():
                        did = document_id(text)
                        docs[did] = {"id": did, "title": did, "source_id": did, "text": text.strip()}
    else:
        if corpus is None:
            raise ValueError("MultiHop requires --corpus (official corpus.json)")
        for row in json.loads(Path(corpus).read_text(encoding="utf-8")):
            did = document_id(row.get("url") or row["title"])
            body = row.get("body") or row.get("content") or ""
            if not body.strip():
                raise ValueError(f"Empty corpus document: {did}")
            metadata = "\n".join(str(row.get(k) or "") for k in
                                 ("title", "author", "source", "published_at"))
            docs[did] = {"id": did, "title": row["title"],
                         "source_id": row.get("url") or row["title"],
                         "text": metadata + "\n" + body}
    return docs


def prepare(path, fmt, output, corpus=None, limit=None, seed=42, distractors=None):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be empty; use a new directory to avoid stale indexes")
    questions = load_benchmark(str(path), fmt)
    documents = source_documents(path, fmt, corpus)
    evidence_by_doc = {}
    for question in questions:
        for evidence in question.get("supporting_evidence", []):
            evidence_by_doc.setdefault(evidence["document_id"], {})[evidence["evidence_id"]] = evidence

    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        questions = random.Random(seed).sample(questions, min(limit, len(questions)))
    needed = {d for q in questions for d in q["relevant_docs"]}
    if needed - documents.keys():
        raise ValueError(f"Missing supporting documents: {sorted(needed - documents.keys())}")
    if distractors is not None:
        if distractors < 0:
            raise ValueError("distractors must be nonnegative")
        other = sorted(documents.keys() - needed)
        keep = needed | set(random.Random(seed).sample(other, min(distractors, len(other))))
        documents = {k: v for k, v in documents.items() if k in keep}
    # Fixed overlapping windows for both languages; article IDs stay separate
    # from chunk IDs so document recall does not count every window as a label.
    chunks = []
    mapping = {}
    source_mapping = {}
    for did, doc in sorted(documents.items()):
        for number, offset in enumerate(range(0, len(doc["text"]), 520)):
            text = doc["text"][offset:offset + 600]
            cid = f"{did}_{number:04d}"
            evidence_ids = [eid for eid, ev in evidence_by_doc.get(did, {}).items()
                            if normalize_evidence(ev["fact"]) in normalize_evidence(text)]
            source_mapping[cid] = {"chunk_id": cid, "document_id": did,
                                   "source_id": doc["source_id"],
                                   "benchmark_evidence_id": evidence_ids}
            chunks.append({"chunk_id": cid, "book": doc["title"], "chapter": str(number),
                           "text": text, "char_count": len(text)})
            mapping[cid] = did
            if offset + 600 >= len(doc["text"]):
                break
    if not questions or not chunks:
        raise ValueError("Empty benchmark")
    (output / "chunks").mkdir(parents=True)
    chunk_path = output / "chunks/chunks.jsonl"
    chunk_path.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks), encoding="utf-8")
    (output / "questions.json").write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {"format": fmt, "dataset_sha256": sha256(path),
                "corpus_sha256": sha256(corpus) if corpus else sha256(path),
                "questions_sha256": sha256(output / "questions.json"),
                "chunks_sha256": sha256(chunk_path), "seed": seed,
                "question_count": len(questions), "document_count": len(documents),
                "chunk_count": len(chunks), "chunk_to_doc": mapping,
                "corpus_scope": "sampled_supports_plus_distractors" if distractors is not None else
                ("crud_qa_source_articles_only" if fmt == "crud-rag" else "full_multihop_corpus")}
    (output / "chunk_mapping.json").write_text(json.dumps(source_mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["mapping_sha256"] = sha256(output / "chunk_mapping.json")
    (output / "benchmark_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--format", required=True, choices=["crud-rag", "multihop-rag"])
    parser.add_argument("--corpus")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distractors", type=int, help="Reduced-corpus smoke only; omit for all source documents")
    args = parser.parse_args()
    result = prepare(args.dataset, args.format, args.output, args.corpus,
                     args.limit, args.seed, args.distractors)
    print(json.dumps({k: v for k, v in result.items() if k != "chunk_to_doc"}, indent=2))


if __name__ == "__main__":
    main()
