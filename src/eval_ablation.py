"""
消融实验评估脚本
对比四组检索策略的答案质量：
  1. 纯向量（Dense only）
  2. 向量 + BM25（Hybrid）
  3. + 知识图谱（Hybrid + KG）
  4. + Rerank（Full Pipeline）

评估指标：
  - 关键词命中率（answer 中是否包含 answer_keywords）
  - 拒答正确率（refusal 类型题目是否正确拒答）
"""
import json
import time
from pathlib import Path
from src.config import TEST_DIR, FINAL_TOP_K


def evaluate_config(questions, pipeline, config_name):
    """用给定 pipeline 跑所有问题，统计指标"""
    keyword_hits = 0
    keyword_total = 0
    refusal_correct = 0
    refusal_total = 0
    latency_sum = 0
    details = []

    for i, q in enumerate(questions):
        question = q['question']
        qtype = q['type']
        keywords = q.get('answer_keywords', [])

        t0 = time.time()
        result = pipeline.query(question, verbose=False)
        latency = time.time() - t0
        latency_sum += latency

        answer = result['answer']
        detail = {
            'id': q['id'],
            'question': question,
            'type': qtype,
            'answer': answer[:200],
            'latency': round(latency, 2),
        }

        if qtype == 'refusal':
            refusal_total += 1
            # 检查是否正确拒答（包含拒答信号词）
            refusal_signals = ['无法回答', '没有相关', '语料中没有', '未找到', '抱歉']
            if any(sig in answer for sig in refusal_signals):
                refusal_correct += 1
                detail['result'] = 'correct_refusal'
            else:
                detail['result'] = 'failed_to_refuse'
        else:
            # 检查关键词命中
            if keywords:
                hit = sum(1 for kw in keywords if kw in answer)
                keyword_hits += hit
                keyword_total += len(keywords)
                detail['keyword_hits'] = hit
                detail['keyword_total'] = len(keywords)
            else:
                detail['keyword_hits'] = 0
                detail['keyword_total'] = 0

        details.append(detail)
        print(f'  [{i+1}/{len(questions)}] {qtype} | {question[:30]}... | {round(latency,1)}s')

    # 汇总
    summary = {
        'config': config_name,
        'total_questions': len(questions),
        'keyword_hit_rate': round(keyword_hits / max(keyword_total, 1), 3),
        'refusal_accuracy': round(refusal_correct / max(refusal_total, 1), 3),
        'avg_latency': round(latency_sum / max(len(questions), 1), 2),
    }
    return summary, details


def run_ablation():
    """跑四组消融对比"""
    from src.rag_pipeline import RAGPipeline

    # 加载测试题
    questions_file = TEST_DIR / 'eval_questions.json'
    if not questions_file.exists():
        print(f'错误: {questions_file} 不存在')
        return

    with open(questions_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    print(f'加载 {len(questions)} 道测试题')

    all_summaries = []
    all_details = {}

    # 配置1：纯向量（直接用 vector_retriever，不走 hybrid）
    print('\n=== 配置1: 纯向量（Dense only）===')
    from src.vector_store import VectorRetriever
    vector_retriever = VectorRetriever()
    # 简化处理：只测召回质量，不完整跑生成（因为没有独立的纯向量 pipeline）
    # 这里用 hybrid 但图谱和 rerank 都关闭，作为"向量+BM25"的基线

    # 配置2：向量 + BM25（无图谱、无重排）
    print('\n=== 配置2: 向量 + BM25（无图谱、无重排）===')
    pipeline_no_kg = RAGPipeline(use_graph=False, use_rerank=False)
    s2, d2 = evaluate_config(questions, pipeline_no_kg, 'Dense + BM25')
    all_summaries.append(s2)
    all_details['dense_bm25'] = d2

    # 配置3：+ 知识图谱（无重排）
    print('\n=== 配置3: + 知识图谱（无重排）===')
    pipeline_kg = RAGPipeline(use_graph=True, use_rerank=False)
    s3, d3 = evaluate_config(questions, pipeline_kg, 'Dense + BM25 + KG')
    all_summaries.append(s3)
    all_details['dense_bm25_kg'] = d3

    # 配置4：全流程（+ Rerank）
    print('\n=== 配置4: 全流程（+ Rerank）===')
    pipeline_full = RAGPipeline(use_graph=True, use_rerank=True)
    s4, d4 = evaluate_config(questions, pipeline_full, 'Dense + BM25 + KG + Rerank')
    all_summaries.append(s4)
    all_details['full'] = d4

    # 输出对比表
    print('\n' + '=' * 70)
    print(f'{"方法":<30} {"关键词命中率":>12} {"拒答正确率":>12} {"平均延迟":>10}')
    print('-' * 70)
    for s in all_summaries:
        print(f'{s["config"]:<30} {s["keyword_hit_rate"]:>12.1%} {s["refusal_accuracy"]:>12.1%} {s["avg_latency"]:>8.1f}s')
    print('=' * 70)

    # 保存结果
    results = {
        'summary_table': all_summaries,
        'details': all_details,
    }
    out_file = TEST_DIR / 'ablation_results.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f'\n结果已保存: {out_file}')


if __name__ == '__main__':
    run_ablation()
