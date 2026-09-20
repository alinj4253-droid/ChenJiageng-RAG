"""
RAG评估模块
检索指标（Precision/Recall/F1/MRR）+ 生成质量（LLM-as-Judge）
支持四组消融对比：纯向量 / 向量+BM25 / +图谱 / +Rerank
"""
import json
from pathlib import Path
from typing import List, Dict
from src.config import TEST_DIR


class RAGEvaluator:
    """RAG系统评估器"""

    def __init__(self, pipeline=None):
        self.pipeline = pipeline
        self.results = []

    def load_test_questions(self, filepath: str = None) -> List[Dict]:
        """加载测试问题集"""
        if filepath is None:
            filepath = TEST_DIR / 'eval_questions.json'
        if not Path(filepath).exists():
            print(f'测试问题文件不存在: {filepath}')
            return []
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def evaluate_retrieval(self, question: str, relevant_chunk_ids: List[str]) -> Dict:
        """
        评估检索质量。
        relevant_chunk_ids: 人工标注的相关chunk_id列表
        """
        if self.pipeline is None:
            return {}

        # 执行检索（不生成）
        query_analysis = self.pipeline.query_analyzer.analyze(question)

        hybrid = self.pipeline.hybrid_retriever.search(question, top_k=10)
        retrieved_ids = [c['chunk_id'] for c in hybrid]

        # 计算指标
        relevant_set = set(relevant_chunk_ids)
        retrieved_set = set(retrieved_ids[:5])  # top5

        hit = len(relevant_set & retrieved_set)
        precision = hit / len(retrieved_set) if retrieved_set else 0
        recall = hit / len(relevant_set) if relevant_set else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # MRR
        mrr = 0
        for i, cid in enumerate(retrieved_ids):
            if cid in relevant_set:
                mrr = 1.0 / (i + 1)
                break

        return {
            'question': question,
            'precision@5': precision,
            'recall@5': recall,
            'f1@5': f1,
            'mrr': mrr,
            'retrieved_count': len(retrieved_ids),
        }

    def evaluate_generation(self, question: str, reference_answer: str) -> Dict:
        """
        评估生成质量（用LLM作为裁判）。
        """
        if self.pipeline is None:
            return {}

        result = self.pipeline.query(question)
        generated_answer = result['answer']

        # 用LLM评估答案质量
        from src.llm_client import get_client
        client = get_client()
        eval_prompt = f"""请评估以下RAG系统生成的答案质量，从0-5分打分。

问题：{question}
参考答案：{reference_answer}
生成答案：{generated_answer}

请从以下维度评分：
1. 事实准确性（答案是否与参考资料一致）
2. 完整性（是否覆盖了关键点）
3. 相关性（是否回答了问题）
4. 流畅性（语言是否通顺）

输出JSON格式：{{"accuracy": 分数, "completeness": 分数, "relevance": 分数, "fluency": 分数, "overall": 综合分, "comment": "简短评价"}}"""

        eval_result = client.extract_json(eval_prompt, system_prompt='你是一个严格的答案质量评估专家。')

        return {
            'question': question,
            'generated_answer': generated_answer,
            'reference_answer': reference_answer,
            'evaluation': eval_result,
        }

    def run_full_evaluation(self, test_questions: List[Dict] = None) -> Dict:
        """运行完整评估"""
        if test_questions is None:
            test_questions = self.load_test_questions()

        if not test_questions:
            return {'error': '无测试问题'}

        print(f'开始评估，共 {len(test_questions)} 个问题')

        retrieval_metrics = []
        generation_metrics = []

        for i, q in enumerate(test_questions):
            print(f'[{i+1}/{len(test_questions)}] {q["question"][:30]}...')

            # 检索评估
            if 'relevant_chunks' in q:
                r_metric = self.evaluate_retrieval(q['question'], q['relevant_chunks'])
                retrieval_metrics.append(r_metric)

            # 生成评估
            if 'reference_answer' in q:
                g_metric = self.evaluate_generation(q['question'], q['reference_answer'])
                generation_metrics.append(g_metric)

        # 汇总
        summary = {
            'total_questions': len(test_questions),
            'retrieval': {},
            'generation': {},
        }

        if retrieval_metrics:
            summary['retrieval'] = {
                'avg_precision@5': sum(m['precision@5'] for m in retrieval_metrics) / len(retrieval_metrics),
                'avg_recall@5': sum(m['recall@5'] for m in retrieval_metrics) / len(retrieval_metrics),
                'avg_f1@5': sum(m['f1@5'] for m in retrieval_metrics) / len(retrieval_metrics),
                'avg_mrr': sum(m['mrr'] for m in retrieval_metrics) / len(retrieval_metrics),
            }

        if generation_metrics:
            all_eval = [m['evaluation'] for m in generation_metrics if m.get('evaluation')]
            if all_eval:
                summary['generation'] = {
                    'avg_accuracy': sum(e.get('accuracy', 0) for e in all_eval) / len(all_eval),
                    'avg_completeness': sum(e.get('completeness', 0) for e in all_eval) / len(all_eval),
                    'avg_relevance': sum(e.get('relevance', 0) for e in all_eval) / len(all_eval),
                    'avg_fluency': sum(e.get('fluency', 0) for e in all_eval) / len(all_eval),
                    'avg_overall': sum(e.get('overall', 0) for e in all_eval) / len(all_eval),
                }

        # 保存结果
        TEST_DIR.mkdir(parents=True, exist_ok=True)
        with open(TEST_DIR / 'evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump({'summary': summary, 'details': retrieval_metrics + generation_metrics},
                      f, ensure_ascii=False, indent=2)

        print(f'\n=== 评估完成 ===')
        print(json.dumps(summary, ensure_ascii=False, indent=2))

        return summary


if __name__ == '__main__':
    print('评估模块加载成功')
