# 评估数据集

## 内置数据集

`qa_eval.json` 是一套随仓库分发的、脱敏精简的 RAG 评估集（共 12 题），用于复现
七组单变量消融实验（`python -m evaluation.run_ablation`）。题目均围绕陈嘉庚领域语料，
不含任何隐私或密钥信息。

### 字段说明

| 字段 | 含义 |
| --- | --- |
| `id` | 题目编号 |
| `question` | 用户问题 |
| `type` | `fact` / `relation` / `overview` / `refusal` |
| `category` | 中文题型 |
| `answer_keywords` | 参考答案应包含的关键词（事实/关系/概述题） |
| `ground_truth` | **可选**：标准答案文本，用于 LLM-as-Judge Answer Correctness |
| `relevant_chunks` | **可选**：人工标注的相关 `chunk_id` 列表，用于标准 Recall/MRR/nDCG |
| `note` | 备注（拒答题说明语料中无答案） |

## 公开 Benchmark

除内置数据集外，系统支持接入公开 RAG 评测集，真正衡量系统效果。

### 支持的数据集

| 数据集 | 说明 | 下载 |
| --- | --- | --- |
| **CRUD-RAG** | 面向 RAG 系统增删改查（Create/Read/Update/Delete）操作的评测集 | `python -m evaluation.download_benchmarks --dataset crud-rag` |
| **MultiHop-RAG** | 多跳推理 RAG 评测集，含 supporting_facts 标注 | `python -m evaluation.download_benchmarks --dataset multihop-rag` |

下载后的数据集保存在 `evaluation/datasets/public/` 目录。

### 在公开 Benchmark 上运行

```bash
# 下载数据集
python -m evaluation.download_benchmarks

# 在 CRUD-RAG 上运行消融（自动识别格式）
python -m evaluation.run_ablation --dataset evaluation/datasets/public/crud_rag.json --benchmark

# 指定格式
python -m evaluation.run_ablation --dataset evaluation/datasets/public/multihop_rag.json --benchmark --benchmark-format multihop-rag

# 启用 LLM-as-Judge（Faithfulness + Answer Correctness）
python -m evaluation.run_ablation --dataset evaluation/datasets/public/crud_rag.json --benchmark --llm-judge
```

公开 Benchmark 加载器（`evaluation/benchmark_loader.py`）会自动将各数据集的字段
映射到内部统一格式，包括 `ground_truth`（标准答案）和 `relevant_chunks`（相关文档）。

## 评估指标体系

### 检索侧指标

| 指标 | 口径 | 说明 |
| --- | --- | --- |
| Recall@5 / Recall@10 | chunk 级 或 keyword-proxy | top-K 检索结果对相关文档/答案关键词的覆盖率 |
| MRR | chunk 级 或 keyword-proxy | 首个相关结果的排名倒数 |
| nDCG@5 / nDCG@10 | chunk 级 或 keyword-proxy | 归一化折损累计增益，考虑排序质量 |

**Ground Truth 口径自动切换**：
1. **chunk 级标准口径**：当题目提供 `relevant_chunks` 时，按信息检索标准定义计算；
2. **keyword-proxy 口径（内置数据集默认）**：用 `answer_keywords` 是否出现在 top-K chunk
   文本中作为弱监督信号。该口径确定性可复现，但不等同于人工标注，结果中以
   `ground_truth: "keyword_proxy"` 标明。

### 生成侧指标

| 指标 | 说明 |
| --- | --- |
| Answer Keyword Accuracy | 答案命中 `answer_keywords` 的比例（确定性，不依赖 LLM） |
| Refusal Success Rate | 拒答题是否正确拒答（确定性） |
| **Faithfulness**（LLM-Judge，可选） | 答案事实是否全部被检索上下文支撑（无幻觉） |
| **Answer Correctness**（LLM-Judge，可选） | 答案与标准答案的语义相似度 |

启用 LLM-Judge：`python -m evaluation.run_ablation --llm-judge`

### 系统侧指标

| 指标 | 说明 |
| --- | --- |
| Avg Latency | 单题端到端耗时 |
| Avg Retrieval Rounds | 检索轮次（含重试） |
| Avg Retrieval Calls | 实际 Retriever 调用总次数（dense/bm25/entity_graph/relation_graph 分别计数） |
| Retry Rate | 证据不足触发补检的题目比例 |

### Evidence Judge 校准（FP/FN 统计）

当 profile 启用 Evidence Judge（如 G 组）时，自动统计裁判的误判情况：

| 指标 | 说明 |
| --- | --- |
| TP | 判充分且答案正确 |
| FP | 判充分但答案错误（误报，过于乐观） |
| FN | 判不足但检索实际有证据（漏报，过于保守） |
| TN | 判不足且检索确实无证据 |
| FP Rate / FN Rate | 误报率 / 漏报率 |
| Precision / Recall | 裁判的精确率 / 召回率 |

> 注：FP/FN 基于 proxy 指标（accuracy / recall）近似判定，非严格人工标注。
> 接入带 ground_truth 的公开 Benchmark 后，校准结果更可靠。

## 运行消融实验

```bash
# 全部七组
python -m evaluation.run_ablation

# 只跑某一组
python -m evaluation.run_ablation --only agent_retry

# 只跑前 N 题（调试）
python -m evaluation.run_ablation --limit 3
```

结果写入 `evaluation/results/<key>.json`，汇总写入 `evaluation/results/summary_table.json`。
