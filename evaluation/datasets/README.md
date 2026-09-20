# 评估数据集

`qa_eval.json` 是一套随仓库分发的、脱敏精简的 RAG 评估集（共 12 题），用于复现
六组消融实验（`python -m evaluation.run_ablation`）。题目均围绕语料《陈嘉庚传》，
不含任何隐私或密钥信息。

## 字段说明

| 字段 | 含义 |
| --- | --- |
| `id` | 题目编号 |
| `question` | 用户问题 |
| `type` | `fact` / `relation` / `overview` / `refusal` |
| `category` | 中文题型 |
| `answer_keywords` | 参考答案应包含的关键词（事实/关系/概述题） |
| `note` | 备注（拒答题说明语料中无答案） |
| `relevant_chunks` | **可选**：人工标注的相关 `chunk_id` 列表 |

## Ground Truth 口径

检索指标（Recall@5 / Recall@10 / MRR）支持两种口径，由 `evaluation/metrics.py`
自动选择：

1. **chunk 级标准口径**：当题目提供 `relevant_chunks`（相关 `chunk_id` 列表）时，
   按信息检索标准定义计算。
2. **keyword-proxy 口径（当前默认）**：当前数据集尚未做 chunk 级人工标注，
   退而用 `answer_keywords` 是否出现在 top-K chunk 文本中作为弱监督信号：
   - Recall@K = top-K 覆盖到的答案关键词比例；
   - MRR = 首个命中任意答案关键词的 chunk 的排名倒数。

   该口径确定性、可复现、无需额外标注，但**不等同于人工 chunk 标注**，结果 JSON
   中以 `ground_truth: "keyword_proxy"` 标明。后续若补齐 `relevant_chunks` 字段，
   评估代码会自动切换到标准口径，无需改动。

拒答题（`type=refusal`）不计入检索 Recall/MRR，单独统计拒答成功率
（答案是否包含“未找到 / 无法确定 / 语料中没有”等拒答信号）。

## 生成指标

- 非拒答题：Answer Keyword Accuracy = 答案命中的 `answer_keywords` 比例；
- 拒答题：正确拒答记 1，否则记 0。

这是确定性的关键词口径，不依赖 LLM 裁判，保证可复现；如需 LLM-as-Judge 的
Faithfulness/Completeness 评分，可使用 `src/evaluator.py`（需配置模型 API）。
