# MultiHop-RAG 第一阶段评测

本轮运行中，尚未生成最终分数。此文件将在 100 题完成后填写真实结果，不使用小样本或 Mock 分数代替。

范围：官方完整 609 篇 corpus、12,478 个 chunk，随机种子 42 抽取 100 题；其中 83 题有 supporting
facts，17 题为 null_query。只运行 Query Analyzer → Router → 动态 Dense/BM25/Graph/Relation → RRF
→ 可选 Reranker → Evidence Judge → 最多一次重试 → Generation，不运行 A–G 或 CRUD。

- Recall@10：最后一轮 RRF Top 10 对官方 evidence_list.fact 的覆盖比例。沿用官方文本归一化匹配规则。
  不是文档召回，也不是最终 Top 5 或生成 Top 3 的召回；null_query 排除召回均值。
- F1：标准答案与完整生成答案的规范化英文词元重合 F1，多重集合计数，按 100 题平均。
  [官方 QA 脚本](https://github.com/yixuantt/MultiHop-RAG/blob/main/qa_evaluate.py)当前使用任意词交集成功率，
  与此 token F1 定义不同，不能把两者的同名数值直接混比。
- Latency：perf_counter 测得单次 query 进入至完整答案生成，seconds/query，初始化与评分计算不计入。
- Retrieval Calls、Retry Rate 只作辅助诊断。不额外计算其它主指标或调用 LLM 评分。

官方参考：[数据与语料](https://huggingface.co/datasets/yixuantt/MultiHopRAG/tree/main)、
[检索评测脚本](https://github.com/yixuantt/MultiHop-RAG/blob/main/retrieval_evaluate.py)。
ID 对齐、切块、隔离与复现命令见 evaluation/datasets/README.md；本地产物位于
outputs/benchmarks/multihop-full-100，结果将写到该目录 results/full_pipeline.json。

字段映射保存在 chunk_mapping.json，source_id 保留官方 URL，document/evidence ID 为稳定派生标识，
证据必须真实出现于 chunk 文本。映射仅供离线核验，不注入检索模型或生成模型。

## 当前限制

BGE 嵌入与重排仍采用项目原有中文模型，上下文仍为最多三块、约 1200 字符，没有为英文数据调参。
题目抽样固定，但模型输出有随机性，单次 100 题不构成统计显著性或模块贡献证明。
正式结果之后再分析成功/失败案例，再决定后续优化；不根据未完成结果修改权重、Judge 或上下文策略。
