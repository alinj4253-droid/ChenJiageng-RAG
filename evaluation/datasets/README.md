# 第一阶段数据集

本轮只跑 MultiHop-RAG 完整系统；CRUD 加载器是历史兼容代码，本轮不运行。
内置 qa_eval.json 的 12 题只是领域冒烟集，没有 chunk 级人工 Ground Truth。

## 官方来源

- [MultiHop-RAG 官方仓库](https://github.com/yixuantt/MultiHop-RAG)
- [官方数据与 corpus.json](https://huggingface.co/datasets/yixuantt/MultiHopRAG/tree/main)

`python -m evaluation.download_benchmarks --dataset multihop-rag` 下载题目与独立语料。
只使用 corpus 中的文章正文和元数据构建知识库，不使用 `evidence_list.fact` 或标准答案作为语料。
下载验证 JSON，失败返回非零状态，已损坏缓存不能被当作成功跳过。

## 准备与构建

项目 README 给出了完整 PowerShell 命令。第一阶段为全部 609 篇文章、随机种子 42、100 题。
`--limit` 只限制题目，不减少语料；不要使用 `--distractors` 运行正式结果，该参数只供缩小语料冒烟。

产物：

- `chunks/chunks.jsonl`：600 字符窗口，80 字符重叠，与当前系统预算一致。
- `questions.json`：问题、答案、题型及原始官方 supporting facts。
- `chunk_mapping.json`：`chunk_id`、`document_id`、`source_id`、`benchmark_evidence_id`。
- `benchmark_manifest.json`：原始数据与产物哈希、固定种子、语料范围、数量与文档映射。
- `vector_store/`、`kg/`：独立的 Dense、Entity、Relation FAISS 索引；BM25 在初始化时基于相同 chunks 构建。

官方文件未提供统一的数值文档/证据 ID，因此 document_id 是来源 URL 的稳定 SHA-256 标识，
source_id 保留原始 URL（缺失时使用原始 title），evidence_id 是来源与原始 fact 的稳定派生标识。
这些 ID 不是冒充官方数值 ID。benchmark_evidence_id 是实际包含在该 chunk 文本中的事实 ID 列表，
只用于离线审计，不送入 embedding、Retriever、Router 或生成 Prompt。

用 `RAG_DATA_DIR` 隔离 corpus、KG、向量文件，共享项目模型目录。运行入口验证哈希和 metadata
一致性，缺图谱索引、语料不一致或精排模型不可用时拒绝生成伪完整系统结果。

## 三项主指标

1. Recall@10：最后一次检索的 RRF 候选 Top 10 覆盖的 gold facts 数 / gold facts 总数。
   沿用官方检索脚本的去空格和换行后逐 chunk 子串匹配，同一 fact 仅计一次。
   跨边界被切断的事实不算命中；不会因为命中文档就放宽评分。null_query 不计召回。
2. F1：英文 lower-case、去 ASCII 标点和冠词后的词元重合，多重集合处理重复词。
   不额外调用 LLM 打分。与官方脚本当前的任意词重合成功率不同，不能直接比较官方同名数值。
3. Latency：perf_counter 实测 query 进入至答案完成；排除组件初始化与评测计算，清除 query cache。

辅助诊断只记录 Retrieval Calls 与 Retry Rate。不计算 MRR、nDCG、Faithfulness、Correctness
或 Judge FP/FN。无标准证据时召回为 null，不记作 0。输出答案为空时终止，避免把服务故障记成质量。

正式入口：`python -m evaluation.run_benchmark --data-dir outputs/benchmarks/multihop-full-100`。
该入口只构建完整动态 Pipeline，最多一次重试，没有 profile 循环。
第一版结果、成功/失败案例与分析见项目根目录 BENCHMARK_EVALUATION.md。

正式评测逐题原子落盘到 results/checkpoint.json；网络故障后可追加 `--resume` 继续。
恢复会校验代码、数据、问题数量及问题顺序，配置或输入变化时拒绝混合结果。
