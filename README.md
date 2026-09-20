# 嘉庚智答 · 知识图谱增强的 Agentic RAG 系统

> 面向陈嘉庚领域文献的 **Knowledge-Graph Enhanced Agentic RAG** 问答系统。
> 系统以**固定 Workflow** 保证整体执行可控，仅在**检索路由**、**证据充分性裁判**、
> **证据不足时的有限查询改写重试**三个关键节点引入 LLM 决策（Workflow-first，
> 不使用 Multi-Agent / LangGraph 编排）。

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![tests](https://img.shields.io/badge/tests-115%20passed-success.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#)

---

## 📖 项目简介

“嘉庚智答”面向**陈嘉庚文献语料库**（《南侨回忆录》《陈嘉庚言论集》《新中国观感集》等），
将文献构建为可检索的向量索引、BM25 索引与实体-关系知识图谱，结合多路混合检索、
Cross-Encoder 精排与三个轻量 Agentic 决策点，提供**有引用、可溯源、低幻觉**的问答服务。

核心链路：

```
Query Analysis
  → Retrieval Routing（Agentic 决策点 1：按 query_mode 选择检索器组合）
  → Dense / BM25 / Graph（实体 + 关系）检索
  → Weighted RRF 融合
  → Cross-Encoder Rerank
  → Evidence Judge（Agentic 决策点 2：证据是否足以回答原问题）
  → （不足且未超上限）Query Rewrite & Bounded Retry（Agentic 决策点 3）
  → Generation（证据不足时谨慎作答，不编造）
```

---

## ✨ 核心特性

| 能力 | 技术实现 |
|---|---|
| **Retrieval Router** | 规则 + LLM 把问题分为 `naive / local / global / hybrid` 四类，输出显式 `RetrievalPlan` 决定启用哪些检索器与是否精排/裁判；短问题（<15 字）跳过 LLM 直接走轻量检索以降延迟 |
| **多路混合检索** | BGE 向量语义检索（FAISS）+ BM25 关键词检索（jieba 分词）+ 知识图谱实体检索，按 Plan 动态组合 |
| **图谱实体检索** | 实体精确/别名匹配 + 向量匹配，**双向邻接表**两跳权重衰减 BFS 子图扩展（边带 `outgoing/incoming` 方向，不丢失关系语义） |
| **图谱关系检索** | 对 `global / hybrid` 问题，用 high-level 关键词检索 **relation 向量索引**（FAISS），经三元组 `source_chunks` 召回关系语义证据 |
| **Weighted RRF 融合** | 任意多路召回用带权 Reciprocal Rank Fusion 去重融合，向量/BM25/图谱权重可配 |
| **重排精排** | bge-reranker-base Cross-Encoder 对融合候选二次排序 |
| **Evidence Judge** | 生成前用 LLM 对照原问题裁判证据是否充分、缺哪些信息点；空证据直接判不足且不调用 LLM，裁判失败 fail-open 不阻塞 |
| **Bounded Query Rewrite + Retry** | 证据不足时按缺失点改写检索 query 并补检，**上限 1 次**；裁判始终针对原始问题；改写结果与历史重复立即停止，杜绝死循环 |
| **谨慎生成** | 重试后证据仍不足时，向生成器注入“只依据材料、不编造、无据明说”的约束 |
| **增量滚动摘要记忆** | 仅对滑出近期窗口（6 条）的旧消息做一次摘要，用 `summarized_until_message_id` 游标保证每条消息只摘要一次，避免重复摘要 |
| **结构化 Trace** | 每次 query 以 JSONL 记录 query_mode、检索 Plan、候选/精排文档数、证据是否充分、重试次数与各阶段延迟（`logs/rag_trace.jsonl`） |
| **消融评估** | 一键运行 6 组消融（Dense / +BM25 / +KG / +Rerank / Agent Router / Agent Router+Retry），输出 Recall@K、MRR、Answer Accuracy、Latency、Retrieval Calls、Retry Rate |
| **流式输出** | SSE 打字机效果，后台异步生成，实时推送“检索 / 裁判 / 改写重试”状态 |
| **引用溯源 / 抗幻觉** | 答案附书名、章节与原文片段；对语料外问题礼貌拒答 |

---

## 🏗️ 系统架构

```
                          用户提问
                            │
                            ▼
                  ┌───────────────────┐
                  │  Query Analysis   │  短问题(<15字)跳过LLM → naive
                  └───────────────────┘
                            │
                            ▼
              ┌──────────────────────────┐
              │   Retrieval Router       │  Agentic ①  naive/local/global/hybrid
              │   → RetrievalPlan        │
              └──────────────────────────┘
                            │
        ┌───────────┬───────┴────────┬──────────────┐
        ▼           ▼                ▼              ▼
  ┌──────────┐ ┌────────┐  ┌────────────────┐ ┌──────────────┐
  │ Dense    │ │ BM25   │  │ Graph 实体     │ │ Graph 关系   │
  │ BGE+FAISS│ │ jieba  │  │ 双向两跳BFS    │ │ relation索引 │
  └──────────┘ └────────┘  └────────────────┘ └──────────────┘
        └───────────┴───────┴────────┴──────────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │ Weighted RRF 融合  │  向量0.5/BM25 0.3/图谱0.2
                  └───────────────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │ bge-reranker 精排 │  Cross-Encoder
                  └───────────────────┘
                            │
                            ▼
              ┌──────────────────────────┐
              │     Evidence Judge       │  Agentic ② 对照原问题裁判
              └──────────────────────────┘
                            │
        证据不足且未超上限（≤1 次）│ 充分 / 达上限
                            ▼
              ┌──────────────────────────┐
              │ Query Rewrite + 补检     │  Agentic ③（Bounded，防死循环）
              └──────────────────────────┘
                            │  （回到融合后再次裁判）
                            ▼
                  ┌───────────────────┐
                  │ Prompt 组装       │  长期摘要 + 近期对话
                  │ + 不足时谨慎约束  │
                  └───────────────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │ DeepSeek LLM 流式 │  SSE 输出
                  └───────────────────┘
                            │
                            ▼
                     答案 + 引用来源 + Trace
```

### 离线索引构建流程

```
原始文献 (PDF/TXT)
    │
    ▼
文本清洗与解析 → 语义分块 (600字目标, 80字重叠)
    │
    ├──► BGE 向量化 ──► FAISS 向量索引
    │
    └──► LLM 三元组抽取 ──► 实体归一化
              ├──► 实体 / 关系 FAISS 索引（在线检索使用）
              └──► 导入 Neo4j（持久化与可视化）
```

> ⚠️ **在线问答检索不依赖 Neo4j 在线服务**：在线检索直接读取本地
> `entities.jsonl / triples.jsonl` 与实体/关系 FAISS 索引在 Python 内存中完成；
> Neo4j 仅用于图谱持久化存储与 `web/graph.html` 可视化。未启动 Neo4j 不影响问答。

---

## 🛠️ 技术栈

| 层次 | 选型 |
|---|---|
| **后端框架** | FastAPI + Uvicorn |
| **向量检索** | FAISS（IndexFlatIP 余弦相似度） |
| **嵌入模型** | BAAI/bge-base-zh-v1.5（本地加载，进程内 lazy 单例缓存） |
| **重排模型** | BAAI/bge-reranker-base（本地 Cross-Encoder） |
| **关键词检索** | BM25 + jieba 中文分词 |
| **知识图谱** | 实体/关系三元组 + 实体与关系双 FAISS 索引；Neo4j 5.x 仅用于持久化与可视化 |
| **LLM** | DeepSeek-Chat API（OpenAI 兼容接口，支持 Ollama 降级） |
| **对话存储** | SQLite + SQLAlchemy ORM |
| **前端** | 原生 HTML / JavaScript + vis-network + marked.js + DOMPurify |
| **流式传输** | SSE（Server-Sent Events） |
| **测试 / 评估** | pytest（全部 mock，不依赖真实模型/API）；内置 6 组消融脚本 |

---

## 📁 项目结构

```
ChenJiageng-RAG/
├── src/                        # 后端核心代码
│   ├── app.py                  # FastAPI 主入口（路由 + SSE 流式）
│   ├── config.py               # 全局配置（路径、模型、超参数、Agent 开关）
│   ├── rag_pipeline.py         # RAG 主流水线：路由→检索→裁判→有限重试→生成
│   ├── query_analyzer.py       # 查询分析（关键词 + query_mode）
│   ├── retrieval_router.py     # 【Agentic ①】检索路由：RetrievalPlan + 规则选路
│   ├── evidence_judge.py       # 【Agentic ②】证据充分性裁判
│   ├── query_rewriter.py       # 【Agentic ③】证据不足时的查询改写
│   ├── memory_manager.py       # 增量滚动摘要（RollingSummaryManager）
│   ├── trace_logger.py         # 结构化执行 trace（JSONL）
│   ├── hybrid_retriever.py     # 向量 + BM25 混合检索
│   ├── vector_store.py         # FAISS 向量索引与检索
│   ├── bm25_store.py           # BM25 索引与检索
│   ├── graph_retriever.py      # 图谱实体/关系检索、双向子图扩展
│   ├── reranker.py             # Cross-Encoder 重排
│   ├── rag_generator.py        # 答案生成（同步 + 流式，支持谨慎作答）
│   ├── llm_client.py           # LLM 客户端（DeepSeek / Ollama 降级）
│   ├── chunker.py              # 文档语义分块
│   ├── kg_extractor.py         # 知识图谱三元组抽取
│   ├── kg_normalizer.py        # 实体归一化消歧
│   ├── kg_indexer.py           # 图谱实体/关系向量索引
│   ├── neo4j_importer.py       # 三元组批量导入 Neo4j
│   ├── db.py                   # 会话与消息存储（SQLAlchemy）
│   ├── prompts.py              # 各类 Prompt 模板
│   └── evaluator.py            # LLM-as-Judge 评估（可选）
├── evaluation/                 # 端到端消融评估
│   ├── datasets/qa_eval.json   # 随仓库分发的 12 题精简评估集
│   ├── profiles.py             # 6 组消融配置
│   ├── metrics.py             # Recall@K / MRR / Answer Accuracy 指标
│   ├── run_ablation.py         # 一键消融入口
│   └── results/                # 消融结果 JSON（运行后生成）
├── tests/                      # pytest 测试（LLM/检索全部 mock）
├── web/                        # 前端页面（聊天 / 图谱可视化）
├── .env.example                # 环境变量模板（复制为 .env 后填写）
├── requirements.txt
└── README.md
```

> 📌 `data/models/`、`data/chunks/`、`data/kg/`、`data/vector_store/`、
> `data/raw/`、`data/clean/`、`logs/`、`.env` 等已通过 `.gitignore` 排除。

---

## 🚀 快速开始

### 环境要求

- Python 3.9+
- 一个 DeepSeek API Key（或本地 Ollama）
- Neo4j 5.x（**仅图谱可视化/持久化需要，不影响在线问答**）

### 1. 克隆仓库并安装依赖

```bash
git clone https://github.com/alinj4253-droid/ChenJiageng-RAG.git
cd ChenJiageng-RAG
pip install -r requirements.txt
```

### 2. 下载本地模型

嵌入与重排模型需手动下载后放入 `data/models/`：

```bash
huggingface-cli download BAAI/bge-base-zh-v1.5 --local-dir data/models/models/BAAI--bge-base-zh-v1.5/snapshots/master
huggingface-cli download BAAI/bge-reranker-base --local-dir data/models/models/BAAI--bge-reranker-base/snapshots/master
```

### 3. 配置环境变量

```bash
cp .env.example .env   # Windows: copy .env.example .env
```

编辑 `.env`，填入 DeepSeek API Key（Neo4j 仅在需要可视化时填写）：

```env
OPENAI_API_BASE=https://api.deepseek.com/v1
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

### 4. 准备数据与索引

将文献语料放入 `data/raw/`，依次运行：

```bash
python -m src.chunker          # 文本分块
python -m src.vector_store     # 向量索引
python -m src.kg_extractor     # 三元组抽取
python -m src.kg_normalizer    # 实体归一化
python -m src.kg_indexer       # 实体/关系向量索引
python -m src.neo4j_importer   # 导入 Neo4j（可选，仅可视化需要）
```

### 5. 启动服务

```bash
python -m uvicorn src.app:app --host 0.0.0.0 --port 8000
```

| 页面 | 地址 |
|---|---|
| 聊天问答 | http://localhost:8000 |
| 知识图谱可视化 | http://localhost:8000/graph |
| API 文档 | http://localhost:8000/docs |

---

## 🔌 主要 API

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/chat` | 同步问答（返回含 query_mode、retrieval_plan、evidence_sufficient、retry_count、trace） |
| POST | `/api/chat/stream` | SSE 流式问答（推送 status / token / done 事件） |
| GET | `/api/sessions` | 会话列表 |
| GET | `/api/sessions/{id}/messages` | 会话历史消息 |
| POST | `/api/sessions` | 新建会话 |
| DELETE | `/api/sessions/{id}` | 删除会话 |
| GET | `/api/graph?entity=陈嘉庚` | 获取图谱子图数据 |
| GET | `/api/health` | 健康检查 |

---

## 📊 消融评估（可复现）

一键运行 6 组方案对比：

```bash
python -m evaluation.run_ablation
```

可选参数：`--only <key>`（只跑某组）、`--limit N`（只取前 N 题）、
`--dataset PATH`（自定义评估集）。结果写入 `evaluation/results/<key>.json`，
汇总写入 `evaluation/results/summary_table.json`。

六组配置（前四组为固定工作流，后两组为 Agentic）：

| 组 | key | 检索器组合 | Agent 决策 |
|---|---|---|---|
| A | `dense` | 仅 Dense | 无 |
| B | `dense_bm25` | Dense + BM25 | 无 |
| C | `plus_kg` | Dense + BM25 + 图谱实体 | 无 |
| D | `plus_rerank` | Dense + BM25 + 实体 + 关系 + Rerank | 无（最强固定工作流） |
| E | `agent_router` | Router 按 query_mode 动态选路 | 路由（不裁判/不重试） |
| F | `agent_retry` | Router 动态选路 | 路由 + 证据裁判 + 有限重试 |

指标：**Recall@5 / Recall@10 / MRR**（检索，在**精排前的融合候选池（10 个）**上
计算，衡量“检索 + 融合”阶段的召回与排序；Cross-Encoder 精排不增加召回，其价值由
Answer Accuracy 体现）、**Answer Accuracy**（关键词命中与拒答成功率，确定性口径）、
**Avg Latency**、**Avg Retrieval Calls**、**Retry Rate**。

> 评估集 `evaluation/datasets/qa_eval.json` 含 12 题（事实/关系/概述/拒答）。
> 当前检索指标采用 **keyword-proxy 口径**（top-K chunk 文本对答案关键词的覆盖），
> 因为数据集尚未做 chunk 级人工标注；一旦为题目补充 `relevant_chunks` 字段，
> 指标会自动切换到标准 chunk 级 Recall/MRR，详见 `evaluation/datasets/README.md`。

**实测结果（本机、12 题单次运行；以 `evaluation/results/summary_table.json` 为准）：**

<!-- ABLATION_TABLE_START -->
| Method | Recall@5 | Recall@10 | MRR | Accuracy | Avg Latency | Avg Calls | Retry% |
|---|---|---|---|---|---|---|---|
| Dense | 0.742 | 0.808 | 0.812 | 0.556 | 3.89s | 1.00 | 0.0% |
| Dense+BM25 | 0.708 | 0.808 | 0.746 | 0.472 | 3.47s | 1.00 | 0.0% |
| +KG | 0.683 | 0.808 | 0.743 | 0.333 | 3.68s | 1.00 | 0.0% |
| +Rerank | 0.658 | 0.808 | 0.810 | 0.514 | 6.83s | 1.00 | 0.0% |
| Agent Router | 0.717 | 0.808 | 0.760 | 0.451 | 6.32s | 1.00 | 0.0% |
| Agent Router + Retry | **0.700** | 0.792 | **0.863** | 0.514 | 11.56s | 1.58 | 58.3% |
<!-- ABLATION_TABLE_END -->

**结论（如实报告，含不符合预期之处）：**

- **MRR 是 Agentic Retry 最稳定的收益**：Agent Router+Retry 的 MRR=0.863 全场最高，
  比最强固定工作流 +Rerank（0.810）高约 5 个百分点；在两次独立运行中该组 MRR 均为最高
  （0.853 / 0.863）。说明证据不足时的改写补检能把正确证据**排到更靠前的位置**。
- **Recall@10 各组接近（0.79–0.81）**：融合候选池的召回上界相当，说明瓶颈不在“有没有
  召回”，而在“排序是否靠前（MRR）”与“证据是否被判定充分”，这正是裁判 + 重试作用之处。
- **拒答成功率全部为 1.0**：6 组都能正确拒答 2 道语料外问题，抗幻觉底线稳定。
- **成本清晰可观测**：Retry 组平均检索 1.58 次、58.3% 的复杂问题触发补检，平均延迟
  约为 Dense 的 3 倍（3.89s → 11.56s）；重试上界（默认 1 次）使该成本可控。
- **局限（不夸大）**：① 仅 12 题、单次运行，LLM 生成存在温度随机性，Recall@5 与
  Accuracy 的组间差距多在 1 题（≈0.083）以内，不宜过度解读；② 在 keyword-proxy 口径下
  Dense 基线偏强（答案多为高频实体词，纯向量即可命中），BM25/图谱经 RRF 融合后反而小幅
  稀释了 Dense 的前排排名；③ Router“简单问题降延迟”的收益在本数据集不明显（长问题占多数，
  且查询分析本身要调用一次 LLM）。要得到统计显著的结论，需要扩大题库、为题目补充 chunk 级
  `relevant_chunks` 人工标注并多次运行取均值。

---

## ✅ 测试

所有测试用 mock 隔离 LLM、FAISS 与网络，无需 API Key 或模型即可运行：

```bash
python -m pytest -q
```

覆盖：查询分析、检索路由、证据裁判、查询改写、有限重试循环（含死循环防护）、
混合检索融合、图谱双向遍历与关系检索、增量滚动摘要、会话存储、评估指标、trace 落盘等。

---

## ⚙️ Agent 相关配置（`src/config.py`）

| 配置 | 默认 | 说明 |
|---|---|---|
| `ENABLE_AGENT_ROUTING` | `True` | 关闭后所有问题走全量 hybrid（用于消融 Agent Router） |
| `ENABLE_EVIDENCE_JUDGE` | `True` | 关闭证据裁判即同时关闭重试 |
| `MAX_RETRIEVAL_RETRY` | `1` | 证据不足时最多补检次数（构造 Pipeline 时可传 `max_retry=0` 关闭） |
| `ENABLE_RELATION_RETRIEVAL` | `True` | global/hybrid 模式是否启用关系向量检索 |
| `ENABLE_TRACE_LOG` | `True` | 是否写结构化 trace 到 `logs/rag_trace.jsonl` |
| `RECENT_WINDOW` | `6` | 对话短期记忆窗口（消息条数） |

---

## 🧠 关键技术点说明

- **Workflow-first 的三个 Agentic 决策点**：不引入多 Agent 编排，只在路由、证据裁判、
  有限重试三处让 LLM 做决策，其余步骤固定可控。重试有严格上界（默认 1 次），
  且裁判始终针对**原始问题**、重试只替换检索 query（保留原分析与 Plan），
  改写结果重复即停，从机制上杜绝无限循环与成本失控。
- **RetrievalPlan 显式化**：路由输出一个 frozen 的 `RetrievalPlan`（各检索器/精排/
  裁判开关），检索、融合、裁判都只依赖该 Plan，使消融实验只需替换 Plan 或开关，
  无需改业务代码。
- **任意多路 Weighted RRF**：不写死“三路”，按 Plan 实际启用的检索路数做
  `weight / (k + rank + 1)` 累加融合与去重，naive 模式可只融合 Dense+BM25。
- **双向知识图谱 + 关系索引真正生效**：邻接表同时建 `head→tail(outgoing)` 与
  `tail→head(incoming)`，两跳 BFS 可沿任意方向遍历且保留关系方向；
  high-level 抽象问题检索关系 FAISS 索引并经三元组 `source_chunks` 取 chunk，
  让 QueryAnalyzer 的两类关键词（low/high level）各自对应实体与关系检索。
- **增量滚动摘要**：Session 表用 `summarized_until_message_id` 游标记录摘要进度，
  每轮只对“已过游标且滑出近期窗口”的旧消息摘要一次并推进游标，修复了旧实现
  “每轮重复摘要全部历史”的冗余与首条消息无法生成标题的问题。
- **工程可观测**：每次 query 的路由决策、候选/精排文档数、证据充分性、重试次数与
  分阶段延迟以 JSONL 落盘，便于离线统计重试率/延迟分布与 Debug。

---

## 📝 License

本项目仅供学习与交流使用，请勿用于商业用途。文献版权归原作者与出版社所有。


