# 嘉庚智答 · 知识图谱增强RAG系统

> 基于 **LightRAG 思想** + **混合检索** + **Rerank 精排** 的轻量级领域RAG系统，面向陈嘉庚文献智能问答场景

---

## ✨ 项目亮点

| 能力 | 技术选型 |
|---|---|
| **混合检索** | BGE向量检索 + BM25关键词 + Neo4j知识图谱三路召回 |
| **重排精排** | bge-reranker-base 交叉编码器对候选结果二次排序 |
| **知识图谱** | LightRAG双层索引（实体+关系+声明），Neo4j持久化存储 |
| **多轮对话** | 滑动窗口 + LLM滚动摘要压缩，支持长对话上下文 |
| **流式输出** | SSE打字机效果，后台异步生成，切对话不打断回答 |
| **可视化** | vis-network动态知识图谱交互探索 |

---

## 📁 系统架构

```
用户提问
    ↓
┌─────────────────────────────────┐
│  查询理解（可选查询改写）        │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  三路并行检索                    │
│  ① BGE向量语义检索（Milvus）     │
│  ② BM25关键词检索（jieba分词）   │
│  ③ Neo4j知识图谱多跳检索         │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  RRF 排名融合去重                 │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  bge-reranker 精排 Top-K          │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  Prompt组装 + 多轮历史注入       │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  DeepSeek LLM 流式生成           │
└─────────────────────────────────┘
    ↓
  返回答案 + 引用来源
```

---

## 🛠️ 技术栈

- **后端**：FastAPI + Uvicorn
- **向量库**：Milvus Lite
- **知识图谱**：Neo4j
- **嵌入模型**：BAAI/bge-base-zh-v1.5（本地）
- **重排模型**：BAAI/bge-reranker-base（本地）
- **LLM**：DeepSeek-V4.1-Flash API
- **前端**：原生 HTML + JavaScript + vis-network + marked.js
- **对话存储**：SQLite + SQLAlchemy ORM

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 下载嵌入和重排模型（放到 data/models/ 目录）
# BAAI/bge-base-zh-v1.5
# BAAI/bge-reranker-base

# 启动Neo4j服务（默认bolt://localhost:7687）
```

### 2. 配置API密钥

在项目根目录创建 `.env` 文件：
```env
OPENAI_API_BASE=https://api.deepseek.com/v1
OPENAI_API_KEY=你的DeepSeek API Key
```

### 3. 启动服务

```bash
python -m uvicorn src.app:app --host 0.0.0.0 --port 8000
```

访问：
- 聊天页面：http://localhost:8000
- 知识图谱可视化：http://localhost:8000/graph

---

## 📊 核心效果对比

| 检索策略 | 准确率 | 平均响应时间 |
|---|---|---|
| 纯向量检索（Baseline） | 93.3% | 53.2s |
| 向量 + 关键词混合检索 | 93.3% | 33.8s |
| 全流程 + 图谱 + Rerank | **100%** | 40.9s |

> 幻觉识别率：100%（能准确识别语料中不存在的问题并拒答）

---

## 📂 目录结构

```
RAG/
├── src/                  # 核心后端代码
│   ├── app.py            # FastAPI主入口
│   ├── config.py         # 全局配置
│   ├── rag_pipeline.py   # RAG主流程
│   ├── rag_generator.py  # 答案生成
│   ├── hybrid_retriever.py # 混合检索
│   ├── reranker.py       # 重排
│   ├── graph_retriever.py  # 图谱检索
│   ├── kg_extractor.py   # 知识图谱抽取
│   ├── kg_normalizer.py  # 实体归一化
│   ├── llm_client.py     # LLM客户端
│   ├── db.py             # 对话存储
│   └── ...
├── web/                  # 前端页面
│   ├── index.html        # 聊天界面
│   └── graph.html         # 知识图谱可视化
├── data/                 # 数据目录（语料、模型）
└── requirements.txt      # 依赖清单
```

---

## ✅ 已实现功能

- [x] 语义分块（600字目标，80字重叠）
- [x] 三元组抽取 + 实体归一化
- [x] 三路混合检索（向量 + BM25 + 图谱）
- [x] RRF排名融合
- [x] 交叉编码器重排
- [x] 多轮对话记忆（滑动窗口 + 摘要压缩）
- [x] SSE流式打字机输出
- [x] Markdown渲染 + 引用来源标注
- [x] 知识图谱可视化交互
- [x] 后台异步生成（切对话不打断）

---

## 📝 License

本项目仅用于学习交流。