"""
全局配置：模型、路径、超参数
"""
import os
from pathlib import Path

# ============ 项目路径 ============
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CLEAN_DIR = DATA_DIR / "clean"
RAW_DIR = DATA_DIR / "raw"
CHUNKS_DIR = DATA_DIR / "chunks"
KG_DIR = DATA_DIR / "kg"
VECTOR_DIR = DATA_DIR / "vector_store"
MODELS_DIR = DATA_DIR / "models"
TEST_DIR = DATA_DIR / "test"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"

# 确保目录存在
for d in [CHUNKS_DIR, KG_DIR, VECTOR_DIR, MODELS_DIR, TEST_DIR, OUTPUTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============ API 配置 ============
# 从 .env 读取（简单实现，不依赖 python-dotenv）
_env_file = PROJECT_ROOT / ".env"
if _env_file.exists():
    for line in _env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

# ============ 模型池（DeepSeek API，统一用DeepSeek）============
# DeepSeek官方模型，质量高速度快
PREMIUM_MODELS = [
    "deepseek-chat",  # DeepSeek-V4.1-Flash，主力模型
    "deepseek-reasoner",  # 推理模型，备用
]

# 快速模型（用于简单任务）
FAST_MODELS = [
    "deepseek-chat",
]

ALL_ONLINE_MODELS = PREMIUM_MODELS + FAST_MODELS

# 默认模型选择策略（全部用DeepSeek，质量速度都好）
DEFAULT_CHAT_MODEL = "deepseek-chat"        # 问答生成
DEFAULT_EXTRACT_MODEL = "deepseek-chat"     # 三元组抽取
DEFAULT_EVAL_MODEL = "deepseek-chat"        # 评估

# ============ 切块配置 ============
CHUNK_TARGET_SIZE = 600       # 目标块大小（字）
CHUNK_MAX_SIZE = 800          # 最大块大小
CHUNK_OVERLAP = 80            # 块间重叠（字）
CHUNK_MIN_SIZE = 100          # 最小块大小（小于此值合并到前一块）

# ============ 检索配置 ============
# 三级 Top-K 拓扑（逐级收敛）：
#   1. VECTOR_TOP_K / BM25_TOP_K / KG_*_TOP_K → 各路检索器原始召回
#   2. RERANK_TOP_N → Weighted RRF 融合后保留的候选数（送入 Cross-Encoder 精排）
#   3. FINAL_TOP_K  → Rerank 后保留的证据块数（Evidence Judge 与 Generation 的输入上限）
#   4. MAX_CONTEXT_CHUNKS → 生成阶段实际送入 LLM 的 chunk 数（≤ FINAL_TOP_K）
VECTOR_TOP_K = 20             # 向量检索候选数
BM25_TOP_K = 20               # BM25检索候选数
HYBRID_TOP_K = 30             # 混合检索最终候选数
RRF_K = 60                    # RRF融合常数
KG_ENTITY_TOP_K = 10          # 图谱实体检索候选数
KG_RELATION_TOP_K = 10        # 图谱关系检索候选数
RERANK_TOP_N = 10             # RRF融合后保留的候选数（送入精排）
FINAL_TOP_K = 5               # Rerank后保留的证据块数（Evidence Judge / Generation 输入上限）

# 融合权重（两路图谱检索是独立来源，分别配置权重，即便当前数值相同）
VECTOR_WEIGHT = 0.5
BM25_WEIGHT = 0.3
ENTITY_GRAPH_WEIGHT = 0.2   # 实体图谱检索（双向子图扩展）在 Weighted RRF 中的权重
RELATION_GRAPH_WEIGHT = 0.2  # 关系向量检索在 Weighted RRF 中的权重

# ============ 嵌入模型 ============
EMBED_MODEL_NAME = "BAAI/bge-base-zh-v1.5"
EMBED_DIM = 768
EMBED_MODEL_PATH = str(MODELS_DIR / "bge-base-zh-v1.5")

# ============ 分词与停用词 ============
STOPWORDS_FILE = DATA_DIR / "stopwords.txt"
CUSTOM_DICT_FILE = DATA_DIR / "userdict.txt"

# ============ 重排模型 ============
RERANK_MODEL_NAME = "BAAI/bge-reranker-base"
RERANK_MODEL_PATH = str(MODELS_DIR / "models" / "BAAI--bge-reranker-base" / "snapshots" / "master")
# 精排后保留数由 FINAL_TOP_K 统一控制（见检索配置区），不再单独设 RERANK_TOP_K

# ============ 生成配置 ============
GEN_TEMPERATURE = 0.2
GEN_MAX_TOKENS = 1024
GEN_TOP_P = 0.9

# ============ 上下文组装配置 ============
# FINAL_TOP_K（精排输出）≥ MAX_CONTEXT_CHUNKS（生成实际使用）；
# Evidence Judge 看全部 FINAL_TOP_K 个证据，生成时只取排名最高的 MAX_CONTEXT_CHUNKS 个。
MAX_CONTEXT_CHUNKS = 3          # 生成阶段实际送入LLM的chunk数
MAX_CONTEXT_CHARS = 1200        # 上下文最大总字数（按句子边界截断）

# ============ 图谱抽取配置 ============
KG_MAX_ENTITIES_PER_CHUNK = 10
KG_MAX_RELATIONS_PER_CHUNK = 15

# ============ 图谱检索配置 ============
GRAPH_TOP_K = 10              # 图谱检索返回chunk数
GRAPH_EXPAND_DEPTH = 2        # 子图扩展深度
GRAPH_ENTITY_SIM_THRESHOLD = 0.5    # 实体向量匹配相似度阈值
GRAPH_RELATION_SIM_THRESHOLD = 0.5  # 关系向量检索相似度阈值

# ============ Agentic RAG 配置 ============
ENABLE_AGENT_ROUTING = True       # 是否启用查询模式路由（关闭则所有查询走全量检索）
ENABLE_EVIDENCE_JUDGE = True      # 是否启用证据充分性判断
ENABLE_RELATION_RETRIEVAL = True  # 是否启用关系检索（global/hybrid 模式）
MAX_RETRIEVAL_RETRY = 1           # 证据不足时最多重试检索次数（第一版固定 1 次）
RECENT_WINDOW = 6                 # 对话短期记忆窗口（消息条数）

# ============ 可观测性：结构化执行 trace ============
ENABLE_TRACE_LOG = True           # 是否把每次 query 的执行轨迹写入 JSONL
TRACE_LOG_FILE = LOGS_DIR / "rag_trace.jsonl"

# ============ Neo4j 连接配置（从 .env 读取，不要硬编码） ============
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")
