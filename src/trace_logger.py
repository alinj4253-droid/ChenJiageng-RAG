"""
结构化执行 Trace 日志（Observability）

每次 query 结束后追加一行 JSON 到 JSONL 文件，记录：
query / query_mode / retrieval_plan / 检索与精排文档数 /
证据是否充分 / 重试次数 / 各阶段 latency。

不依赖任何外部平台（LangSmith / OpenTelemetry 等），便于 Debug、
离线评估与统计 Agent 行为（重试率、延迟分布等）。
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from src.config import TRACE_LOG_FILE, ENABLE_TRACE_LOG


class StructuredTraceLogger:
    """把 RAG 单次执行轨迹以 JSONL 形式追加落盘"""

    def __init__(self, log_file: Optional[Path] = None,
                 enabled: Optional[bool] = None):
        self.log_file = Path(log_file) if log_file else Path(TRACE_LOG_FILE)
        self.enabled = ENABLE_TRACE_LOG if enabled is None else enabled

    def build_trace(self, *, question: str, analysis: Dict, plan,
                    contexts, graph_entities, judgement,
                    retry_count: int, candidate_count: int,
                    latency: Dict) -> Dict:
        """组装一条结构化 trace（与是否落盘解耦，便于单测断言字段）"""
        plan_dict = plan.to_dict() if hasattr(plan, "to_dict") else dict(plan)
        return {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "query": question,
            "query_mode": analysis.get("query_mode"),
            "retrieval_plan": plan_dict,
            "retrieved_docs": candidate_count,
            "reranked_docs": len(contexts),
            "matched_entities": len(graph_entities),
            "evidence_sufficient": (
                judgement.sufficient if judgement is not None else None
            ),
            "retry_count": retry_count,
            "latency": {k: round(float(v), 4) for k, v in latency.items()},
        }

    def log(self, trace: Dict) -> Optional[Path]:
        """追加一条 trace；关闭时直接跳过。返回写入路径（关闭时为 None）"""
        if not self.enabled:
            return None
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")
        return self.log_file
