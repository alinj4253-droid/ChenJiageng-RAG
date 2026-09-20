"""
检索路由器（Retrieval Router）

Agentic 决策点之一：LLM 在 QueryAnalyzer 中判断 query_mode，
本模块只负责把 query_mode 确定性地映射为 RetrievalPlan。

刻意不在这里再调用一次 LLM——判断已经在 QueryAnalyzer 完成，
规则映射保持可解释、可测试、可控、便于消融。
"""
from dataclasses import dataclass, asdict
from typing import Dict, List


@dataclass(frozen=True)
class RetrievalPlan:
    """一次查询要启用哪些检索 / 后处理能力"""
    use_dense: bool = True
    use_bm25: bool = True
    use_graph: bool = False
    use_relation: bool = False
    use_rerank: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)

    def enabled_retrievers(self) -> List[str]:
        """返回启用的检索器名称（用于日志 / 消融统计）"""
        names = []
        if self.use_dense:
            names.append("dense")
        if self.use_bm25:
            names.append("bm25")
        if self.use_graph:
            names.append("graph")
        if self.use_relation:
            names.append("relation")
        return names


# 各 query_mode 的固定路由策略
_POLICY = {
    # 简单事实查询：稠密 + 关键词即可，不引图谱、不精排，控制延迟与成本
    "naive": RetrievalPlan(
        use_dense=True,
        use_bm25=True,
        use_graph=False,
        use_relation=False,
        use_rerank=False,
    ),
    # 围绕特定实体：稠密 + 实体图谱，精排
    "local": RetrievalPlan(
        use_dense=True,
        use_bm25=False,
        use_graph=True,
        use_relation=False,
        use_rerank=True,
    ),
    # 主题 / 全局问题：稠密 + 实体图谱 + 关系检索，精排
    "global": RetrievalPlan(
        use_dense=True,
        use_bm25=False,
        use_graph=True,
        use_relation=True,
        use_rerank=True,
    ),
    # 复杂混合问题：全部启用
    "hybrid": RetrievalPlan(
        use_dense=True,
        use_bm25=True,
        use_graph=True,
        use_relation=True,
        use_rerank=True,
    ),
}

# 无法识别模式时的兜底：全量检索
_DEFAULT_PLAN = _POLICY["hybrid"]


def build_plan(query_analysis: Dict) -> RetrievalPlan:
    """
    根据 QueryAnalyzer 的输出构建检索计划。

    Args:
        query_analysis: 必须包含 query_mode（naive/local/global/hybrid）

    Returns:
        RetrievalPlan
    """
    mode = (query_analysis or {}).get("query_mode", "hybrid")
    return _POLICY.get(mode, _DEFAULT_PLAN)
