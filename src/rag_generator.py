"""
RAG问答生成层
上下文组装 + LLM生成 + 引用标注
"""
import json
import re
from typing import List, Dict, Optional

from src.llm_client import get_client
from src.prompts import RAG_SYSTEM, RAG_USER
from src.config import MAX_CONTEXT_CHUNKS, MAX_CONTEXT_CHARS, RECENT_WINDOW


# 句子结束符（中文 + 英文 + 换行）
_SENTENCE_ENDINGS = '。！？；!?;\n'


def _truncate_at_sentence(text: str, max_len: int) -> str:
    """
    在 max_len 预算内，按完整句子边界截断文本。
    从 max_len 位置向前找最近的句子结束符；找不到则回退到逗号/空格，
    再找不到才硬截断并加省略号。
    """
    if len(text) <= max_len:
        return text
    # 向前搜索句子结束符（从 max_len-1 一直到开头，找到最近的完整句子边界）
    cut_pos = -1
    for i in range(max_len - 1, -1, -1):
        if text[i] in _SENTENCE_ENDINGS:
            cut_pos = i + 1  # 保留结束符
            break
    if cut_pos == -1:
        # 回退：找逗号/顿号/空格
        for i in range(max_len - 1, -1, -1):
            if text[i] in '，,、 ':
                cut_pos = i + 1
                break
    if cut_pos == -1:
        # 实在找不到边界，硬截断
        return text[:max_len].rstrip() + '...'
    return text[:cut_pos].rstrip()


# 证据最终不足时追加给生成器的谨慎指令
CAUTION_NOTE = (
    "\n\n注意：检索到的资料可能不足以完整回答该问题。"
    "请只依据上述材料作答，不要编造材料中没有的事实；"
    "若材料确实无法支撑答案，请明确说明现有语料中缺少相关信息。"
)


class RAGGenerator:
    """RAG问答生成器"""

    def __init__(self):
        self.client = get_client()

    def _build_context(self, chunks: List[Dict], max_chunks: int = MAX_CONTEXT_CHUNKS,
                       max_chars: int = MAX_CONTEXT_CHARS) -> str:
        """
        组装检索到的chunk为上下文文本。
        按相关性排序，限制数量和总字数。
        """
        context_parts = []
        total_chars = 0

        for i, chunk in enumerate(chunks[:max_chunks]):
            text = chunk['text']
            if total_chars + len(text) > max_chars:
                # 按完整句子边界截断，避免把一个 chunk 后半部分的关键事实切掉
                remaining = max_chars - total_chars
                # 剩余预算太少（不足 max_chars 的 15% 且 < 30 字）则跳过该 chunk
                if remaining < min(30, max_chars * 0.15):
                    break
                text = _truncate_at_sentence(text, remaining)

            source = f"[{i+1}]"
            book = chunk.get('book', '')
            chapter = chunk.get('chapter', '')
            header = f"{source} 《{book}》{chapter}："

            context_parts.append(header + text)
            total_chars += len(header) + len(text)

        return '\n\n'.join(context_parts)

    def generate(self, query: str, chunks: List[Dict],
                 query_analysis: Optional[Dict] = None,
                 caution: bool = False) -> Dict:
        """
        基于检索结果生成答案。

        Args:
            query: 用户查询
            chunks: 检索到的chunk列表
            query_analysis: 查询分析结果（可选）
            caution: 证据不足时为 True，要求模型谨慎作答、不得编造

        返回: {
            "answer": 生成的答案,
            "references": 引用的chunk列表,
            "context_used": 使用的上下文chunk数,
        }
        """
        if not chunks:
            return {
                'answer': '抱歉，未找到与您问题相关的资料。请尝试换一种问法，或提供更多关键词。',
                'references': [],
                'context_used': 0,
            }

        # 组装上下文
        context = self._build_context(chunks)

        # 生成始终针对用户原始问题；检索侧的 query rewrite 不改变最终提问
        prompt = RAG_USER.format(
            context=context,
            query=query,
        )

        system_content = RAG_SYSTEM + (CAUTION_NOTE if caution else "")

        # 调用LLM
        answer = self.client.chat(
            messages=[
                {'role': 'system', 'content': system_content},
                {'role': 'user', 'content': prompt},
            ],
            temperature=0.3,
            max_tokens=1024,
        )

        # 后处理：清理
        answer = answer.strip()

        # 引用的chunk（去重）
        references = []
        seen_ids = set()
        for c in chunks[:MAX_CONTEXT_CHUNKS]:
            if c['chunk_id'] not in seen_ids:
                seen_ids.add(c['chunk_id'])
                references.append({
                    'chunk_id': c['chunk_id'],
                    'book': c.get('book', ''),
                    'chapter': c.get('chapter', ''),
                    'score': c.get('score', 0),
                    'source': c.get('source', ''),
                })

        return {
            'answer': answer,
            'references': references,
            'context_used': min(len(chunks), MAX_CONTEXT_CHUNKS),
        }

    def generate_stream(self, query: str, chunks: List[Dict],
                        query_analysis: Optional[Dict] = None,
                        history: list = None, caution: bool = False):
        """
        流式生成答案，yield (delta_text, references)
        history: 历史对话列表 [{"role":"user","content":"..."}, ...]
        caution: 证据不足时为 True，要求模型谨慎作答、不得编造
        最后一次yield时references非空。
        """
        if not chunks:
            yield ('抱歉，未找到与您问题相关的资料。请尝试换一种问法，或提供更多关键词。', [])
            return

        context = self._build_context(chunks)
        # 生成始终针对用户原始问题
        prompt = RAG_USER.format(context=context, query=query)

        # 拼接messages：system + 长期摘要 + 近期对话 + 当前问题
        messages = [{'role': 'system',
                     'content': RAG_SYSTEM + (CAUTION_NOTE if caution else "")}]
        if history:
            # 提取长期记忆（system 角色的摘要消息，由 app.py 注入）
            summary_messages = [m for m in history if m.get('role') == 'system']
            # 提取近期对话（user/assistant）
            recent_turns = [m for m in history if m.get('role') in ('user', 'assistant')]
            # 摘要拼接进 system prompt 作为长期记忆
            for sm in summary_messages:
                messages[0]['content'] += '\n\n' + sm['content']
            # 最近 RECENT_WINDOW 条对话作为短期上下文（集中配置，不再手写魔法数）
            for msg in recent_turns[-RECENT_WINDOW:]:
                messages.append({'role': msg['role'], 'content': msg['content']})
        messages.append({'role': 'user', 'content': prompt})

        # 流式调用LLM
        full_answer = ""
        for delta in self.client.chat_stream(
            messages=messages,
            temperature=0.3,
            max_tokens=1024,
        ):
            full_answer += delta
            yield (delta, None)

        # 引用的chunk
        references = []
        seen_ids = set()
        for c in chunks[:MAX_CONTEXT_CHUNKS]:
            if c['chunk_id'] not in seen_ids:
                seen_ids.add(c['chunk_id'])
                references.append({
                    'chunk_id': c['chunk_id'],
                    'book': c.get('book', ''),
                    'chapter': c.get('chapter', ''),
                    'score': c.get('score', 0),
                    'source': c.get('source', ''),
                })

        yield ('', references)


if __name__ == '__main__':
    # 测试（需要先有检索结果）
    print('RAG生成器模块加载成功')
