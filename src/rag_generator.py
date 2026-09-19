"""
RAG问答生成层
上下文组装 + LLM生成 + 引用标注
"""
import json
from typing import List, Dict, Optional

from src.llm_client import get_client
from src.prompts import RAG_SYSTEM, RAG_USER
from src.config import MAX_CONTEXT_CHUNKS, MAX_CONTEXT_CHARS


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
                # 截断
                remaining = max_chars - total_chars
                if remaining > 100:
                    text = text[:remaining] + '...'
                else:
                    break

            source = f"[{i+1}]"
            book = chunk.get('book', '')
            chapter = chunk.get('chapter', '')
            header = f"{source} 《{book}》{chapter}："

            context_parts.append(header + text)
            total_chars += len(header) + len(text)

        return '\n\n'.join(context_parts)

    def generate(self, query: str, chunks: List[Dict],
                 query_analysis: Optional[Dict] = None) -> Dict:
        """
        基于检索结果生成答案。

        Args:
            query: 用户查询
            chunks: 检索到的chunk列表
            query_analysis: 查询分析结果（可选）

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

        # 构建prompt
        if query_analysis and query_analysis.get('rewritten_query'):
            effective_query = query_analysis['rewritten_query']
        else:
            effective_query = query

        prompt = RAG_USER.format(
            context=context,
            query=effective_query,
        )

        # 调用LLM
        answer = self.client.chat(
            messages=[
                {'role': 'system', 'content': RAG_SYSTEM},
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
                        history: list = None):
        """
        流式生成答案，yield (delta_text, references)
        history: 历史对话列表 [{"role":"user","content":"..."}, ...]
        最后一次yield时references非空。
        """
        if not chunks:
            yield ('抱歉，未找到与您问题相关的资料。请尝试换一种问法，或提供更多关键词。', [])
            return

        context = self._build_context(chunks)
        if query_analysis and query_analysis.get('rewritten_query'):
            effective_query = query_analysis['rewritten_query']
        else:
            effective_query = query

        prompt = RAG_USER.format(context=context, query=effective_query)

        # 拼接messages：system + 历史对话 + 当前用户问题
        messages = [{'role': 'system', 'content': RAG_SYSTEM}]
        if history:
            # 历史对话最多带最近1轮（2条消息），避免上下文过长
            for msg in history[-2:]:
                if msg['role'] in ('user', 'assistant'):
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
