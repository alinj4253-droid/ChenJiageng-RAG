"""
切块器：按章节-句子层级语义切分
适配OCR文本（段落间无空行，靠短行判断段落结束）
"""
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple

from src.config import (
    CLEAN_DIR,
    CHUNKS_DIR,
    CHUNK_TARGET_SIZE,
    CHUNK_MAX_SIZE,
    CHUNK_OVERLAP,
    CHUNK_MIN_SIZE,
)


# ============ 章节标题识别 ============

CHAPTER_PATTERNS = [
    (re.compile(r'^第[一二三四五六七八九十百零\d]+[篇章节卷回部]\s*'), 'chapter'),
    (re.compile(r'^[一二三四五六七八九十○零]{1,3}\s+\S'), 'section'),
    (re.compile(r'^[一二三四五六七八九十○零]{1,3}、\s*\S'), 'section'),
    (re.compile(r'^\d{1,2}\.\s+\S'), 'subsection'),
    (re.compile(r'^附[录一二三四五六七八九十\d]'), 'appendix'),
]

TOC_PATTERN = re.compile(r'[（(]\s*\d+\s*[)）]|\.{3,}\s*\d+\s*$')
SENTENCE_END = re.compile(r'[。！？；!?;]')

# 正文平均行宽（OCR文本）
AVG_LINE_WIDTH = 35


def is_chapter_title(line: str) -> Tuple[bool, str]:
    """判断一行是否为章节标题"""
    stripped = line.strip()
    if not stripped or len(stripped) > 35:
        return False, ''
    if TOC_PATTERN.search(stripped):
        return False, ''
    if re.match(r'^\d+$', stripped):
        return False, ''
    for pattern, level in CHAPTER_PATTERNS:
        if pattern.match(stripped):
            return True, level
    return False, ''


def find_body_start(lines: List[str]) -> int:
    """
    检测正文开始位置，跳过封面/版权页/目录。
    策略：找目录区（大量短行匹配标题模式）结束后的正文。
    """
    # 统计每行是否像目录项（短行+匹配标题模式）
    toc_scores = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if 0 < len(stripped) < 35:
            is_title, _ = is_chapter_title(stripped)
            toc_scores.append(1 if is_title else 0)
        else:
            toc_scores.append(0)

    # 找连续的目录区（窗口内目录项比例>0.5）
    window = 50
    best_end = 0
    best_score = 0
    for start in range(0, len(lines) - window, window // 2):
        end = min(start + window, len(lines))
        score = sum(toc_scores[start:end]) / (end - start)
        if score > 0.4 and end > best_end:
            best_end = end
            best_score = score

    if best_end > 100:  # 目录区超过100行才认为有目录
        # 从目录结束往后找第一个非空长行
        for i in range(best_end, min(best_end + 100, len(lines))):
            if len(lines[i].strip()) > AVG_LINE_WIDTH * 0.8:
                return i
        return best_end

    return 0


def extract_chapters(lines: List[str]) -> List[Dict]:
    """
    从行列表中提取章节结构。
    返回: [{title, level, start_line, end_line}, ...]
    """
    chapters = []
    current = {'title': '前言', 'level': 'preface', 'start_line': 0}

    for i, line in enumerate(lines):
        stripped = line.strip()
        is_title, level = is_chapter_title(stripped)

        if is_title:
            # 确认是正文中的标题（不是目录）：前一行是空行或短行，且不是连续标题
            prev_empty = i == 0 or not lines[i-1].strip() or len(lines[i-1].strip()) < 20
            # 检查是否在目录区（后面很多连续短行标题）
            next_titles = 0
            for j in range(i+1, min(i+20, len(lines))):
                t, _ = is_chapter_title(lines[j].strip())
                if t:
                    next_titles += 1
                elif len(lines[j].strip()) > AVG_LINE_WIDTH:
                    break

            if prev_empty and next_titles < 5:  # 不是目录区
                current['end_line'] = i - 1
                if current['end_line'] > current['start_line']:
                    chapters.append(current)
                current = {'title': stripped, 'level': level, 'start_line': i}

    current['end_line'] = len(lines) - 1
    if current['end_line'] > current['start_line']:
        chapters.append(current)

    return chapters


def lines_to_text(lines: List[str]) -> str:
    """将OCR行合并为连续文本，处理换行"""
    # 合并连续行，短行（段落结束）后加换行
    text_parts = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            text_parts.append('\n')
            continue

        is_short = len(stripped) < AVG_LINE_WIDTH * 0.7
        next_empty = i == len(lines) - 1 or not lines[i+1].strip()
        next_short = i < len(lines) - 1 and len(lines[i+1].strip()) < AVG_LINE_WIDTH * 0.7

        text_parts.append(stripped)
        # 短行且下一行非空长行 → 可能是段落结束，加换行
        if is_short and not next_empty and not next_short:
            text_parts.append('\n')
        # 短行且下一行空 → 段落结束
        elif is_short and next_empty:
            text_parts.append('\n')

    return ''.join(text_parts)


def split_sentences(text: str) -> List[str]:
    """按句末标点切分句子，保留标点"""
    sentences = []
    current = ''
    for char in text:
        current += char
        if char in '。！？；!?;':
            sentences.append(current.strip())
            current = ''
    if current.strip():
        sentences.append(current.strip())
    return [s for s in sentences if s]


def chunk_text(text: str, book_name: str, chapter_title: str, chapter_idx: int,
               start_chunk_idx: int = 0) -> List[Dict]:
    """
    将文本按句子切分后合并为目标大小的chunk。
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks = []
    current_text = ''
    current_sentences = []
    chunk_idx = start_chunk_idx

    for sent in sentences:
        if len(current_text) + len(sent) > CHUNK_MAX_SIZE and current_text:
            chunks.append({
                'chunk_id': f'{book_name}_ch{chapter_idx:03d}_{chunk_idx:03d}',
                'book': book_name,
                'chapter': chapter_title,
                'chapter_idx': chapter_idx,
                'chunk_idx': chunk_idx,
                'text': current_text.strip(),
                'char_count': len(current_text.strip()),
            })
            chunk_idx += 1

            # overlap：保留最后几个句子
            overlap_text = ''
            overlap_len = 0
            for s in reversed(current_sentences):
                if overlap_len + len(s) > CHUNK_OVERLAP:
                    break
                overlap_text = s + overlap_text
                overlap_len += len(s)
            current_text = overlap_text
            current_sentences = [s for s in current_sentences if s in overlap_text] if overlap_text else []

        current_text += sent
        current_sentences.append(sent)

    if current_text.strip() and len(current_text.strip()) >= CHUNK_MIN_SIZE:
        chunks.append({
            'chunk_id': f'{book_name}_ch{chapter_idx:03d}_{chunk_idx:03d}',
            'book': book_name,
            'chapter': chapter_title,
            'chapter_idx': chapter_idx,
            'chunk_idx': chunk_idx,
            'text': current_text.strip(),
            'char_count': len(current_text.strip()),
        })

    return chunks


def chunk_book(filepath: Path, book_name: str) -> List[Dict]:
    """处理一本书"""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 跳过目录
    body_start = find_body_start(lines)
    if body_start > 0:
        print(f'  跳过前{body_start}行（封面/目录）')
        lines = lines[body_start:]

    # 提取章节
    chapters = extract_chapters(lines)
    print(f'  识别到{len(chapters)}个章节')

    # 每章切块
    all_chunks = []
    for ch_idx, chapter in enumerate(chapters):
        ch_lines = lines[chapter['start_line']:chapter['end_line'] + 1]
        text = lines_to_text(ch_lines)

        if len(text) < CHUNK_MIN_SIZE:
            continue

        chunks = chunk_text(text, book_name, chapter['title'], ch_idx)
        all_chunks.extend(chunks)

    return all_chunks


def run_chunking():
    """执行所有书的切块"""
    all_chunks = []
    book_stats = {}

    for txt_file in sorted(CLEAN_DIR.glob('*.txt')):
        book_name = txt_file.stem
        print(f'处理: {book_name}')

        chunks = chunk_book(txt_file, book_name)
        all_chunks.extend(chunks)

        char_counts = [c['char_count'] for c in chunks]
        book_stats[book_name] = {
            'chunk_count': len(chunks),
            'total_chars': sum(char_counts),
            'avg_chars': sum(char_counts) // len(chunks) if chunks else 0,
            'min_chars': min(char_counts) if char_counts else 0,
            'max_chars': max(char_counts) if char_counts else 0,
        }
        print(f'  生成 {len(chunks)} 个chunk, 平均{book_stats[book_name]["avg_chars"]}字')

    # 保存
    output_file = CHUNKS_DIR / 'chunks.jsonl'
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

    stats = {
        'total_chunks': len(all_chunks),
        'total_chars': sum(c['char_count'] for c in all_chunks),
        'books': book_stats,
        'config': {
            'target_size': CHUNK_TARGET_SIZE,
            'max_size': CHUNK_MAX_SIZE,
            'overlap': CHUNK_OVERLAP,
            'min_size': CHUNK_MIN_SIZE,
        },
    }
    stats_file = CHUNKS_DIR / 'chunk_stats.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f'\n=== 切块完成 ===')
    print(f'总计: {len(all_chunks)} 个chunk')
    print(f'总字数: {stats["total_chars"]:,}')

    return all_chunks, stats


if __name__ == '__main__':
    run_chunking()
