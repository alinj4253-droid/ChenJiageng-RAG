"""
公开 RAG Benchmark 数据集下载脚本

下载 CRUD-RAG 和 MultiHop-RAG 到 evaluation/datasets/public/ 目录。

用法：
    python -m evaluation.download_benchmarks                # 下载全部
    python -m evaluation.download_benchmarks --dataset crud # 只下载 CRUD-RAG
    python -m evaluation.download_benchmarks --dataset multihop

下载后可用以下命令在公开数据集上运行消融：
    python -m evaluation.run_ablation --dataset evaluation/datasets/public/crud_rag.json --benchmark
"""
import argparse
import json
import urllib.request
from pathlib import Path


EVAL_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = EVAL_DIR / "datasets" / "public"

# 数据集下载配置（GitHub raw 文件链接）
BENCHMARKS = {
    "crud-rag": {
        "description": "CRUD-RAG: 面向 RAG 系统增删改查操作的中文评测集（IAAR-Shanghai）",
        "urls": [
            "https://raw.githubusercontent.com/IAAR-Shanghai/CRUD_RAG/main/data/crud_split/split_merged.json",
        ],
        "output": "crud_rag.json",
    },
    "multihop-rag": {
        "description": "MultiHop-RAG: 多跳推理 RAG 评测集（含 supporting_facts 标注）",
        "urls": [
            # GitHub 仓库中的数据集文件
            "https://raw.githubusercontent.com/yixuantt/MultiHop-RAG/main/dataset/MultiHopRAG.json",
            "https://raw.githubusercontent.com/yixuantt/MultiHop-RAG/main/data/MultiHopRAG.json",
            # HuggingFace 数据集直链
            "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/MultiHopRAG.json",
        ],
        "output": "multihop_rag.json",
    },
}


def download_file(url: str, dest: Path, timeout: int = 60) -> bool:
    """下载单个文件，成功返回 True"""
    print(f"  下载: {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            f.write(data)
        print(f"  保存: {dest} ({len(data)} bytes)")
        return True
    except Exception as e:
        print(f"  下载失败: {e}")
        return False


def download_benchmark(name: str) -> bool:
    """下载指定 benchmark，尝试所有 URL，成功一个即可"""
    if name not in BENCHMARKS:
        print(f"未知 benchmark: {name}，可选: {list(BENCHMARKS.keys())}")
        return False

    config = BENCHMARKS[name]
    print(f"\n{'=' * 60}")
    print(f"下载: {name}")
    print(f"描述: {config['description']}")
    print(f"{'=' * 60}")

    dest = PUBLIC_DIR / config["output"]
    if dest.exists():
        print(f"  文件已存在，跳过: {dest}")
        return True

    for url in config["urls"]:
        if download_file(url, dest):
            # 验证 JSON 可解析
            try:
                with open(dest, "r", encoding="utf-8") as f:
                    data = json.load(f)
                count = len(data) if isinstance(data, list) else len(data.get("data", data))
                print(f"  验证通过: {count} 条数据")
            except Exception as e:
                print(f"  JSON 解析警告: {e}（可能格式特殊，加载器会尝试处理）")
            return True

    print(f"  所有 URL 均下载失败。")
    print(f"  请手动下载并保存到: {dest}")
    print(f"  参考仓库:")
    if name == "crud-rag":
        print(f"    https://github.com/IAAR-Shanghai/CRUD_RAG")
        print(f"    数据文件: data/crud_split/split_merged.json")
    elif name == "multihop-rag":
        print(f"    https://github.com/yixuantt/MultiHop-RAG")
        print(f"    HuggingFace: https://huggingface.co/datasets/yixuantt/MultiHopRAG")
    return False


def main():
    parser = argparse.ArgumentParser(description="下载公开 RAG Benchmark 数据集")
    parser.add_argument(
        "--dataset",
        choices=["all", "crud-rag", "multihop-rag"],
        default="all",
        help="指定要下载的数据集（默认全部）",
    )
    args = parser.parse_args()

    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

    if args.dataset == "all":
        results = {name: download_benchmark(name) for name in BENCHMARKS}
    else:
        results = {args.dataset: download_benchmark(args.dataset)}

    print(f"\n{'=' * 60}")
    print("下载汇总:")
    for name, ok in results.items():
        status = "成功" if ok else "失败"
        print(f"  {name}: {status}")
    print(f"{'=' * 60}")
    print(f"\n数据集目录: {PUBLIC_DIR}")
    print("\n运行评估示例:")
    print("  python -m evaluation.run_ablation --dataset evaluation/datasets/public/crud_rag.json --benchmark")


if __name__ == "__main__":
    main()
