"""
嘉庚智答 - RAG系统后端
FastAPI + RAG Pipeline + 对话存储
"""
import os
import sys
import json
import asyncio
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# 确保项目根目录在sys.path中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_pipeline import RAGPipeline
from src.db import DBManager
from src.config import RECENT_WINDOW

# ============ 初始化 ============
app = FastAPI(title="嘉庚智答", version="1.0.0")

# 全局实例（懒加载）
_pipeline = None
_db = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        print("正在初始化RAG Pipeline...")
        _pipeline = RAGPipeline(use_graph=True, use_rerank=True)
    return _pipeline


def get_db():
    global _db
    if _db is None:
        _db = DBManager(db_url="sqlite:///data/chat.db")
    return _db


# ============ 请求/响应模型 ============
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


class SessionCreateRequest(BaseModel):
    title: Optional[str] = "新对话"


# ============ API接口 ============

@app.get("/api/health")
def health_check():
    """健康检查"""
    return {"status": "ok", "service": "嘉庚智答 RAG"}


@app.post("/api/chat")
def chat(req: ChatRequest):
    """单轮同步 RAG 查询接口。

    职责边界：每次请求独立执行「检索 → 生成」，**不把会话历史传入生成器**，
    即不维护完整 Conversation Memory。它仍会把问答写入会话表（便于记录/回看），
    但回答不依赖上文。适合一次性问答、批量调用或对延迟敏感的场景。
    多轮有上下文的聊天请使用 ``/api/chat/stream``。
    """
    db = get_db()
    pipeline = get_pipeline()

    # 如果没有session_id，创建新会话
    session_id = req.session_id
    if not session_id:
        session_id = db.create_session(req.message[:30])

    # 保存用户消息
    db.add_message(session_id, "user", req.message)

    # 执行RAG查询
    try:
        result = pipeline.query(req.message, verbose=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG查询失败: {str(e)}")

    # 保存助手消息
    sources_json = json.dumps(result.get("references", []), ensure_ascii=False)
    db.add_message(session_id, "assistant", result["answer"], sources=sources_json)

    return {
        "session_id": session_id,
        "answer": result["answer"],
        "references": result.get("references", []),
        "latency": result.get("latency", {}),
    }


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """完整多轮聊天主接口（SSE 流式）。

    职责边界：这是前端聊天主链路，维护 **完整会话记忆**——
    Session 持久化、近期对话窗口（RECENT_WINDOW）、增量滚动摘要，
    并以 SSE 流式输出 token。后台异步生成，前端断开不打断。
    与 ``/api/chat``（单轮、无上下文）明确分工，不复用同一套会话状态。
    """
    db = get_db()
    pipeline = get_pipeline()

    session_id = req.session_id
    if not session_id:
        session_id = db.create_session(req.message[:30])

    # 1. 先取历史对话（必须在本轮用户消息入库之前），
    #    否则当前问题会同时出现在 history 和最终 prompt 中，造成重复。
    history = db.build_chat_history(session_id, recent_window=RECENT_WINDOW)

    # 2. 再存本轮用户问题
    db.add_message(session_id, "user", req.message)

    # 全局任务状态表：key=session_id
    global running_tasks
    if 'running_tasks' not in globals():
        running_tasks = {}
    # 初始化该任务状态
    running_tasks[session_id] = {
        "status": "running",
        "full_answer": "",
        "references": [],
        "done": False
    }

    # 增量滚动摘要器（复用 pipeline 的 LLM 客户端）
    from src.memory_manager import RollingSummaryManager
    memory_manager = RollingSummaryManager(
        db, client=pipeline.generator.client, recent_window=RECENT_WINDOW
    )

    # 后台生成协程：不依赖前端连接，生成完自动存库
    async def background_generate():
        full_answer = ""
        refs = []
        try:
            for event in pipeline.query_stream(req.message, history=history):
                if event["type"] == "token":
                    full_answer += event["content"]
                    running_tasks[session_id]["full_answer"] = full_answer
                if event.get("type") == "done":
                    refs = event.get("references", [])
                    running_tasks[session_id]["references"] = refs
            # 生成完成存库
            db.add_message(session_id, "assistant", full_answer, sources=json.dumps(refs, ensure_ascii=False))
            running_tasks[session_id]["status"] = "done"
            running_tasks[session_id]["done"] = True
            # 增量滚动摘要：只摘要已滑出 recent window、且尚未摘要过的消息
            try:
                memory_manager.maybe_summarize(session_id)
            except Exception as summary_err:
                print(f"[memory] 滚动摘要失败（不影响主流程）: {summary_err}")
        except Exception as e:
            running_tasks[session_id]["status"] = "error"
            running_tasks[session_id]["error"] = str(e)

    # 启动后台任务
    asyncio.create_task(background_generate())

    async def event_generator():
        yield f"data: {json.dumps({'type': 'session', 'session_id': session_id}, ensure_ascii=False)}\n\n"
        # 每200ms轮询任务状态推给前端
        last_len = 0
        while True:
            task = running_tasks.get(session_id)
            if not task:
                break
            if task["done"]:
                yield f"data: {json.dumps({'type': 'done', 'references': task['references']}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'saved'}, ensure_ascii=False)}\n\n"
                break
            if task["status"] == "error":
                yield f"data: {json.dumps({'type': 'error', 'message': task.get('error', '')}, ensure_ascii=False)}\n\n"
                break
            # 只推送新增的增量内容
            current = task["full_answer"]
            if len(current) > last_len:
                delta = current[last_len:]
                yield f"data: {json.dumps({'type': 'token', 'content': delta}, ensure_ascii=False)}\n\n"
                last_len = len(current)
            await asyncio.sleep(0.2)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


# 查询任务状态接口：前端切回来时拉取
@app.get("/api/task/{session_id}/status")
def get_task_status(session_id: str):
    global running_tasks
    if 'running_tasks' not in globals():
        return {"status": "idle", "answer": "", "done": True}
    task = running_tasks.get(session_id)
    if not task:
        return {"status": "idle", "answer": "", "done": True}
    return {
        "status": task["status"],
        "answer": task["full_answer"],
        "references": task.get("references", []),
        "done": task["done"]
    }


@app.get("/api/sessions")
def list_sessions():
    """获取对话列表"""
    db = get_db()
    sessions = db.get_sessions()
    return {"sessions": sessions}


@app.get("/api/sessions/{session_id}/messages")
def get_session_messages(session_id: str):
    """获取某个对话的历史消息"""
    db = get_db()
    messages = db.get_session_messages(session_id)
    return {"messages": messages}


@app.post("/api/sessions")
def create_session(req: SessionCreateRequest):
    """创建新对话"""
    db = get_db()
    session_id = db.create_session(req.title)
    return {"session_id": session_id}


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    """删除对话"""
    db = get_db()
    db.delete_session(session_id)
    return {"status": "ok"}


@app.get("/api/graph")
def get_graph_data(entity: Optional[str] = None, depth: int = 2):
    """获取知识图谱数据（用于前端可视化）"""
    from neo4j import GraphDatabase
    from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    try:
        with driver.session() as session:
            # 如果指定了实体，查询它的2层邻居
            if entity:
                query = """
                MATCH (n:Entity {name: $entity})-[r:RELATED*1..2]-(m:Entity)
                RETURN n, r, m
                LIMIT 200
                """
                result = session.run(query, entity=entity)
            else:
                # 默认从陈嘉庚这个核心实体开始，返回2层邻居
                query = """
                MATCH (n:Entity {name: '陈嘉庚'})-[r:RELATED*1..2]-(m:Entity)
                RETURN n, r, m
                LIMIT 200
                """
                result = session.run(query)

            nodes_dict = {}
            edges_list = []

            for record in result:
                # 处理起点节点
                n = record["n"]
                if n and n["name"] not in nodes_dict:
                    nodes_dict[n["name"]] = {
                        "id": n["name"],
                        "label": n["name"],
                        "group": n.get("type", "OTHER")
                    }

                # 处理终点节点
                m = record["m"]
                if m and m["name"] not in nodes_dict:
                    nodes_dict[m["name"]] = {
                        "id": m["name"],
                        "label": m["name"],
                        "group": m.get("type", "OTHER")
                    }

                # 处理边（变长路径下r是关系数组，直接用n和m连接）
                if n and m:
                    edge_key = n["name"] + "->" + m["name"]
                    if edge_key not in [e.get("key") for e in edges_list]:
                        edges_list.append({
                            "key": edge_key,
                            "from": n["name"],
                            "to": m["name"],
                            "label": "相关"
                        })

            return {
                "nodes": list(nodes_dict.values()),
                "edges": edges_list
            }
    finally:
        driver.close()


# ============ 静态文件服务 ============
# 前端HTML文件放在 web/ 目录下
WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")

if os.path.exists(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
def index():
    """首页 - 聊天界面"""
    html_path = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"message": "前端页面待开发，API已就绪"}


@app.get("/graph")
def graph_page():
    """图谱可视化页面"""
    html_path = os.path.join(WEB_DIR, "graph.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"message": "图谱页面待开发"}


# ============ 启动 ============
if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("  嘉庚智答 RAG系统")
    print("  访问地址: http://localhost:8000")
    print("  API文档: http://localhost:8000/docs")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)
