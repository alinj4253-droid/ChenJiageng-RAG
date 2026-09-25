"""
LLM 客户端：线上 API 优先，多模型轮询，额度用完自动降级到本地 Ollama
使用标准库 urllib，零第三方依赖
"""
import json
import time
import urllib.request
import urllib.error
from typing import Optional

from src.config import (
    OPENAI_API_BASE,
    OPENAI_API_KEY,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    ALL_ONLINE_MODELS,
    FAST_MODELS,
    PREMIUM_MODELS,
)


def _http_post(url: str, headers: dict, payload: dict, timeout: int = 60) -> tuple:
    """
    发送 POST 请求，返回 (status_code, response_dict_or_text)
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            try:
                return resp.status, json.loads(body)
            except json.JSONDecodeError:
                return resp.status, body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(body)
        except json.JSONDecodeError:
            return e.code, body
    except urllib.error.URLError as e:
        return -1, str(e)
    except Exception as e:
        return -1, str(e)


class LLMClient:
    """统一 LLM 调用接口"""

    def __init__(self, preferred_models: Optional[list] = None, use_online: bool = True):
        """
        Args:
            preferred_models: 优先使用的模型列表，None则用全部
            use_online: 是否使用线上API，False则直接用本地
        """
        self.use_online = use_online and bool(OPENAI_API_KEY)
        self.models = preferred_models if preferred_models else ALL_ONLINE_MODELS
        self._model_idx = 0  # 轮询索引
        self._local_fallback = False  # 是否已经降级到本地
        self.last_model = ""  # 最近一次成功调用使用的模型名（用于落盘可追溯）

    def _next_model(self) -> str:
        """轮询获取下一个模型"""
        model = self.models[self._model_idx % len(self.models)]
        self._model_idx += 1
        return model

    def chat(
        self,
        messages: list,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        response_format: Optional[dict] = None,
        timeout: int = 180,
    ) -> str:
        """
        聊天补全。线上API优先，失败/额度用完自动降级本地。

        Args:
            messages: [{"role": "system"/"user"/"assistant", "content": "..."}]
            model: 指定模型，None则轮询
            temperature: 温度
            max_tokens: 最大生成token
            response_format: {"type": "json_object"} 等
            timeout: 超时秒数

        Returns:
            生成的文本内容
        """
        if self.use_online and not self._local_fallback:
            result = self._chat_online(messages, model, temperature, max_tokens, response_format, timeout)
            if result is not None:
                return result
            # 线上全部失败，标记降级并尝试本地 Ollama
            print(f"[LLM] 线上API全部失败，降级到本地 Ollama ({OLLAMA_MODEL})")
            self._local_fallback = True

        # 本地降级路径（use_online=False 时直接走这里）
        try:
            return self._chat_local(messages, temperature, max_tokens)
        except Exception as e:
            print(f"[LLM] 本地Ollama也失败: {e}")
            return ""

    def _chat_online(self, messages, model, temperature, max_tokens, response_format, timeout) -> Optional[str]:
        """调用线上 OpenAI 兼容 API"""
        # 尝试每个模型，最多轮询2轮
        for attempt in range(len(self.models) * 2):
            use_model = model or self._next_model()
            url = f"{OPENAI_API_BASE}/chat/completions"
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": use_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if response_format:
                payload["response_format"] = response_format

            status, body = _http_post(url, headers, payload, timeout)

            if status == 200 and isinstance(body, dict):
                content = body["choices"][0]["message"]["content"]
                self.last_model = use_model
                return content
            elif status == 429:
                # 额度用完/限流，换模型
                print(f"[LLM] {use_model} 限流(429)，换下一个模型")
                time.sleep(1)
                continue
            elif status == 401:
                print(f"[LLM] {use_model} 认证失败(401)，API key无效")
                return None
            elif status == 403:
                # 特定模型额度用完或无权限，换模型继续试
                print(f"[LLM] {use_model} 403(额度用完/无权限)，换下一个模型")
                time.sleep(1)
                continue
            elif status == -1:
                print(f"[LLM] {use_model} 网络异常: {body}")
                time.sleep(1)
                continue
            else:
                err_msg = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)[:200]
                print(f"[LLM] {use_model} 错误({status}): {err_msg}")
                time.sleep(1)
                continue

        return None

    def _chat_local(self, messages, temperature, max_tokens) -> str:
        """调用本地 Ollama（非流式）"""
        url = f"{OLLAMA_BASE_URL}/api/chat"
        payload = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        status, body = _http_post(url, {"Content-Type": "application/json"}, payload, timeout=120)
        if status == 200 and isinstance(body, dict):
            self.last_model = OLLAMA_MODEL
            return body["message"]["content"]
        raise RuntimeError(f"本地Ollama调用失败: status={status}, body={body}")

    def _chat_local_stream(self, messages, temperature, max_tokens):
        """调用本地 Ollama（流式），yield 每个 token"""
        url = f"{OLLAMA_BASE_URL}/api/chat"
        payload = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data,
                                      headers={"Content-Type": "application/json"},
                                      method="POST")
        try:
            resp = urllib.request.urlopen(req, timeout=120)
            for line in resp:
                line = line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    if chunk.get("done"):
                        return
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yield content
                except json.JSONDecodeError:
                    continue
            self.last_model = OLLAMA_MODEL
        except Exception as e:
            print(f"[LLM] 本地Ollama流式失败: {e}")
            yield ""

    def extract_json(self, prompt: str, system_prompt: str = "你是一个信息抽取助手，严格输出JSON格式，不要输出任何其他内容。") -> dict:
        """
        从文本中抽取结构化JSON。自动处理JSON解析。

        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词

        Returns:
            解析后的dict，失败返回{}
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt + "\n\n请严格只输出JSON，不要输出解释、markdown标记或其他任何内容。"},
        ]
        content = self.chat(
            messages,
            temperature=0.1,
            max_tokens=1024,
            timeout=120,
        )
        return self._parse_json(content)

    @staticmethod
    def _parse_json(content: str) -> dict:
        """从LLM输出中解析JSON，处理各种格式问题"""
        content = content.strip()
        # 去掉 ```json ... ``` 包裹
        if content.startswith("```"):
            content = content.strip("`")
            if content.lower().startswith("json"):
                content = content[4:]
            content = content.strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # 尝试提取第一个 { 到最后一个 }
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(content[start:end + 1])
                except json.JSONDecodeError:
                    pass
            print(f"[LLM] JSON解析失败，原始内容前200字: {content[:200]}")
            return {}


    def chat_stream(
        self,
        messages: list,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: int = 180,
    ):
        """
        流式聊天补全，生成器，yield每个token。
        一旦开始输出内容，后续出错就不换模型，避免前端拼接重复内容。
        线上全部失败后自动降级到本地 Ollama 流式。
        """
        # 已降级或禁用线上 → 直接走本地流式
        if not self.use_online or self._local_fallback:
            yield from self._chat_local_stream(messages, temperature, max_tokens)
            return

        # 尝试每个线上模型
        for attempt in range(len(self.models) * 2):
            use_model = model or self._next_model()
            url = f"{OPENAI_API_BASE}/chat/completions"
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": use_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            }

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            has_output = False
            try:
                resp = urllib.request.urlopen(req, timeout=timeout)
                # 流式读取SSE
                for line in resp:
                    line = line.decode("utf-8").strip()
                    if not line or not line.startswith("data: "):
                        continue
                    if line == "data: [DONE]":
                        return
                    try:
                        chunk = json.loads(line[6:])
                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {}).get("content", "")
                        if delta:
                            has_output = True
                            yield delta
                    except json.JSONDecodeError:
                        continue
                    except (KeyError, IndexError):
                        continue
                return  # 成功结束
            except urllib.error.HTTPError as e:
                code = e.code
                if has_output:
                    # 已经输出内容了，直接结束，不换模型避免重复
                    return
                if code in (403, 429):
                    print(f"[LLM] {use_model} {code}，换下一个模型")
                    time.sleep(1)
                    continue
                else:
                    print(f"[LLM] {use_model} HTTP {code}")
                    time.sleep(1)
                    continue
            except Exception as e:
                print(f"[LLM] {use_model} 流式错误: {e}")
                if has_output:
                    # 已经输出内容了，直接结束
                    return
                time.sleep(1)
                continue

        # 线上全部失败（或已降级），尝试本地 Ollama 流式
        if self.use_online:
            print(f"[LLM] 线上API流式全部失败，降级到本地 Ollama ({OLLAMA_MODEL})")
            self._local_fallback = True
        yield from self._chat_local_stream(messages, temperature, max_tokens)


# 便捷实例
def get_fast_client() -> LLMClient:
    """获取快速模型客户端（用于抽取、分类等量大任务）"""
    return LLMClient(preferred_models=FAST_MODELS)


def get_premium_client() -> LLMClient:
    """获取高质量模型客户端（用于生成、复杂推理）"""
    return LLMClient(preferred_models=PREMIUM_MODELS)


def get_client() -> LLMClient:
    """获取默认客户端（全部模型轮询）"""
    return LLMClient()


if __name__ == "__main__":
    # 测试
    client = get_client()
    print("测试线上API...")
    result = client.chat([{"role": "user", "content": "用一句话介绍陈嘉庚，不超过30字。"}])
    print(f"结果: {result}")
