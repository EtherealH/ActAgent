import requests
import json
import logging
from typing import List, Dict, Any, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain.schema import LLMResult, AIMessage, HumanMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue

# 启用日志
logging.basicConfig(level=logging.DEBUG)

def custom_json_serialize(obj):
    """处理 LangChain 对象的 JSON 序列化"""
    if isinstance(obj, ChatPromptValue):
        return obj.messages[0].content if obj.messages else ""
    elif isinstance(obj, (AIMessage, HumanMessage, SystemMessage)):
        return obj.content
    elif isinstance(obj, str):
        return obj
    elif isinstance(obj, tuple):
        return [custom_json_serialize(item) for item in obj]
    raise TypeError(f"无法序列化对象: {obj.__class__.__name__}")


class LocalLLM(BaseLanguageModel):
    model: str = "deepseek-r1:7b"
    temperature: float = 0.7
    api_url: str = "http://localhost:11343/api/chat"

    class Config:
        arbitrary_types_allowed = True  # 允许使用非标准类型

    def __init__(self, model: Optional[str] = None, temperature: Optional[float] = None, api_url: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.model = model or self.model
        self.temperature = temperature or self.temperature
        self.api_url = api_url or self.api_url

    def _generate(self, prompts: List[str], **kwargs) -> LLMResult:
        """LangChain 需要实现的 `_generate` 方法"""
        generations = []
        for prompt in prompts:
            response_text = self.invoke(prompt, **kwargs)
            generations.append([{"text": response_text}])

        return LLMResult(generations=generations)

    def invoke(self, input: Any, **kwargs) -> str:
        """调用本地 LLM 并返回结果"""
        headers = {"Content-Type": "application/json"}
        max_tokens = kwargs.get("max_tokens", 5000)

        # 解析输入（支持对话历史）
        messages = []
        if isinstance(input, list):  # 处理对话上下文
            for msg in input:
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                messages.append({"role": role, "content": custom_json_serialize(msg)})
        else:
            messages.append({"role": "user", "content": custom_json_serialize(input)})

        # 组装请求数据
        data = {
            "model": self.model,
            "options": {"temperature": self.temperature, "max_tokens": max_tokens},
            "stream": False,  # 这里默认不开启流式
            "messages": messages,
        }
        logging.debug(f"请求数据: {json.dumps(data, ensure_ascii=False, default=custom_json_serialize)}")

        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()  # 触发 HTTP 错误时抛出异常
            result = response.json()
            return result.get("message", {}).get("content", "").strip() or "未返回有效内容"
        except requests.exceptions.RequestException as e:
            logging.error(f"API 请求错误: {e}")
            return "请求失败，请检查 API 服务器是否正常运行"
        except json.JSONDecodeError:
            logging.error("响应解析失败")
            return "无法解析服务器返回的内容"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """用于标识 LLM 的参数"""
        return {"model": self.model, "temperature": self.temperature}

    @property
    def _llm_type(self) -> str:
        """返回模型类型"""
        return "custom_local_llm"
