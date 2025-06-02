import requests
import json
import logging
from typing import List, Dict, Any, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.outputs import LLMResult
from langchain_core.prompt_values import ChatPromptValue
from langchain.schema import AIMessage, HumanMessage, SystemMessage, BaseMessage

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
        arbitrary_types_allowed = True

    def __init__(self, model: Optional[str] = None, temperature: Optional[float] = None, api_url: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.model = model or self.model
        self.temperature = temperature or self.temperature
        self.api_url = api_url or self.api_url

    def _generate(self, prompts: List[str], **kwargs) -> LLMResult:
        generations = []
        for prompt in prompts:
            response_text = self.invoke(prompt, **kwargs)
            generations.append([{"text": response_text}])
        return LLMResult(generations=generations)

    def invoke(self, input: Any, config=None,**kwargs) -> str:
        headers = {"Content-Type": "application/json"}
        max_tokens = kwargs.get("max_tokens", 5000)

        messages = []
        if isinstance(input, list):
            for msg in input:
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                messages.append({"role": role, "content": custom_json_serialize(msg)})
        else:
            messages.append({"role": "user", "content": custom_json_serialize(input)})

        data = {
            "model": self.model,
            "options": {"temperature": self.temperature, "max_tokens": max_tokens},
            "stream": False,
            "messages": messages,
        }

        logging.debug(f"请求数据: {json.dumps(data, ensure_ascii=False, default=custom_json_serialize)}")

        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result.get("message", {}).get("content", "").strip() or "未返回有效内容"
        except requests.exceptions.RequestException as e:
            logging.error(f"API 请求错误: {e}")
            return "请求失败，请检查 API 服务器是否正常运行"
        except json.JSONDecodeError:
            logging.error("响应解析失败")
            return "无法解析服务器返回的内容"

    def predict(self, text: str, **kwargs) -> str:
        return self.invoke(text, **kwargs)

    def predict_messages(self, messages: List[BaseMessage], **kwargs) -> BaseMessage:
        output = self.invoke(messages, **kwargs)
        return AIMessage(content=output)

    def generate_prompt(self, prompt: ChatPromptValue, **kwargs) -> LLMResult:
        return self._generate([prompt.to_string()], **kwargs)

    async def apredict(self, text: str, **kwargs) -> str:
        return self.predict(text, **kwargs)

    async def apredict_messages(self, messages: List[BaseMessage], **kwargs) -> BaseMessage:
        return self.predict_messages(messages, **kwargs)

    async def agenerate_prompt(self, prompt: ChatPromptValue, **kwargs) -> LLMResult:
        return self.generate_prompt(prompt, **kwargs)

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model": self.model, "temperature": self.temperature}

    @property
    def _llm_type(self) -> str:
        return "custom_local_llm"
