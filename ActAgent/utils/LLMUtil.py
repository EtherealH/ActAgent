import requests
import json
import logging

from langchain_core.prompt_values import ChatPromptValue
from openai import BaseModel
from pydantic import Field
from typing import List, Dict

# 启用日志记录
logging.basicConfig(level=logging.DEBUG)

def custom_json_serialize(obj):
    if isinstance(obj, ChatPromptValue):
        # 提取 ChatPromptValue 中的 messages
        messages = obj.messages
        if messages and hasattr(messages[0], 'content'):  # 检查是否有 content 属性
            return messages[0].content  # 返回第一个消息的 content
    elif isinstance(obj,str):
        return obj
    elif isinstance(obj, BaseModel):
        return obj.dict()
    elif isinstance(obj, tuple):
        contents = []
        for item in obj:
            if hasattr(item, 'content'):
                contents.append(item.content)
        return contents
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")



class LocalLLM():
    model: str = "deepseek-r1:7b"
    temperature: float = 0.7
    api_url: str = "http://localhost:11343/api/chat"

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, model: str = None, temperature: float = None, api_url: str = None, **kwargs):
        super().__init__(**kwargs)
        self.model = model or self.model
        self.temperature = temperature or self.temperature
        self.api_url = api_url or self.api_url

    def invoke(self, input: str, **kwargs) -> str:
        headers = {"Content-Type": "application/json"}
        max_tokens = kwargs.get("max_tokens", 5000)

        # 准备请求数据
        data = {
            "model": self.model,
            "options": {"temperature": self.temperature, "max_tokens": max_tokens},
            "stream": True,  # 启用流式响应
            "messages": [{"role": "user", "content": custom_json_serialize(input)}],
        }
        print("请求数据:", data)
        logging.debug(f"请求数据: {json.dumps(data, ensure_ascii=False,default=custom_json_serialize)}")  # 记录请求数据

        try:
            response = requests.post(self.api_url, headers=headers, data=json.dumps(data,default=custom_json_serialize), stream=True)

            if response.status_code == 200:
                content = []
                # 逐块读取响应内容，直到完成
                for chunk in response.iter_lines():
                    if chunk:
                        part = json.loads(chunk.decode("utf-8"))
                        # 处理每个响应片段
                        logging.debug(f"收到部分响应: {part}")  # 记录每个部分的内容
                        content.append(part.get("message", {}).get("content", ""))

                        # 判断是否完成
                        if part.get("done", False):
                            break

                # 返回合并后的消息内容
                return "".join(content).strip() or "未返回有效的内容"
            else:
                raise Exception(f"请求模型服务器失败: {response.status_code} - {response.text}")
        except Exception as e:
            logging.error(f"API 请求发生错误: {str(e)}")
            raise

    def __call__(self, input: str, *args, **kwargs):
        return self.invoke(input, **kwargs)

    def generate(self, prompts: List[str], **kwargs) -> Dict[str, List[List[Dict[str, str]]]]:
        generations = []
        for prompt in prompts:
            response_text = self.invoke(prompt, **kwargs)
            generations.append([{"text": response_text}])
        return {"generations": generations}
