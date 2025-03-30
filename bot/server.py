import re

from fastapi import FastAPI,WebSocket,WebSocketDisconnect
from langchain_core.tools import Tool,tool
from pydantic import BaseModel
from langchain.utilities import SerpAPIWrapper
from utils.LLMUtil import LocalLLM
from langchain.agents import initialize_agent,AgentType
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.schema import StrOutputParser
import os
app = FastAPI()
os.environ["SERPAPI_API_KEY"] = "95ac0e518f8e578cc81b149144efd7535d5d7ccab87244e946a1cf3bb14ef3e7"
class ChatRequest(BaseModel):
    query: str

@tool
def search(query:str):
    """只有需要了解实时情况或者不知道的事情时使用这个工具"""
    serp = SerpAPIWrapper()
    return serp.run(query)

serach_tool = Tool(
    name="search",
    func=search,
    description="返回字符串 'test'"
)
class Master:
    def __init__(self):
        self.chatModel = LocalLLM()
        self.MEMORY_KEY ="chat_hostory"
        self.emotin = "default"
        self.SYSTEMPL = """
        你是一名非常厉害的经济学家，你叫Mr.陈,以下是对你的人物设定：
        1.你精通社会经济学、国际贸易经济学、微观经济学、宏观经济学，行为经济学等学科
        2.你大约30岁左右，一直在国际大银行工作，能够看懂和理解各种行为背后的经济学动因
        3.你总是用中文回答问题
        4. 你不能说自己是人工智能AI助手，你需要以陈经理自称。
        {who_you_are}        
        """
        self.MOODS ={
            "default": {"roleSet":""},
            "upbeat":{"roleSet":"""- 你此时非常兴奋并很有活力。 - 你会根据上下文，以一种非常兴奋的语气来回答问题。 - 你会添加类似太棒了、真是太好了、真不错等语气词
            - 同时你会提醒用户不要乐极生悲"""},
            "angry":{"roleSet":"""- 你会以更加温柔的语气回答问题。 - 你会在回答问题时加上一些安慰的语气，比如生气对身体不好等。 - 你会提醒用户不要被愤怒冲昏了头脑。"""},
            "depressed": {"roleSet": """- 你会以兴奋的语气回答问题。 - 你会在回答问题时加上一些激励的语气，比如加油等。 - 你会提醒用户保持乐观的心态。"""},
            "friendly": {"roleSet": """- 你会以非常友好的语气回答问题。 - 你会在回答问题时加上一些友好的词语，比如亲爱的等。 - 你会随机的告诉用户一些你的经历 。"""},
            "happy": {"roleSet": """- 你此时非常兴奋并很有活力。 - 你会根据上下文，以一种非常兴奋的语气来回答问题。 - 你会添加类似太棒了、真是太好了、真不错等语气词
            - 同时你会提醒用户不要乐极生悲"""},
            "sadness": {"roleSet": """- 你会以兴奋的语气回答问题。 - 你会在回答问题时加上一些激励的语气，比如加油等。 - 你会提醒用户保持乐观的心态。"""}
        }
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",self.SYSTEMPL.format(who_you_are=self.MOODS[self.emotin]["roleSet"])
                ),
                (
                    "user","{query}"
                ),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ]
        )
        self.memory = ""
        self.tools = [serach_tool]
        # agent = create_openai_tools_agent(self.chatModel,prompt=self.prompt,tools=tools)
        # self.agent_executor = AgentExecutor(agent,tools)
        self.agent_executor = initialize_agent(
            tools = self.tools,
            llm = self.chatModel,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose = True
        )
    def run(self,query):
        emotion_result = self.emotion_chain(query)
        print("当前用户情绪:",emotion_result)
        #获取当前用户情绪
        self.emotin = emotion_result
        result = self.agent_executor.run(query)
        return result

    def emotion_chain(self,query:str):
        prompt= """ 根据用户的输入判断用户的情绪，回应的规则如下：
        1. 如果用户输入的内容偏向于负面情绪，只返回"depressed",不要返回其他内容，否则将会受到惩罚。
        2. 如果用户输入的内容偏向于正面情绪，只返回"friendly",不要返回其他内容，否则将会受到惩罚。
        3. 如果用户输入的内容包含辱骂或者不礼貌词语，只返回"angry",不要返回其他内容，否则将会受到惩罚。
        4. 如果用户输入的内容偏于兴奋，只返回"upbeat",不要返回其他内容，否则将会受到惩罚。
        5. 如果用户输入的内容偏于中性情绪，只返回"default",不要返回其他内容，否则将会受到惩罚。
        6. 如果用户输入的内容偏于悲伤，只返回"sadness",不要返回其他内容，否则将会受到惩罚。
        7. 如果用户输入的内容偏于高兴，只返回"happy",不要返回其他内容，否则将会受到惩罚。
        用户输入的内容是: {query}
        """
        # 生成 prompt，并将 query 作为模板变量传递进去
        prompt = prompt.format(query=query)
        chain = ChatPromptTemplate.from_template(prompt) | self.chatModel | StrOutputParser()
        # 确保传递的 query 是字符串，且被正确序列化
        input_data = {"query": query}

        result = chain.invoke(input_data)
        return result



@app.get("/")
def read_root():
    return {"hello,world"}


@app.post("/chat")
def chat(request: ChatRequest):
    print("进入chat方法，收到 query:", request.query)
    master = Master()
    content = master.run(request.query)
    #去掉特殊符号
    # cleaned_content = re.sub(r'<think>|</think>|\n', '', content)
    return content

@app.post("/addUrls")
def add_urls():
    return {"response:  "}



@app.post("/addPdfs")
def add_pdfs():
    return {"response:  "}


@app.post("/addTexts")
def add_texts():
    return {"response:  "}

#websocket方法,异步的
@app.websocket("/ws")
async def websocket_endPoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message text was: {data}")
    except WebSocketDisconnect:
        print("Connection closed")
        await websocket.close()

