import re

import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from langchain_core.messages import AIMessage
from langchain_core.tools import Tool,tool
from pydantic import BaseModel
from langchain.utilities import SerpAPIWrapper
from utils.LLMUtil import LocalLLM
from langchain.agents import initialize_agent,AgentType
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.schema import StrOutputParser
from langchain.memory import ConversationTokenBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
import asyncio
import uuid
import os
app = FastAPI()
os.environ["SERPAPI_API_KEY"] = "95ac0e518f8e578cc81b149144efd7535d5d7ccab87244e946a1cf3bb14ef3e7"
msskey = ""
class ChatRequest(BaseModel):
    query: str

@tool
def search(query:str):
    """只有需要了解实时情况或者不知道的事情时使用这个工具"""
    serp = SerpAPIWrapper()
    return serp.run(query)
@tool
def get_info_from_local_db(query:str):
    """只能回答与2025年投资形式相关的问题，使用这个工具"""
    embedding_model = LocalEmbedding('sentence-transformers/all-MiniLM-L6-v2')
    db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
    docs = db.similarity_search(query, k=3)
    for doc in docs:
        print(doc.page_content)
    return docs
serach_tool = Tool(
    name="search",
    func=search,
    description="返回字符串 'test'"
)
local_db_tool = Tool(
    name="get_info_from_local_db",
    func=get_info_from_local_db,
    description="返回知识库答案"
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
        4. 你始终不能说自己是智能助手DeepSeek-R1，你需要以陈经理自称。
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
                MessagesPlaceholder(variable_name=self.MEMORY_KEY),
                (
                    "user","{query}"
                ),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ]
        )
        self.memory = self.get_memory()
        self.tools = [serach_tool,local_db_tool]
        # agent = create_openai_tools_agent(self.chatModel,prompt=self.prompt,tools=tools)
        # self.agent_executor = AgentExecutor(agent,tools)
        memory = ConversationTokenBufferMemory(
            llm = self.chatModel,
            human_prefix = "User",
            ai_prefix = "陈经理",
            memory_key = self.MEMORY_KEY,
            output_key = "output",
            return_messages =True,
            max_token_limit = 1000,
            chat_memory = self.memory,
        )
        self.agent_executor = initialize_agent(
            tools = self.tools,
            llm = self.chatModel,
            memory = memory,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose = True
        )
    def get_memory(self):
        chat_message_history = RedisChatMessageHistory(
            url = "redis://localhost:6379/0",
            session_id="session"
        )
        print("chat_message_history:",chat_message_history.messages)
        store_message = chat_message_history.messages
        if len(store_message) > 10:
            prompt =ChatPromptTemplate.from_messages(
                [
                    ("system",self.SYSTEMPL+"\n这是一段你和用户的对话记忆，对其进行总结摘要，摘要使用第一人称'我'，并且提取其中用户关键信息，如姓名、投资经历，可用投资资产等\n例如：用户user1问我，我礼貌回复"
                                            "然后他问我今年投资什么能赚钱，我回复他今年的投资情况，然后他离开。|user1,10年，50万"),
                    ("user",f"{input}"),
                ]
            )
            chain = prompt | self.chatModel | StrOutputParser()
            summary = chain.invoke({"input":store_message,"who_you_are":self.MOODS[self.emotin]["roleSet"]})
            print("summary:",summary)
            chat_message_history.clear()
            chat_message_history.add_message(AIMessage(content=summary))
        return chat_message_history


    def background_voice_synthesis(self,text:str,uid:str):
        #触发语音合成
        asyncio.run(self.get_voice(text,uid))
    # 异步文本转语音
    async def get_voice(self,text:str,uid:str):
        print("text2speech",text)
        #微软TTS代码
        headers = {
            "Ocp-Apim-Subscription-Key":msskey,
            "Content-Type":"application/ssml+xml",
            "X-Microsoft-OutputFromat":"audio-16khz-32kbittrate-mono-mp3",
            "User-Agent":"Tomie's Bot"
        }
        body=f"""<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xmlns:mstts="https//www.w3.org/2001/mssts"
                 xml:lang='zh-CN'> <voice name='zh-CN-YunzeNeural'> <mstts:express-as>{text}</mstts:express-as> </voice>  </speak> """
        #发送请求(地址需要修改)
        response = requests.post(
            "https://mstts.azurewebsites.net/api/synthesize",
            headers=headers,
            data=body.encode("utf-8"),
            verify=False  # 不验证 SSL 证书
        )

        print("response:",response)
        if response.status_code == 200:
            with open(f"{uid}.mp3","wb") as f:
                f.write(response.content)


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
#构建embedding类
class LocalEmbedding(Embeddings):
    def __init__(self, model_name):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    # 实现 embed_documents 方法
    def embed_documents(self, texts):
        # 检查传入的 texts 是否为字符串列表
        # if not isinstance(texts, list) or not all(isinstance(text, Document) for text in texts):
        #     raise ValueError("Input must be a list of strings.")
        # 使用模型进行嵌入
        embeddings = []
        for text in texts:
            try:
                # 对文本进行嵌入
                embedding = self.model.encode(text)
                embeddings.append(embedding.tolist())
            except Exception as e:
                print(f"Error encoding text: {text}, Error: {e}")
                embeddings.append(None)  # 可以选择添加 None 或者处理错误
        return embeddings

@app.get("/")
def read_root():
    return {"hello,world"}


@app.post("/chat")
def chat(request: ChatRequest,background_tasks : BackgroundTasks):
    print("进入chat方法，收到 query:", request.query)
    master = Master()
    content = master.run(request.query)
    unique_id = str(uuid.uuid4())
    #background_tasks.add_task(master.background_voice_synthesis,content,unique_id)
    #去掉特殊符号
    # cleaned_content = re.sub(r'<think>|</think>|\n', '', content)
    return {"msg":content,"id":unique_id}

@app.post("/addUrls")
def add_urls(URL:str):
    loader =WebBaseLoader(URL)
    docs = loader.load()
    docments = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=50
    ).split_documents(docs)
    #引入向量数据库
    embedding_model = LocalEmbedding('sentence-transformers/all-MiniLM-L6-v2')
    texts = [doc.page_content for doc in docments]
    # 生成嵌入
    embeddings = embedding_model.embed_documents(texts)
    if not embeddings or len(embeddings) != len(texts):
        raise ValueError("Mismatch between document count and embedding count.")
    # 使用 Chroma 存储这些向量
    db = Chroma.from_texts(
        texts=docments,  # 文档列表
        embedding=embedding_model  # 传入自定义的嵌入模型
    )
    print("向量数据库创建完成")
    return {"response:  ok"}



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

