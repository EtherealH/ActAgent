import re
import json

import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from langchain_core.messages import AIMessage
from langchain_core.tools import Tool,tool
from pydantic import BaseModel
from langchain_community.utilities import SerpAPIWrapper
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
from typing import List, Optional
import asyncio
from bot.wechat import wechat_api
import uuid
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 添加CORS中间件，支持微信小程序跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应设置具体的域名，如: ["https://servicewechat.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.environ["SERPAPI_API_KEY"] = "95ac0e518f8e578cc81b149144efd7535d5d7ccab87244e946a1cf3bb14ef3e7"
msskey = ""

# 向量数据库配置
VECTOR_DB_PATH = "./chroma_db"
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
DEFAULT_RETRIEVAL_K = 5  # 默认检索文档数量

class ChatRequest(BaseModel):
    query: str
    use_auto_rag: Optional[bool] = False  # 是否使用自动RAG模式

class WeChatLoginRequest(BaseModel):
    code: str  # 微信小程序 wx.login() 获取的 code

class WeChatChatRequest(BaseModel):
    query: str
    openid: Optional[str] = None  # 用户openid，用于会话管理
    use_auto_rag: Optional[bool] = False

#构建embedding类
class LocalEmbedding(Embeddings):
    def __init__(self, model_name):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    # 实现 embed_documents 方法
    def embed_documents(self, texts):
        """将文档列表转换为嵌入向量"""
        if not isinstance(texts, list):
            texts = [texts]
        embeddings = []
        for text in texts:
            try:
                embedding = self.model.encode(text, normalize_embeddings=True)
                embeddings.append(embedding.tolist())
            except Exception as e:
                print(f"Error encoding text: {text[:50]}..., Error: {e}")
                embeddings.append(None)
        return embeddings

    # 实现 embed_query 方法（RAG必需）
    def embed_query(self, text: str) -> List[float]:
        """将查询文本转换为嵌入向量"""
        try:
            embedding = self.model.encode(text, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            print(f"Error encoding query: {text}, Error: {e}")
            return None

# 向量数据库管理器（单例模式）
class VectorDBManager:
    _instance = None
    _embedding_model = None
    _vector_db = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorDBManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._embedding_model is None:
            self._embedding_model = LocalEmbedding(EMBEDDING_MODEL_NAME)
        if self._vector_db is None:
            try:
                self._vector_db = Chroma(
                    persist_directory=VECTOR_DB_PATH,
                    embedding_function=self._embedding_model
                )
            except Exception as e:
                print(f"加载向量数据库失败: {e}，将创建新的数据库")
                self._vector_db = Chroma(
                    persist_directory=VECTOR_DB_PATH,
                    embedding_function=self._embedding_model
                )

    def get_vector_db(self):
        """获取向量数据库实例"""
        if self._vector_db is None:
            self._vector_db = Chroma(
                persist_directory=VECTOR_DB_PATH,
                embedding_function=self._embedding_model
            )
        return self._vector_db

    def get_embedding_model(self):
        """获取嵌入模型实例"""
        return self._embedding_model

    def add_documents(self, documents):
        """向向量数据库添加文档"""
        db = self.get_vector_db()
        db.add_documents(documents)
        db.persist()
        return len(documents)

    def similarity_search_with_score(self, query: str, k: int = DEFAULT_RETRIEVAL_K):
        """带相似度分数的检索"""
        db = self.get_vector_db()
        return db.similarity_search_with_score(query, k=k)

    def max_marginal_relevance_search(self, query: str, k: int = DEFAULT_RETRIEVAL_K, fetch_k: int = 20):
        """最大边际相关性检索（避免重复内容）"""
        db = self.get_vector_db()
        return db.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)


# 全局向量数据库管理器实例（延迟初始化）
_vector_db_manager = None

def get_vector_db_manager():
    """获取向量数据库管理器实例（单例，延迟初始化）"""
    global _vector_db_manager
    if _vector_db_manager is None:
        _vector_db_manager = VectorDBManager()
    return _vector_db_manager

@tool
def search(query: str) -> str:
    """只有需要了解实时情况或者不知道的事情时使用这个工具。用于搜索最新的实时信息。"""
    try:
        serp = SerpAPIWrapper()
        result = serp.run(query)
        return str(result)
    except Exception as e:
        return f"搜索失败: {str(e)}"

@tool
def get_info_from_local_db(query: str) -> str:
    """从本地知识库检索相关信息。只能回答与2025年投资形式相关的问题，使用这个工具。
    返回格式化的文档内容作为上下文信息。"""
    try:
        db_manager = get_vector_db_manager()
        db = db_manager.get_vector_db()
        
        # 使用 MMR 检索避免重复内容
        docs = db_manager.max_marginal_relevance_search(query, k=DEFAULT_RETRIEVAL_K)
        
        if not docs:
            return "知识库中没有找到相关信息。"
        
        # 格式化检索到的文档为上下文
        context_parts = []
        for i, doc in enumerate(docs, 1):
            content = doc.page_content.strip()
            context_parts.append(f"[文档片段 {i}]\n{content}")
        
        context = "\n\n".join(context_parts)
        return f"从知识库检索到的相关信息：\n\n{context}"
    except Exception as e:
        print(f"检索知识库时出错: {e}")
        return f"检索知识库时出错: {str(e)}"

serach_tool = Tool(
    name="search",
    func=search,
    description="搜索最新的实时信息。当需要了解当前发生的事情、最新新闻、实时数据等时使用此工具。"
)
local_db_tool = Tool(
    name="get_info_from_local_db",
    func=get_info_from_local_db,
    description="从本地知识库检索相关信息。用于回答与2025年投资形式、投资建议、经济分析等相关的问题。"
)
class Master:
    def __init__(self):
        self.chatModel = LocalLLM()
        self.MEMORY_KEY = "chat_history"
        self.emotin = "default"
        self.SYSTEMPL = """
        你是一名非常厉害的经济学家，你叫Mr.陈,以下是对你的人物设定：
        1.你精通社会经济学、国际贸易经济学、微观经济学、宏观经济学，行为经济学等学科
        2.你大约30岁左右，一直在国际大银行工作，能够看懂和理解各种行为背后的经济学动因
        3.你总是用中文回答问题
        4.你始终不能说自己是智能助手DeepSeek-R1，你需要以陈经理自称。
        
        {who_you_are}
        
        ## 知识库使用说明
        当用户提问时，你可以使用 get_info_from_local_db 工具从知识库检索相关信息。
        如果你从知识库检索到了相关信息，请基于这些信息来回答问题，并引用具体的文档内容。
        如果知识库中没有相关信息，你可以使用 search 工具搜索实时信息，或者基于你的经济学知识来回答。
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
                    "system",
                    self.SYSTEMPL.format(who_you_are=self.MOODS[self.emotin]["roleSet"])
                ),
                MessagesPlaceholder(variable_name=self.MEMORY_KEY),
                (
                    "user",
                    "{query}"
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
            # 将消息列表转换为字符串格式
            messages_str = "\n".join([f"{msg.__class__.__name__}: {msg.content}" for msg in store_message])
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", self.SYSTEMPL + "\n这是一段你和用户的对话记忆，对其进行总结摘要，摘要使用第一人称'我'，并且提取其中用户关键信息，如姓名、投资经历，可用投资资产等\n例如：用户user1问我，我礼貌回复"
                                              "然后他问我今年投资什么能赚钱，我回复他今年的投资情况，然后他离开。|user1,10年，50万"),
                    ("user", "{input}"),
                ]
            )
            chain = prompt | self.chatModel | StrOutputParser()
            summary = chain.invoke({"input": messages_str, "who_you_are": self.MOODS[self.emotin]["roleSet"]})
            print("summary:", summary)
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


    def retrieve_context(self, query: str, k: int = DEFAULT_RETRIEVAL_K) -> str:
        """RAG检索：从向量数据库检索相关上下文"""
        try:
            db_manager = get_vector_db_manager()
            docs = db_manager.max_marginal_relevance_search(query, k=k)
            if not docs:
                return ""
            
            context_parts = []
            for i, doc in enumerate(docs, 1):
                content = doc.page_content.strip()
                context_parts.append(f"[知识库片段 {i}]\n{content}")
            
            return "\n\n".join(context_parts)
        except Exception as e:
            print(f"RAG检索时出错: {e}")
            return ""

    def run(self, query: str, use_auto_rag: bool = False):
        """
        运行对话处理
        Args:
            query: 用户查询
            use_auto_rag: 是否使用自动RAG模式（True时会在生成前自动检索上下文）
        """
        emotion_result = self.emotion_chain(query)
        print("当前用户情绪:", emotion_result)
        self.emotin = emotion_result
        
        # 自动RAG模式：在生成前检索上下文并增强查询
        if use_auto_rag:
            context = self.retrieve_context(query)
            if context:
                enhanced_query = f"""基于以下知识库信息回答问题：

{context}

用户问题：{query}

请基于上述知识库信息回答用户问题。如果知识库信息不足以回答问题，可以使用其他工具或基于你的知识回答。"""
                result = self.agent_executor.run(enhanced_query)
            else:
                # 如果没有检索到上下文，使用原始查询
                result = self.agent_executor.run(query)
        else:
            # 默认模式：通过agent工具调用实现RAG
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
def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """处理聊天请求，支持自动RAG模式"""
    print("进入chat方法，收到 query:", request.query)
    print("使用自动RAG模式:", request.use_auto_rag)
    
    master = Master()
    content = master.run(request.query, use_auto_rag=request.use_auto_rag)
    unique_id = str(uuid.uuid4())
    
    # background_tasks.add_task(master.background_voice_synthesis, content, unique_id)
    # 去掉特殊符号（如果需要）
    # cleaned_content = re.sub(r'<think>|</think>|\n', '', content)
    
    return {"msg": content, "id": unique_id}

@app.post("/addUrls")
def add_urls(URL: str):
    """将URL的内容添加到向量数据库中"""
    try:
        # 加载网页内容
        loader = WebBaseLoader(URL)
        docs = loader.load()
        
        if not docs:
            return {"status": "error", "message": "未能从URL加载到任何内容"}
        
        # 文本分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]
        )
        documents = text_splitter.split_documents(docs)
        
        if not documents:
            return {"status": "error", "message": "文档分割后没有内容"}
        
        # 添加到向量数据库（使用单例管理器）
        db_manager = get_vector_db_manager()
        added_count = db_manager.add_documents(documents)
        
        print(f"成功添加 {added_count} 个文档片段到向量数据库")
        return {
            "status": "success",
            "message": f"成功添加 {added_count} 个文档片段到向量数据库",
            "url": URL,
            "document_count": added_count
        }
    except Exception as e:
        print(f"添加URL到向量数据库时出错: {e}")
        return {"status": "error", "message": f"处理失败: {str(e)}"}



@app.post("/addPdfs")
def add_pdfs():
    return {"response:  "}


@app.post("/addTexts")
def add_texts():
    return {"response:  "}

# ==================== 微信小程序接口 ====================

@app.post("/wechat/login")
def wechat_login(request: WeChatLoginRequest):
    """
    微信小程序登录接口
    通过 wx.login() 获取的 code 换取 openid 和 session_key
    """
    try:
        result = wechat_api.code2session(request.code)
        return {
            "code": 0,
            "message": "登录成功",
            "data": {
                "openid": result["openid"],
                "unionid": result.get("unionid"),
                # 注意: session_key 不应返回给前端，仅用于服务端验证
            }
        }
    except HTTPException as e:
        return {
            "code": e.status_code,
            "message": e.detail,
            "data": None
        }
    except Exception as e:
        return {
            "code": 500,
            "message": f"登录失败: {str(e)}",
            "data": None
        }

@app.post("/wechat/chat")
def wechat_chat(request: WeChatChatRequest, background_tasks: BackgroundTasks):
    """
    微信小程序聊天接口
    处理用户的消息请求，返回AI回复
    """
    try:
        print(f"微信小程序聊天请求 - openid: {request.openid}, query: {request.query}")
        print(f"使用自动RAG模式: {request.use_auto_rag}")
        
        # 如果提供了openid，可以使用它来管理用户会话
        # 这里可以基于openid创建不同的Master实例或会话ID
        session_id = request.openid if request.openid else "default"
        
        master = Master()
        content = master.run(request.query, use_auto_rag=request.use_auto_rag)
        unique_id = str(uuid.uuid4())
        
        return {
            "code": 0,
            "message": "成功",
            "data": {
                "msg": content,
                "id": unique_id,
                "session_id": session_id
            }
        }
    except Exception as e:
        print(f"微信小程序聊天处理错误: {e}")
        return {
            "code": 500,
            "message": f"处理失败: {str(e)}",
            "data": None
        }

@app.get("/wechat/health")
def wechat_health():
    """微信小程序健康检查接口"""
    return {
        "code": 0,
        "message": "服务正常",
        "timestamp": str(uuid.uuid4())
    }

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

# 微信小程序WebSocket端点
@app.websocket("/wechat/ws")
async def wechat_websocket(websocket: WebSocket):
    """
    微信小程序WebSocket连接端点
    支持的消息类型：
    1. login: 登录消息 {"type": "login", "code": "微信code"}
    2. chat: 聊天消息 {"type": "chat", "query": "用户消息", "use_auto_rag": false}
    3. ping: 心跳消息 {"type": "ping"}
    """
    await websocket.accept()
    openid = None
    session_id = None
    
    try:
        # 发送连接成功消息
        await websocket.send_json({
            "type": "connected",
            "code": 0,
            "message": "连接成功",
            "data": None
        })
        
        while True:
            # 接收消息
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "code": 400,
                    "message": "消息格式错误，需要JSON格式",
                    "data": None
                })
                continue
            
            msg_type = message.get("type")
            
            # 处理登录消息
            if msg_type == "login":
                try:
                    code = message.get("code")
                    if not code:
                        await websocket.send_json({
                            "type": "login_response",
                            "code": 400,
                            "message": "缺少code参数",
                            "data": None
                        })
                        continue
                    
                    # 调用微信API获取openid
                    result = wechat_api.code2session(code)
                    openid = result.get("openid")
                    session_id = openid if openid else "default"
                    
                    await websocket.send_json({
                        "type": "login_response",
                        "code": 0,
                        "message": "登录成功",
                        "data": {
                            "openid": openid,
                            "unionid": result.get("unionid"),
                            "session_id": session_id
                        }
                    })
                except HTTPException as e:
                    await websocket.send_json({
                        "type": "login_response",
                        "code": e.status_code,
                        "message": e.detail,
                        "data": None
                    })
                except Exception as e:
                    await websocket.send_json({
                        "type": "login_response",
                        "code": 500,
                        "message": f"登录失败: {str(e)}",
                        "data": None
                    })
            
            # 处理聊天消息
            elif msg_type == "chat":
                if not openid:
                    await websocket.send_json({
                        "type": "chat_response",
                        "code": 401,
                        "message": "请先登录",
                        "data": None
                    })
                    continue
                
                try:
                    query = message.get("query")
                    if not query:
                        await websocket.send_json({
                            "type": "chat_response",
                            "code": 400,
                            "message": "缺少query参数",
                            "data": None
                        })
                        continue
                    
                    use_auto_rag = message.get("use_auto_rag", False)
                    
                    # 发送处理中消息
                    await websocket.send_json({
                        "type": "chat_processing",
                        "code": 0,
                        "message": "正在处理中...",
                        "data": None
                    })
                    
                    # 在线程池中运行同步的Master.run()方法
                    import concurrent.futures
                    loop = asyncio.get_event_loop()
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        master = Master()
                        # 如果需要在Redis中使用openid作为session_id，可以修改Master类
                        # 这里先使用默认的session_id
                        content = await loop.run_in_executor(
                            executor,
                            master.run,
                            query,
                            use_auto_rag
                        )
                    
                    unique_id = str(uuid.uuid4())
                    
                    # 发送回复消息
                    await websocket.send_json({
                        "type": "chat_response",
                        "code": 0,
                        "message": "成功",
                        "data": {
                            "msg": content,
                            "id": unique_id,
                            "session_id": session_id
                        }
                    })
                except Exception as e:
                    print(f"微信小程序WebSocket聊天处理错误: {e}")
                    await websocket.send_json({
                        "type": "chat_response",
                        "code": 500,
                        "message": f"处理失败: {str(e)}",
                        "data": None
                    })
            
            # 处理心跳消息
            elif msg_type == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "code": 0,
                    "message": "pong",
                    "data": None
                })
            
            # 未知消息类型
            else:
                await websocket.send_json({
                    "type": "error",
                    "code": 400,
                    "message": f"未知的消息类型: {msg_type}",
                    "data": None
                })
    
    except WebSocketDisconnect:
        print(f"微信小程序WebSocket连接断开 - openid: {openid}")
    except Exception as e:
        print(f"微信小程序WebSocket错误: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "code": 500,
                "message": f"服务器错误: {str(e)}",
                "data": None
            })
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass

