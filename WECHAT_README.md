# 微信小程序接入说明

## ⚠️ 重要提示

**推荐使用 WebSocket 协议进行对接**，WebSocket 提供更好的实时通信体验，支持长连接和双向通信。

- **WebSocket 端点**: `ws://your-domain.com/wechat/ws` (开发环境) 或 `wss://your-domain.com/wechat/ws` (生产环境)
- **HTTP API**: 仍然保留，但建议迁移到 WebSocket

## 环境配置

### 1. 设置微信小程序 AppID 和 AppSecret

在 `.env` 文件中设置环境变量，或在启动服务前设置：

**Windows PowerShell:**
```powershell
$env:WECHAT_APPID="你的AppID"
$env:WECHAT_APP_SECRET="你的AppSecret"
```

**Windows CMD:**
```cmd
set WECHAT_APPID=你的AppID
set WECHAT_APP_SECRET=你的AppSecret
```

**Linux/Mac:**
```bash
export WECHAT_APPID="你的AppID"
export WECHAT_APP_SECRET="你的AppSecret"
```

### 2. 启动服务

```bash
python main.py
```

服务将在 `http://0.0.0.0:8000` 启动

## WebSocket 接口说明（推荐）

### WebSocket 连接地址

- **开发环境**: `ws://localhost:8000/wechat/ws`
- **生产环境**: `wss://your-domain.com/wechat/ws` (必须使用 WSS)

### 消息协议

所有消息均为 JSON 格式，包含以下字段：
- `type`: 消息类型（必需）
- 其他字段根据消息类型而定

### 1. 连接建立

连接成功后，服务器会发送连接确认消息：

```json
{
  "type": "connected",
  "code": 0,
  "message": "连接成功",
  "data": null
}
```

### 2. 登录消息

**客户端发送:**
```json
{
  "type": "login",
  "code": "微信小程序 wx.login() 获取的 code"
}
```

**服务器响应:**
```json
{
  "type": "login_response",
  "code": 0,
  "message": "登录成功",
  "data": {
    "openid": "用户openid",
    "unionid": "用户unionid（如果存在）",
    "session_id": "会话ID"
  }
}
```

### 3. 聊天消息

**客户端发送:**
```json
{
  "type": "chat",
  "query": "用户的问题",
  "use_auto_rag": false  // 可选，是否使用自动RAG模式
}
```

**服务器响应（处理中）:**
```json
{
  "type": "chat_processing",
  "code": 0,
  "message": "正在处理中...",
  "data": null
}
```

**服务器响应（完成）:**
```json
{
  "type": "chat_response",
  "code": 0,
  "message": "成功",
  "data": {
    "msg": "AI回复内容",
    "id": "唯一ID",
    "session_id": "会话ID"
  }
}
```

### 4. 心跳消息

**客户端发送:**
```json
{
  "type": "ping"
}
```

**服务器响应:**
```json
{
  "type": "pong",
  "code": 0,
  "message": "pong",
  "data": null
}
```

### 5. 错误消息

```json
{
  "type": "error",
  "code": 400,
  "message": "错误描述",
  "data": null
}
```

## HTTP API 接口说明（已废弃，建议使用 WebSocket）

### 1. 登录接口

**接口地址:** `POST /wechat/login`

**请求参数:**
```json
{
  "code": "微信小程序 wx.login() 获取的 code"
}
```

**返回示例:**
```json
{
  "code": 0,
  "message": "登录成功",
  "data": {
    "openid": "用户openid",
    "unionid": "用户unionid（如果存在）"
  }
}
```

### 2. 聊天接口

**接口地址:** `POST /wechat/chat`

**请求参数:**
```json
{
  "query": "用户的问题",
  "openid": "用户openid（可选，用于会话管理）",
  "use_auto_rag": false  // 可选，是否使用自动RAG模式
}
```

**返回示例:**
```json
{
  "code": 0,
  "message": "成功",
  "data": {
    "msg": "AI回复内容",
    "id": "唯一ID",
    "session_id": "会话ID"
  }
}
```

### 3. 健康检查接口

**接口地址:** `GET /wechat/health`

**返回示例:**
```json
{
  "code": 0,
  "message": "服务正常",
  "timestamp": "时间戳"
}
```

## 微信小程序前端代码示例（WebSocket 方式）

### 1. WebSocket 连接和登录

```javascript
// pages/chat/chat.js
Page({
  data: {
    messages: [],
    inputValue: '',
    openid: '',
    loading: false,
    socketTask: null,
    connected: false
  },

  onLoad() {
    this.connectWebSocket()
  },

  onUnload() {
    // 页面卸载时关闭WebSocket连接
    if (this.data.socketTask) {
      this.data.socketTask.close()
    }
  },

  // 连接WebSocket
  connectWebSocket() {
    const wsUrl = 'wss://your-domain.com/wechat/ws' // 替换为你的服务器地址
    const socketTask = wx.connectSocket({
      url: wsUrl,
      success: () => {
        console.log('WebSocket连接成功')
      },
      fail: (err) => {
        console.error('WebSocket连接失败', err)
        wx.showToast({
          title: '连接失败，请重试',
          icon: 'none'
        })
      }
    })

    // 监听连接打开
    socketTask.onOpen(() => {
      console.log('WebSocket已连接')
      this.setData({ connected: true, socketTask })
      
      // 连接成功后自动登录
      this.login()
    })

    // 监听消息
    socketTask.onMessage((res) => {
      try {
        const message = JSON.parse(res.data)
        this.handleMessage(message)
      } catch (e) {
        console.error('解析消息失败', e)
      }
    })

    // 监听连接关闭
    socketTask.onClose(() => {
      console.log('WebSocket连接已关闭')
      this.setData({ connected: false })
    })

    // 监听错误
    socketTask.onError((err) => {
      console.error('WebSocket错误', err)
      wx.showToast({
        title: '连接错误',
        icon: 'none'
      })
    })

    this.setData({ socketTask })
  },

  // 处理收到的消息
  handleMessage(message) {
    switch (message.type) {
      case 'connected':
        console.log('服务器确认连接')
        break
      
      case 'login_response':
        if (message.code === 0) {
          this.setData({
            openid: message.data.openid
          })
          wx.setStorageSync('openid', message.data.openid)
          wx.showToast({
            title: '登录成功',
            icon: 'success'
          })
        } else {
          wx.showToast({
            title: message.message || '登录失败',
            icon: 'none'
          })
        }
        break
      
      case 'chat_processing':
        this.setData({ loading: true })
        break
      
      case 'chat_response':
        this.setData({ loading: false })
        if (message.code === 0) {
          const aiMessage = {
            type: 'ai',
            content: message.data.msg
          }
          this.setData({
            messages: [...this.data.messages, aiMessage]
          })
        } else {
          wx.showToast({
            title: message.message || '请求失败',
            icon: 'none'
          })
        }
        break
      
      case 'pong':
        console.log('收到心跳响应')
        break
      
      case 'error':
        wx.showToast({
          title: message.message || '发生错误',
          icon: 'none'
        })
        break
    }
  },

  // 微信登录
  login() {
    wx.login({
      success: (res) => {
        if (res.code) {
          // 通过WebSocket发送登录消息
          this.sendMessage({
            type: 'login',
            code: res.code
          })
        }
      },
      fail: (err) => {
        console.error('wx.login 失败', err)
      }
    })
  },

  // 发送WebSocket消息
  sendMessage(message) {
    if (!this.data.socketTask || !this.data.connected) {
      wx.showToast({
        title: '未连接，请重试',
        icon: 'none'
      })
      return
    }

    this.data.socketTask.send({
      data: JSON.stringify(message),
      success: () => {
        console.log('消息发送成功', message)
      },
      fail: (err) => {
        console.error('消息发送失败', err)
      }
    })
  },

  // 输入框内容变化
  onInput(e) {
    this.setData({
      inputValue: e.detail.value
    })
  },

  // 发送聊天消息
  sendChatMessage() {
    const query = this.data.inputValue.trim()
    if (!query) {
      wx.showToast({
        title: '请输入消息',
        icon: 'none'
      })
      return
    }

    if (!this.data.openid) {
      wx.showToast({
        title: '请先登录',
        icon: 'none'
      })
      return
    }

    // 添加用户消息到聊天记录
    const userMessage = {
      type: 'user',
      content: query
    }
    this.setData({
      messages: [...this.data.messages, userMessage],
      inputValue: ''
    })

    // 通过WebSocket发送聊天消息
    this.sendMessage({
      type: 'chat',
      query: query,
      use_auto_rag: false
    })
  },

  // 发送心跳（可选，用于保持连接）
  sendPing() {
    this.sendMessage({ type: 'ping' })
  }
})
```

### 2. 小程序配置文件

**app.json:**
```json
{
  "pages": [
    "pages/index/index",
    "pages/chat/chat"
  ],
  "window": {
    "backgroundTextStyle": "light",
    "navigationBarBackgroundColor": "#fff",
    "navigationBarTitleText": "AI助手",
    "navigationBarTextStyle": "black"
  },
  "networkTimeout": {
    "request": 60000,
    "connectSocket": 60000
  }
}
```

### 3. 小程序 app.js 配置

```javascript
// app.js
App({
  globalData: {
    wsUrl: 'wss://your-domain.com/wechat/ws' // 替换为你的服务器地址
  },

  onLaunch() {
    // 小程序启动时的逻辑
  }
})
```

## HTTP API 前端代码示例（已废弃）

### 1. 登录获取 openid

```javascript
// pages/index/index.js
Page({
  data: {
    openid: '',
    userInfo: null
  },

  onLoad() {
    this.login()
  },

  // 微信登录
  login() {
    wx.login({
      success: (res) => {
        if (res.code) {
          // 调用后端登录接口
          wx.request({
            url: 'https://your-domain.com/wechat/login', // 替换为你的服务器地址
            method: 'POST',
            data: {
              code: res.code
            },
            success: (res) => {
              if (res.data.code === 0) {
                this.setData({
                  openid: res.data.data.openid
                })
                // 将 openid 存储到本地
                wx.setStorageSync('openid', res.data.data.openid)
              } else {
                wx.showToast({
                  title: '登录失败',
                  icon: 'none'
                })
              }
            },
            fail: (err) => {
              console.error('登录请求失败', err)
              wx.showToast({
                title: '网络错误',
                icon: 'none'
              })
            }
          })
        }
      },
      fail: (err) => {
        console.error('wx.login 失败', err)
      }
    })
  }
})
```

### 2. 发送消息与AI对话

```javascript
// pages/chat/chat.js
Page({
  data: {
    messages: [],
    inputValue: '',
    openid: '',
    loading: false
  },

  onLoad() {
    // 获取 openid
    const openid = wx.getStorageSync('openid')
    if (openid) {
      this.setData({ openid })
    }
  },

  // 输入框内容变化
  onInput(e) {
    this.setData({
      inputValue: e.detail.value
    })
  },

  // 发送消息
  sendMessage() {
    const query = this.data.inputValue.trim()
    if (!query) {
      wx.showToast({
        title: '请输入消息',
        icon: 'none'
      })
      return
    }

    // 添加用户消息到聊天记录
    const userMessage = {
      type: 'user',
      content: query
    }
    this.setData({
      messages: [...this.data.messages, userMessage],
      inputValue: '',
      loading: true
    })

    // 调用聊天接口
    wx.request({
      url: 'https://your-domain.com/wechat/chat', // 替换为你的服务器地址
      method: 'POST',
      data: {
        query: query,
        openid: this.data.openid,
        use_auto_rag: false
      },
      success: (res) => {
        this.setData({ loading: false })
        
        if (res.data.code === 0) {
          // 添加AI回复到聊天记录
          const aiMessage = {
            type: 'ai',
            content: res.data.data.msg
          }
          this.setData({
            messages: [...this.data.messages, aiMessage]
          })
        } else {
          wx.showToast({
            title: res.data.message || '请求失败',
            icon: 'none'
          })
        }
      },
      fail: (err) => {
        this.setData({ loading: false })
        console.error('聊天请求失败', err)
        wx.showToast({
          title: '网络错误',
          icon: 'none'
        })
      }
    })
  }
})
```

### 3. 小程序配置文件

**app.json:**
```json
{
  "pages": [
    "pages/index/index",
    "pages/chat/chat"
  ],
  "window": {
    "backgroundTextStyle": "light",
    "navigationBarBackgroundColor": "#fff",
    "navigationBarTitleText": "AI助手",
    "navigationBarTextStyle": "black"
  },
  "networkTimeout": {
    "request": 60000
  }
}
```

### 4. 小程序 app.js 配置

```javascript
// app.js
App({
  globalData: {
    apiBaseUrl: 'https://your-domain.com' // 替换为你的服务器地址
  },

  onLaunch() {
    // 小程序启动时的逻辑
  }
})
```

## 注意事项

1. **域名配置**: 在微信公众平台的小程序后台配置服务器域名
   - 登录设置 -> 开发设置 -> 服务器域名
   - **socket合法域名**: 添加你的 WebSocket 服务器域名（如: `wss://your-domain.com`）
   - **request合法域名**: 如果仍使用 HTTP API，需要添加 HTTP 服务器域名

2. **HTTPS/WSS**: 生产环境必须使用 HTTPS 和 WSS（WebSocket Secure）
   - 开发环境可以使用 `ws://` 和 `http://`
   - 生产环境必须使用 `wss://` 和 `https://`

3. **WebSocket 连接管理**:
   - 建议在页面 `onLoad` 时建立连接，`onUnload` 时关闭连接
   - 实现断线重连机制
   - 定期发送心跳消息保持连接活跃

3. **Session Key 安全**: 
   - `session_key` 不应返回给前端
   - 仅在服务端用于数据解密和验证

4. **错误处理**: 
   - 实现完善的错误处理机制
   - 处理网络超时、服务器错误等情况

5. **会话管理**: 
   - 可以通过 `openid` 来管理不同用户的会话
   - 可以基于 `openid` 在 Redis 中存储用户对话历史

## 测试

### 使用 WebSocket 客户端测试

可以使用在线 WebSocket 测试工具或编写测试脚本：

**Python 测试脚本示例:**
```python
import asyncio
import websockets
import json

async def test_websocket():
    uri = "ws://localhost:8000/wechat/ws"
    async with websockets.connect(uri) as websocket:
        # 接收连接确认
        response = await websocket.recv()
        print("连接确认:", response)
        
        # 发送登录消息
        login_msg = {
            "type": "login",
            "code": "test_code"  # 替换为真实的code
        }
        await websocket.send(json.dumps(login_msg))
        response = await websocket.recv()
        print("登录响应:", response)
        
        # 发送聊天消息
        chat_msg = {
            "type": "chat",
            "query": "你好",
            "use_auto_rag": False
        }
        await websocket.send(json.dumps(chat_msg))
        
        # 接收处理中消息
        response = await websocket.recv()
        print("处理中:", response)
        
        # 接收回复消息
        response = await websocket.recv()
        print("聊天回复:", response)

asyncio.run(test_websocket())
```

### 使用 curl 测试 HTTP 接口（已废弃）

```bash
# 测试健康检查
curl http://localhost:8000/wechat/health

# 测试登录（需要真实的 code）
curl -X POST http://localhost:8000/wechat/login \
  -H "Content-Type: application/json" \
  -d '{"code": "test_code"}'

# 测试聊天
curl -X POST http://localhost:8000/wechat/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "你好", "openid": "test_openid"}'
```

## 常见问题

1. **WebSocket 连接失败**: 
   - 检查服务器地址是否正确（开发环境使用 `ws://`，生产环境使用 `wss://`）
   - 确认在微信小程序后台配置了 socket 合法域名
   - 检查网络连接和防火墙设置

2. **CORS 错误**: 确保 CORS 中间件已正确配置（仅HTTP API需要）

3. **登录失败**: 检查 AppID 和 AppSecret 是否正确

4. **消息发送失败**: 
   - 确保 WebSocket 连接已建立（检查 `connected` 状态）
   - 确保已成功登录（检查 `openid` 是否存在）

5. **连接断开**: 
   - 实现自动重连机制
   - 定期发送心跳消息保持连接
   - 检查服务器是否正常运行

6. **请求超时**: 
   - 增加 `networkTimeout` 配置中的 `connectSocket` 超时时间
   - 优化后端处理速度

