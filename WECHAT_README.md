# 微信小程序接入说明

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

## API 接口说明

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

## 微信小程序前端代码示例

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
   - 添加你的后端服务器域名

2. **HTTPS**: 生产环境必须使用 HTTPS

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

### 使用 curl 测试接口

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

1. **CORS 错误**: 确保 CORS 中间件已正确配置
2. **登录失败**: 检查 AppID 和 AppSecret 是否正确
3. **请求超时**: 增加 `networkTimeout` 配置，或优化后端处理速度

