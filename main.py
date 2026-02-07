
import uvicorn
import sys
from bot.server import app

if __name__ == '__main__':
    print("=" * 60)
    print("正在启动 AI 助手服务...")
    print("服务地址: http://0.0.0.0:8000")
    print("API 文档: http://0.0.0.0:8000/docs")
    print("按 CTRL+C 停止服务")
    print("=" * 60)
    
    try:
        # 启动 FastAPI 服务，支持微信小程序
        # 生产环境建议使用 gunicorn 或 uvicorn 作为 WSGI 服务器
        uvicorn.run(
            app, 
            host="0.0.0.0",  # 改为 0.0.0.0 以允许外部访问
            port=8000,
            reload=False,  # 生产环境建议设为 False
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n服务已停止")
        sys.exit(0)
    except Exception as e:
        print(f"\n启动服务时出错: {e}")
        sys.exit(1)


