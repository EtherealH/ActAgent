
import uvicorn
import threading
from bot.server import app
from bot.tele import run_telegram_bot
if __name__ == '__main__':
    # 启动 Telegram bot 线程
    t = threading.Thread(target=run_telegram_bot, daemon=True)
    t.start()
    uvicorn.run(app, host="127.0.0.1", port=8000)


