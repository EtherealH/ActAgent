import telebot
import urllib.parse
import requests
import json
import os
import asyncio

bot = telebot.TeleBot("7927820693")

@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id,'你好我是陈经理，请问有什么可以帮助您的吗')

@bot.message_handler(func=lambda message: True)
def echo_all(message):
    try:
        #encoded_text =urllib.parse.quote(message.text)
        response = requests.post(
            'http://localhost:8000/chat',
            json={"query": message.text},  # 这里传 JSON body
            timeout=100
        )
        print(response)
        if response.status_code == 200:
            aisay =json.loads(response.text)
            if"msg" in aisay:
                bot.reply_to(message,aisay["msg"])
                # audio_path = f"{aisay['id']}.mp3"
                # asyncio.run(check_audio(message,audio_path))
            else:
                bot.reply_to(message,"对不起，我不知道怎么回答你的问题")
    except requests.RequestException as e:
        bot.reply_to(message,"对不起，我不知道怎么回答你的问题")


async def check_audio(message,audio_path):
    while True:
        if os.path.exists(audio_path):
            with open(audio_path,'rb') as f:
                bot.send_audio(message.chat.id,f)
            os.remove(audio_path)
            break
        else:
            print("waiting")
            await asyncio.sleep(1) #使用asyncio.sleep(1)来等待1秒\

def run_telegram_bot():
    bot.infinity_polling()