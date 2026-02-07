"""
微信小程序API接口模块
提供微信小程序登录、聊天等功能
"""
import requests
import json
import hashlib
import os
from typing import Optional, Dict
from fastapi import HTTPException


class WeChatAPI:
    """微信小程序API封装类"""
    
    def __init__(self):
        # 微信小程序配置 - 从环境变量读取
        self.appid = os.getenv("WECHAT_APPID", "")
        self.app_secret = os.getenv("WECHAT_APP_SECRET", "")
        self.api_url = "https://api.weixin.qq.com"
        
        if not self.appid or not self.app_secret:
            print("警告: 微信小程序 AppID 或 AppSecret 未设置，相关功能将不可用")
            print("请设置环境变量: WECHAT_APPID 和 WECHAT_APP_SECRET")
    
    def code2session(self, code: str) -> Dict:
        """
        微信小程序登录凭证校验
        通过 wx.login() 获取的 code 换取 openid 和 session_key
        
        Args:
            code: 微信小程序 wx.login() 获取的 code
            
        Returns:
            {
                "openid": "用户唯一标识",
                "session_key": "会话密钥",
                "unionid": "用户在开放平台的唯一标识符（可选）"
            }
        """
        if not self.appid or not self.app_secret:
            raise HTTPException(
                status_code=500,
                detail="微信小程序配置未完成，请联系管理员"
            )
        
        url = f"{self.api_url}/sns/jscode2session"
        params = {
            "appid": self.appid,
            "secret": self.app_secret,
            "js_code": code,
            "grant_type": "authorization_code"
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            result = response.json()
            
            if "errcode" in result and result["errcode"] != 0:
                error_msg = result.get("errmsg", "未知错误")
                raise HTTPException(
                    status_code=400,
                    detail=f"微信登录失败: {error_msg} (错误码: {result['errcode']})"
                )
            
            return {
                "openid": result.get("openid"),
                "session_key": result.get("session_key"),
                "unionid": result.get("unionid"),
                "errcode": 0,
                "errmsg": "ok"
            }
        except requests.RequestException as e:
            raise HTTPException(
                status_code=500,
                detail=f"请求微信API失败: {str(e)}"
            )
    
    def check_signature(self, session_key: str, raw_data: str, signature: str) -> bool:
        """
        验证数据签名
        
        Args:
            session_key: 会话密钥
            raw_data: 原始数据
            signature: 签名
            
        Returns:
            True 如果签名正确，False 否则
        """
        sign_str = raw_data + session_key
        calculated_sign = hashlib.sha1(sign_str.encode('utf-8')).hexdigest()
        return calculated_sign == signature
    
    def decrypt_user_info(self, session_key: str, encrypted_data: str, iv: str) -> Optional[Dict]:
        """
        解密用户信息（使用 AES-128-CBC 解密）
        注意: 这里需要实现解密逻辑，实际使用时建议使用专门的加密库
        
        Args:
            session_key: 会话密钥
            encrypted_data: 加密数据
            iv: 初始向量
            
        Returns:
            解密后的用户信息字典
        """
        # TODO: 实现 AES-128-CBC 解密
        # 这里仅作示例，实际使用时需要实现完整的解密逻辑
        # 可以使用 cryptography 库或其他加密库
        pass


# 全局微信API实例
wechat_api = WeChatAPI()

