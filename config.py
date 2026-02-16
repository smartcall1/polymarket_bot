import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    PK = os.getenv("PK")
    CLOB_API_KEY = os.getenv("CLOB_API_KEY")
    CLOB_API_SECRET = os.getenv("CLOB_API_SECRET")
    CLOB_API_PASSPHRASE = os.getenv("CLOB_API_PASSPHRASE")
    
    MARKET_ID = os.getenv("MARKET_ID")
    ORDER_AMOUNT_USDC = float(os.getenv("ORDER_AMOUNT_USDC", 50.0)) # 주문당 50불로 상향
    SPREAD_PERCENT = float(os.getenv("SPREAD_PERCENT", 0.0001))
    
    # HFT/HFT 전용 설정
    TAKE_PROFIT_PCT = 0.02 # 2% 익절
    STOP_LOSS_PCT = 0.05   # 5% 손절
    MIN_EDGE = 0.005        # 진입 최소 우위 (0.01 -> 0.005로 낮춤)
    
    PAPER_TRADING = os.getenv("PAPER_TRADING", "True").lower() == "true"
    DEBUG_MODE = os.getenv("DEBUG_MODE", "True").lower() == "true"

config = Config()
