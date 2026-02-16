from py_clob_client.client import ClobClient
from config import config
import requests
import time
import math
import json

class PolymarketClient:
    def __init__(self):
        self.authenticated = config.CLOB_API_KEY is not None and config.CLOB_API_KEY != "dummy"
        self.client = None
        if self.authenticated:
            try:
                self.client = ClobClient(
                    host="https://clob.polymarket.com",
                    key=config.CLOB_API_KEY,
                    secret=config.CLOB_API_SECRET,
                    passphrase=config.CLOB_API_PASSPHRASE,
                    private_key=config.PK
                )
            except Exception as e:
                print(f"[Client] Auth init failed: {e}")
        
    def get_order_book(self, market_id):
        """실시간 호가창 데이터를 가져옵니다."""
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json"
        }
        try:
            url = f"https://clob.polymarket.com/book?token_id={market_id}"
            response = requests.get(url, headers=headers, timeout=20)
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None

    def find_active_markets(self):
        """
        5분/15분 단위 UPDOWN 시장을 동적으로 탐색합니다.
        
        핵심 발견:
        - BTC: 5분(btc-updown-5m) + 15분(btc-updown-15m) 시장 모두 존재
        - ETH: 15분(eth-updown-15m)만 존재
        - SOL: 15분(sol-updown-15m)만 존재
        
        slug 패턴: {coin}-updown-{interval}-{unix_timestamp}
        timestamp는 해당 간격(5분=300초, 15분=900초)의 배수
        """
        now = int(time.time())
        
        # 코인별 사냥 리스트 (slug_prefix, interval_seconds)
        hunt_list = [
            ("btc-updown-5m", 300),     # BTC 5분
            ("btc-updown-15m", 900),    # BTC 15분
            ("eth-updown-15m", 900),    # ETH 15분 
            ("sol-updown-15m", 900),    # SOL 15분
            ("xrp-updown-15m", 900),    # XRP 15분 (추가됨)
        ]
        
        found_markets = []
        
        for slug_prefix, interval in hunt_list:
            # 현재 블록과 다음 블록 시도
            current_block = math.floor(now / interval) * interval
            next_block = current_block + interval
            
            for ts in [current_block, next_block]:
                slug = f"{slug_prefix}-{ts}"
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Accept": "application/json"
                }
                try:
                    url = f"https://gamma-api.polymarket.com/events?slug={slug}"
                    r = requests.get(url, headers=headers, timeout=15)
                    events = r.json()
                    if events:
                        ev = events[0]
                        for m in ev.get('markets', []):
                            tids_raw = m.get('clobTokenIds', [])
                            if isinstance(tids_raw, str):
                                tids_raw = json.loads(tids_raw)
                            if tids_raw:
                                # slug에서 timestamp 추출 (마지막 숫자 부분)
                                try:
                                    end_time = int(slug.split('-')[-1])
                                except:
                                    end_time = now + interval
                                    
                                found_markets.append({
                                    'question': m.get('question', ''),
                                    'clobTokenIds': tids_raw,
                                    'slug': slug,
                                    'end_time': end_time
                                })
                except:
                    continue
        
        return found_markets
