"""
Binance 실시간 가격 연동 모멘텀 엔진

핵심 원리:
- Binance API에서 BTC/ETH/SOL의 실시간 가격을 매초 수신
- 최근 N초간의 가격 변화를 추적하여 '추세(모멘텀)'를 판단
- 추세가 명확할 때만 Polymarket UPDOWN 시장에 진입
- 이것이 진짜 '엣지': 외부 가격이 이미 올라가고 있는데
  Polymarket 오즈가 아직 반영 안 됐을 때 선점하는 것
"""

import requests
import time
from collections import deque

class BinancePriceFeed:
    def __init__(self):
        # 최근 가격 히스토리 (최대 300개 = 약 10분간 데이터 확보)
        self.price_history = {
            'BTC': deque(maxlen=300),
            'ETH': deque(maxlen=300),
            'SOL': deque(maxlen=300),
        }
        self.symbols = {
            'BTC': 'BTCUSDT',
            'ETH': 'ETHUSDT', 
            'SOL': 'SOLUSDT',
        }
        self.last_fetch = 0
    
    def fetch_prices(self):
        """Binance에서 실시간 가격 조회 (무료, 인증 불필요)"""
        now = time.time()
        if now - self.last_fetch < 2:  # 2초 쿨다운
            return
        
        try:
            # Binance API로 BTC, ETH, SOL 현재가 한번에 조회
            url = "https://api.binance.com/api/v3/ticker/price"
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                prices = {p['symbol']: float(p['price']) for p in r.json()}
                for coin, symbol in self.symbols.items():
                    if symbol in prices:
                        self.price_history[coin].append({
                            'price': prices[symbol],
                            'time': now
                        })
                self.last_fetch = now
        except:
            pass
    
    def get_momentum(self, coin, lookback_seconds=300):
        """
        특정 코인의 모멘텀(추세 강도) 계산
        
        반환값:
        - momentum: -1.0 ~ +1.0 (양수=상승추세, 음수=하락추세)
        - confidence: 0.0 ~ 1.0 (신호의 확실성)
        - price_change_pct: 실제 가격 변화율(%)
        """
        history = self.price_history.get(coin, deque())
        if len(history) < 3:
            return 0.0, 0.0, 0.0
        
        now = time.time()
        recent = [h for h in history if now - h['time'] <= lookback_seconds]
        
        if len(recent) < 2:
            return 0.0, 0.0, 0.0
        
        # 가격 변화율 계산
        first_price = recent[0]['price']
        last_price = recent[-1]['price']
        price_change_pct = ((last_price - first_price) / first_price) * 100
        
        # 연속 상승/하락 횟수 계산 (추세의 일관성 체크)
        consecutive_up = 0
        consecutive_down = 0
        for i in range(1, len(recent)):
            if recent[i]['price'] > recent[i-1]['price']:
                consecutive_up += 1
                consecutive_down = 0
            elif recent[i]['price'] < recent[i-1]['price']:
                consecutive_down += 1
                consecutive_up = 0
        
        # 모멘텀 계산 (-1 ~ +1)
        if consecutive_up >= 2:
            momentum = min(1.0, consecutive_up * 0.25)
        elif consecutive_down >= 2:
            momentum = max(-1.0, -consecutive_down * 0.25)
        else:
            momentum = price_change_pct * 10  # 약한 신호
            momentum = max(-1.0, min(1.0, momentum))
        
        # 확신도: 연속 횟수가 높을수록 확신
        confidence = min(1.0, max(consecutive_up, consecutive_down) / 5.0)
        
        return round(momentum, 3), round(confidence, 3), round(price_change_pct, 4)
    
    def should_enter(self, coin, min_confidence=0.2):
        """
        진입 신호 판단
        
        반환: (should_enter, direction, confidence, price_change)
        """
        momentum, confidence, price_change = self.get_momentum(coin)
        
        if confidence < min_confidence:
            return False, None, confidence, price_change
        
        if momentum > 0.15:
            return True, 'UP', confidence, price_change
        elif momentum < -0.15:
            return True, 'DOWN', confidence, price_change
        
        return False, None, confidence, price_change
