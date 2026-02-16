import time
import pandas as pd
from config import config

class TradingStrategy:
    def __init__(self, client):
        self.client = client
        self.initial_balance = 1000.0
        # 여러 토큰의 포지션을 관리할 수 있도록 변경
        self.inventory = {"USDC": self.initial_balance, "POSITIONS": {}} 
        self.pnl_history = []
        self.start_time = time.time()
        self.total_pnl = 0.0
        
    def get_elapsed_time_str(self):
        """방금 가동을 시작한 후 경과된 시간 문자열 반환"""
        elapsed_seconds = int(time.time() - self.start_time)
        hours = elapsed_seconds // 3600
        minutes = (elapsed_seconds % 3600) // 60
        seconds = elapsed_seconds % 60
        return f"{hours:02d}시간 {minutes:02d}분 {seconds:02d}초"

    def calculate_pnl(self, current_prices):
        """여러 시장의 가상 자산 가치를 합산하여 PnL 및 APR 산출"""
        position_value = 0.0
        for tid, price in current_prices.items():
            shares = self.inventory["POSITIONS"].get(tid, 0.0)
            position_value += shares * price
            
        current_value = self.inventory["USDC"] + position_value
        self.total_pnl = current_value - self.initial_balance
        
        elapsed_seconds = time.time() - self.start_time
        if elapsed_seconds > 0:
            growth_rate = self.total_pnl / self.initial_balance
            apr = (growth_rate / elapsed_seconds) * (365 * 24 * 3600) * 100
        else:
            apr = 0.0
            
        return self.total_pnl, apr

    def run_simulation_step(self, market_data_list):
        """여러 시장 데이터를 한번에 처리하여 가상 체결 진행"""
        current_prices = {}
        
        for data in market_data_list:
            tid = data['tid']
            question = data['question']
            order_book = data['order_book']
            
            if not order_book or 'bids' not in order_book or not order_book['bids']:
                continue

            best_bid = float(order_book['bids'][0]['price'])
            best_ask = float(order_book['asks'][0]['price'])
            mid_price = (best_bid + best_ask) / 2
            current_prices[tid] = mid_price

            my_bid = mid_price * (1 - config.SPREAD_PERCENT)
            my_ask = mid_price * (1 + config.SPREAD_PERCENT)

            # 매수 체결 확인 (해당 시장에 포지션이 없을 때만)
            if best_bid >= my_bid and self.inventory["USDC"] >= config.ORDER_AMOUNT_USDC:
                if self.inventory["POSITIONS"].get(tid, 0) == 0:
                    shares = config.ORDER_AMOUNT_USDC / my_bid
                    self.inventory["USDC"] -= config.ORDER_AMOUNT_USDC
                    self.inventory["POSITIONS"][tid] = shares
                    print(f"\n[체결] {question[:20]}... | 매수 ({my_bid:.4f})")

            # 매도 체결 확인
            shares = self.inventory["POSITIONS"].get(tid, 0)
            if best_ask <= my_ask and shares > 0:
                sell_amount = shares * my_ask
                self.inventory["USDC"] += sell_amount
                self.inventory["POSITIONS"][tid] = 0
                print(f"\n[체결] {question[:20]}... | 매도 ({my_ask:.4f})")

        current_pnl, current_apr = self.calculate_pnl(current_prices)
        elapsed_str = self.get_elapsed_time_str()
        self.pnl_history.append({"timestamp": time.time(), "pnl": current_pnl, "apr": current_apr})
        
        # \r을 사용하여 한 줄에서 계속 업데이트 (이벤트 발생 시에는 위에서 \n으로 줄바꿈됨)
        status_line = f"[{elapsed_str}] 수익: ${current_pnl:.2f} (APR: {current_apr:.1f}%) | 잔고: ${self.inventory['USDC']:.1f} | 감시: {len(market_data_list)}개"
        print(f"\r{status_line}", end="", flush=True)

    def get_pnl_summary(self):
        return pd.DataFrame(self.pnl_history)
