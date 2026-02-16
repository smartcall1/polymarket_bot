import time
import json
from config import config

def main():
    print("=== 폴리마켓 초단타 봇 (5분/15분 UPDOWN 전용) ===")
    print(f"모드: {'가상 시뮬레이션' if config.PAPER_TRADING else '실전'}")
    
    from client_wrapper import PolymarketClient
    from hft_strategy import HighFrequencyStrategy
    
    try:
        client = PolymarketClient()
    except Exception as e:
        if config.PAPER_TRADING:
            print("[주의] API 클라이언트 초기화 실패 (가상 모드로 계속)")
            client = None
        else:
            print(f"[에러] 클라이언트 초기화 실패: {e}")
            return

    strategy = HighFrequencyStrategy(client)
    
    try:
        active_tokens = []
        last_search = 0
        search_interval = 30  # 30초마다 시장 재탐색 (5분/15분 블록 즉각 대응)
        
        while True:
            try:
                now = time.time()
                
                # 일정 간격마다 시장을 다시 탐색 (5분 시장이 계속 새로 생기니까)
                if not active_tokens or (now - last_search) > search_interval:
                    markets = client.find_active_markets()
                    active_tokens = []
                    for m in markets:
                        tids = m.get('clobTokenIds', [])
                        if isinstance(tids, str):
                            tids = json.loads(tids)
                        if tids:
                            active_tokens.append({
                                'tid': tids[0], 
                                'question': m.get('question', '?'),
                                'slug': m.get('slug', ''),
                                'end_time': m.get('end_time', 0)
                            })
                    last_search = now
                    
                    if not active_tokens:
                        strategy.show_status("시장 탐색 중...")
                        time.sleep(10)
                        continue

                # 각 시장의 실시간 데이터 수집
                sim_data = []
                for item in active_tokens:
                    order_book = client.get_order_book(item['tid'])
                    if order_book:
                        sim_data.append({
                            'tid': item['tid'],
                            'question': item['question'],
                            'order_book': order_book,
                            'end_time': item.get('end_time', 0)
                        })
                
                if sim_data:
                    strategy.run_hft_step(sim_data)
                else:
                    # 모든 시장 실패 시 즉시 재탐색 유도
                    active_tokens = []
                    last_search = 0
                    strategy.show_status("데이터 수집 실패 - 시장 재탐색 중...")
                    
            except Exception as e:
                if config.DEBUG_MODE:
                    print(f"[Error] {e}")
            
            time.sleep(2)  # 2초 간격으로 호가 감시
            
    except KeyboardInterrupt:
        print("\n=== Bot Stopped by User ===")

if __name__ == "__main__":
    main()
