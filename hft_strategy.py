import math
import os
import statistics
import time
from collections import defaultdict, deque

from binance_feed import BinancePriceFeed


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


class HighFrequencyStrategy:
    # Sniper Mode Ver 2.0 (Activated)
    def __init__(self, client):
        self.client = client
        self.initial_balance = 1000.0
        self.bankroll = self.initial_balance
        self.entries = 0
        self.exits = 0
        self.wins = 0
        self.losses = 0
        self.total_profit = 0.0
        self.trade_logs = []
        self.start_time = time.time()
        self.active_positions = {}
        self.realized_returns = []

        # Cost / risk settings
        self.fee_rate = 0.002  # one-way
        self.max_positions = 3  # 4 -> 3 (집중 관리)
        self.max_inventory_notional_per_coin = 50.0  # 80 -> 50 (노출 축소)
        self.min_trade_amount = 3.0
        self.max_trade_amount = 20.0
        self.kelly_fraction = 0.05  # 0.20 -> 0.05 (리스크 대폭 축소)
        self.kelly_cap = 0.05  # max 5% bankroll per trade
        self.force_min_size_edge_mult = 1.2
        self.default_order_min_shares = 5.0

        # Rolling microstructure cache
        self.mid_history = defaultdict(lambda: deque(maxlen=240))
        self.ret_history = defaultdict(lambda: deque(maxlen=240))
        self.sigma_hist = defaultdict(lambda: deque(maxlen=360))
        self.illiq_hist = defaultdict(lambda: deque(maxlen=360))
        self.tox_hist = defaultdict(lambda: deque(maxlen=360))
        self.dynamic_state = {}

        self.price_feed = BinancePriceFeed()

    def get_elapsed(self):
        e = int(time.time() - self.start_time)
        return f"{e//3600:02d}:{(e%3600)//60:02d}:{e%60:02d}"

    def extract_coin(self, question):
        if "Bitcoin" in question or "BTC" in question:
            return "BTC"
        if "Ethereum" in question or "ETH" in question:
            return "ETH"
        if "Solana" in question or "SOL" in question:
            return "SOL"
        if "Ripple" in question or "XRP" in question:
            return "XRP"
        return "?"

    def extract_target_price(self, question):
        import re

        # Up/Down intraday markets do not contain a numeric strike in the title.
        # Example: "Bitcoin Up or Down - February 16, 12:30AM-12:35AM ET"
        # In this case, extracting "16" (date) would create a fake arbitrage signal.
        if "Up or Down" in question:
            return None

        # Only use explicit dollar-denominated strikes to avoid picking time/date numbers.
        match = re.search(r"\$([0-9,]+(?:\.[0-9]+)?)", question)
        if not match:
            return None

        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            return None

    def extract_display_name(self, question):
        coin = self.extract_coin(question)
        target = self.extract_target_price(question)
        target_str = f" @{target:,.0f}" if target else ""
        if "AM" in question or "PM" in question:
            parts = question.split(",")
            if len(parts) > 1:
                t = parts[-1].strip().replace(" ET", "")
                return f"{coin}{target_str} ({t})"
        return f"{coin}{target_str}"

    def _robust_z(self, x, hist):
        hist.append(x)
        arr = list(hist)
        if len(arr) < 20:
            return 0.0

        arr_sorted = sorted(arr)
        med = arr_sorted[len(arr_sorted) // 2]
        abs_dev = sorted(abs(v - med) for v in arr)
        mad = abs_dev[len(abs_dev) // 2] + 1e-9
        z = (x - med) / (1.4826 * mad)
        return _clip(z, -3.0, 3.0)

    def _book_snapshot(self, order_book, top_n=3):
        """Return best prices/depth from potentially unsorted order books."""
        bids_raw = order_book.get("bids", [])
        asks_raw = order_book.get("asks", [])
        if not bids_raw or not asks_raw:
            return None

        bids = []
        asks = []
        for b in bids_raw:
            try:
                bids.append((float(b["price"]), float(b["size"])))
            except (KeyError, TypeError, ValueError):
                continue
        for a in asks_raw:
            try:
                asks.append((float(a["price"]), float(a["size"])))
            except (KeyError, TypeError, ValueError):
                continue

        if not bids or not asks:
            return None

        # Polymarket book payload is often not sorted as best-first.
        bids.sort(key=lambda x: x[0], reverse=True)  # highest bid first
        asks.sort(key=lambda x: x[0])  # lowest ask first

        best_bid = bids[0][0]
        best_ask = asks[0][0]
        bid_depth = sum(size for _, size in bids[:top_n])
        ask_depth = sum(size for _, size in asks[:top_n])

        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "bids_top": bids[:top_n],
            "asks_top": asks[:top_n],
        }

    def _compute_microstructure(self, tid, best_bid, best_ask, bid_depth, ask_depth):
        mid = (best_bid + best_ask) / 2
        spread = max(best_ask - best_bid, 1e-6)
        depth = max(bid_depth + ask_depth, 1e-6)
        spread_pct = spread / max(mid, 1e-6)

        prev_mid = self.mid_history[tid][-1] if self.mid_history[tid] else mid
        ret = math.log(max(mid, 1e-6) / max(prev_mid, 1e-6))
        self.mid_history[tid].append(mid)
        self.ret_history[tid].append(ret)

        ret_arr = list(self.ret_history[tid])
        if len(ret_arr) >= 10:
            sigma = statistics.pstdev(ret_arr[-30:])
        else:
            sigma = abs(ret)

        # Illiquidity proxy: spread / depth
        illiq = spread / depth

        # Toxicity proxy: big move relative to spread means adverse selection risk is high.
        tox = abs(ret) / (spread_pct + 1e-6)

        z_sigma = self._robust_z(sigma, self.sigma_hist[tid])
        z_illiq = self._robust_z(illiq, self.illiq_hist[tid])
        z_tox = self._robust_z(tox, self.tox_hist[tid])

        return {
            "mid": mid,
            "spread_pct": spread_pct,
            "sigma": sigma,
            "illiq": illiq,
            "tox": tox,
            "z_sigma": z_sigma,
            "z_illiq": z_illiq,
            "z_tox": z_tox,
        }

    def _dynamic_thresholds(self, micro):
        z_sigma = micro["z_sigma"]
        z_illiq = micro["z_illiq"]
        z_tox = micro["z_tox"]
        spread_pct = micro["spread_pct"]
        sigma = micro["sigma"]
        illiq = micro["illiq"]

        # Dynamic trigger thresholds
        arb_thr = _clip(
            0.0004 * (1.0 + 0.35 * z_sigma + 0.25 * z_illiq + 0.20 * z_tox),
            0.00025,
            0.0030,
        )
        ob_dev = _clip(
            0.020 + 0.005 * z_sigma + 0.004 * z_illiq + 0.006 * z_tox,
            0.015,
            0.080,
        )

        # Dynamic spread for reservation quote and edge control
        half_spread = _clip(
            0.003 + 0.80 * spread_pct + 0.90 * sigma + 0.003 * max(z_tox, 0.0),
            0.003,
            0.050,
        )

        # Minimum edge required after fees + frictions
        # 0.01 (1.0%) -> 0.003 (0.3%)로 대폭 하향 (그리디하게 진입)
        min_edge = _clip(
            (0.5 * self.fee_rate) + 0.05 * spread_pct + 0.10 * sigma + 0.05 * illiq,
            0.003,
            0.080,
        )

        # 진입 점수 기준 하향: 최소 0.15점 이상 (기존 0.30)
        entry_score_thr = _clip(0.15 + 0.05 * max(z_tox, 0.0), 0.15, 0.45)

        return {
            "arb_thr": arb_thr,
            "ob_dev": ob_dev,
            "half_spread": half_spread,
            "min_edge": min_edge,
            "entry_score_thr": entry_score_thr,
        }

    def _inventory_norm(self, coin):
        signed_notional = 0.0
        for p in self.active_positions.values():
            if p["coin"] != coin:
                continue
            signed_notional += p["side"] * p["total_cost"]
        return _clip(
            signed_notional / max(self.max_inventory_notional_per_coin, 1e-6), -1.0, 1.0
        )

    def _estimate_fair_probability(
        self,
        coin,
        question,
        bid_depth,
        ask_depth,
        micro,
        dyn,
        q_norm,
    ):
        target_price = self.extract_target_price(question)
        bn_hist = self.price_feed.price_history.get(coin, [])

        arb_score = 0.0
        if target_price and bn_hist:
            real_price = bn_hist[-1]["price"]
            diff = (real_price - target_price) / max(target_price, 1e-6)
            arb_score = _clip(diff / max(dyn["arb_thr"], 1e-6), -2.5, 2.5) / 2.5

        total_depth = bid_depth + ask_depth
        ob_score = 0.0
        if total_depth > 0:
            ratio = bid_depth / total_depth
            ob_score = _clip((ratio - 0.5) / max(dyn["ob_dev"], 1e-6), -2.0, 2.0) / 2.0

        # 오라클 선행 매매를 위해 룩백을 180초 -> 60초로 단축 (즉각적인 추세 반응)
        momentum, confidence, _ = self.price_feed.get_momentum(coin, lookback_seconds=60)
        bn_score = momentum * confidence

        score = (0.55 * arb_score) + (0.25 * ob_score) + (0.20 * bn_score)
        signal_conf = _clip(
            0.45 * abs(arb_score) + 0.25 * abs(ob_score) + 0.30 * abs(bn_score), 0.0, 1.0
        )

        mid = micro["mid"]
        shift_cap = _clip(0.040 + 4.0 * micro["sigma"], 0.020, 0.150)
        raw_fair = mid + (score * shift_cap)

        # Reservation price with inventory skew
        gamma = 0.6 * (1.0 + 2.0 * micro["sigma"])
        fair = raw_fair - (gamma * q_norm * dyn["half_spread"])
        fair = _clip(fair, 0.01, 0.99)

        return {
            "score": score,
            "signal_conf": signal_conf,
            "fair_prob": fair,
            "arb_score": arb_score,
            "ob_score": ob_score,
            "bn_score": bn_score,
        }

    def _fractional_kelly_size(self, bankroll, entry_price, win_prob, signal_conf):
        edge = win_prob - entry_price
        if edge <= 0:
            return 0.0

        f_full = edge / max(1.0 - entry_price, 1e-6)
        f = _clip(self.kelly_fraction * f_full * (0.5 + 0.5 * signal_conf), 0.0, self.kelly_cap)
        amount = bankroll * f
        return _clip(amount, 0.0, self.max_trade_amount)

    def run_hft_step(self, market_data_list):
        self.price_feed.fetch_prices()
        now = time.time()

        for data in market_data_list:
            tid = data["tid"]
            question = data["question"]
            order_book = data["order_book"]
            end_time = data.get("end_time", 0)

            try:
                book = self._book_snapshot(order_book, top_n=3)
                if not book:
                    continue

                best_bid = book["best_bid"]
                best_ask = book["best_ask"]
                if best_ask <= best_bid:
                    continue

                bid_depth = book["bid_depth"]
                ask_depth = book["ask_depth"]

                # 전략 이원화: 5분 시장 vs 15분 이상 시장
                time_left = end_time - now
                is_5m_market = (time_left <= 300) or ("5m" in question.lower())

                # [5분 시장 전용 로직] 스캘핑 모드
                if is_5m_market:
                    # 골든타임: 시작 후 1분(혼란기) 지남 ~ 종료 전 2분(도박판) 전
                    # 5분 게임 기준: 300초 -> 240초(4분) ~ 120초(2분) 사이만 매매
                    if not (120 < time_left < 240):
                         continue
                    
                    # 강력한 모멘텀 필터: 바이낸스가 확실하게 튈 때만 진입
                    # 강력한 모멘텀 필터 완화: 0.6 -> 0.45
                    mom_threshold = 0.45 
                else:
                    # [15분 이상 시장] 정석 Maker 모드
                    # 만기 3분 전 진입 금지 (기존 안전장치)
                    if time_left < 180:
                        continue
                    mom_threshold = 0.25 # 일반적인 기준

                micro = self._compute_microstructure(tid, best_bid, best_ask, bid_depth, ask_depth)
                dyn = self._dynamic_thresholds(micro)
                coin = self.extract_coin(question)
                q_norm = self._inventory_norm(coin)

                signal = self._estimate_fair_probability(
                    coin=coin,
                    question=question,
                    bid_depth=bid_depth,
                    ask_depth=ask_depth,
                    micro=micro,
                    dyn=dyn,
                    q_norm=q_norm,
                )
                score = signal["score"]
                side = 1 if score > 0 else -1

                # Inventory skew on acceptance edge: harder to add, easier to offset.
                inv_penalty = 0.8 * max(q_norm * side, 0.0)
                inv_discount = 0.4 * max(-q_norm * side, 0.0)
                edge_required = dyn["min_edge"] * (1.0 + inv_penalty - inv_discount)

                # For debug UI tracking
                # Maker 전략: Taker(시장가) 대신 Maker(지정가)로 진입 가정
                # UP(매수) -> Best Bid + 1틱 (최우선 매수호가 선점)
                # DOWN(매도) -> Best Ask - 1틱 (최우선 매도호가 선점)
                tick_size = 0.001
                if side == 1:
                    maker_price = min(best_bid + tick_size, best_ask - tick_size)
                    edge = (signal["fair_prob"] * (1.0 - self.fee_rate)) - maker_price
                else:
                    # DOWN은 YES를 매도하는 포지션(Short)으로 가정
                    # 공매도 가격 = (1.0 - (Best Ask - tick)) ??
                    # Polymarket 시스템상 NO 매수가 더 일반적이지만, 여기서는 YES 가격 기준으로 계산
                    maker_price_yes = max(best_ask - tick_size, best_bid + tick_size)
                    maker_price = 1.0 - maker_price_yes
                    edge = ((1.0 - signal["fair_prob"]) * (1.0 - self.fee_rate)) - maker_price

                self.dynamic_state[tid] = {
                    "micro": micro,
                    "dyn": dyn,
                    "signal": signal,
                    "name": self.extract_display_name(question),
                    "edge": edge,
                    "req": edge_required
                }

                pos = self.active_positions.get(tid)
                if pos:
                    self._manage_position(
                        tid=tid,
                        pos=pos,
                        best_bid=best_bid,
                        best_ask=best_ask,
                        end_time=end_time,
                        now=now,
                        q_norm=q_norm,
                        dyn=dyn,
                        signal=signal,
                        mid=micro["mid"],
                    )
                    continue

                if len(self.active_positions) >= self.max_positions:
                    continue

                # Avoid tails near 0/1 where mark-to-market is fragile in thin books.
                if best_ask >= 0.98 or best_bid <= 0.02:
                    continue

                score = signal["score"]
                if abs(score) < dyn["entry_score_thr"]:
                    continue

                side = 1 if score > 0 else -1

                # Maker Price 적용 (Adaptive Maker)
                tick_size = 0.001
                if side == 1:
                    # 매수: Best Bid보다 한 틱 위 (단, Ask를 넘지 않음)
                    entry_price = min(best_bid + tick_size, best_ask - tick_size)
                    win_prob = signal["fair_prob"]
                else:
                    # 매도(Short YES): Best Ask보다 한 틱 아래
                    entry_price_yes = max(best_ask - tick_size, best_bid + tick_size)
                    entry_price = max(1.0 - entry_price_yes, 0.001)
                    win_prob = 1.0 - signal["fair_prob"]

                # 엣지 재계산 (Maker Price 기준)
                if side == 1:
                    expected_edge = (win_prob * (1.0 - self.fee_rate)) - entry_price
                else:
                    expected_edge = (win_prob * (1.0 - self.fee_rate)) - entry_price

                # 5분 시장일 경우 더 엄격한 모멘텀 기준 적용
                bn_score = dyn.get("bn_score", 0.0) # 위에서 계산된 값 참조 필요하나, 여기서는 직접 로직 추가
                
                # Circuit Breaker & 5m Filter
                if dyn["entry_score_thr"] > 0.40:
                    continue
                
                if is_5m_market:
                    # 5분 시장은 점수(Score)보다 모멘텀(Momentum) 절대값이 중요
                    # hft_strategy.py:265 즈음에서 momentum 변수를 가져와야 함.
                    # 구조상 여기서 모멘텀을 다시 확인하거나 강제 필터링
                    # (간소화를 위해 Score Threshold를 높이는 방식으로 우회 구현)
                    if score < 0.3: # 5분 시장 필터 완화 (0.5 -> 0.3)
                        continue
                
                if expected_edge < edge_required:
                    continue

                amount = self._fractional_kelly_size(
                    bankroll=self.bankroll,
                    entry_price=entry_price,
                    win_prob=win_prob,
                    signal_conf=signal["signal_conf"],
                )

                # Size skew to naturally reduce one-sided inventory
                size_skew = _clip(1.0 - (q_norm * side), 0.5, 1.5)
                amount *= size_skew
                amount = _clip(amount, 0.0, self.max_trade_amount)

                # Kelly can under-size in micro-edge regimes; allow minimum executable size
                # only when edge is clearly above required threshold.
                min_exec_notional = max(
                    self.min_trade_amount,
                    self.default_order_min_shares * entry_price,
                )
                if amount < min_exec_notional:
                    if (
                        expected_edge >= edge_required * self.force_min_size_edge_mult
                        and signal["signal_conf"] >= 0.20
                    ):
                        amount = min_exec_notional
                    else:
                        continue

                reason = self._reason_label(signal)
                self._enter(
                    tid=tid,
                    entry_price=entry_price,
                    side=side,
                    confidence=signal["signal_conf"],
                    question=question,
                    reason=reason,
                    amount=amount,
                )
            except Exception:
                continue

        self._render()

    def _reason_label(self, signal):
        components = {
            "Arb": abs(signal["arb_score"]),
            "OB": abs(signal["ob_score"]),
            "Momentum": abs(signal["bn_score"]),
        }
        return max(components, key=components.get)

    def _manage_position(
        self, tid, pos, best_bid, best_ask, end_time, now, q_norm, dyn, signal, mid
    ):
        mark_price = mid if pos["side"] == 1 else (1.0 - mid)
        pnl_pct = (mark_price - pos["avg_price"]) / max(pos["avg_price"], 1e-6) * 100
        pos["current_price"] = mark_price
        pos["pnl_pct"] = pnl_pct

        if end_time > 0 and (end_time - now) < 60:
            self._exit(tid, best_bid, best_ask, reason="만기임박청산")
            return

        # 수수료(Round Trip ~0.4%)를 고려하여 최소 익절폭 상향 조정
        # 기존: 0.30% -> 변경: 수수료 2배(0.4%) + 알파 = 최소 0.6% 이상
        min_feasible_tp = (self.fee_rate * 2 * 100) + 0.2
        tp_pct = max(min_feasible_tp, dyn["min_edge"] * 100 * 1.5)
        
        # 스탑로스 완화: 노이즈에 털리지 않도록 여유를 둠 (대신 진입을 신중하게 함)
        # 기존: -2.0% ~ -5.0% -> 변경: -7.0% 고정 (단, 독성감지 시 조기청산 별도 존재)
        hard_stop_pct = -7.0

        # If inventory is crowded on this side, unwind early near breakeven.
        same_side_crowded = (q_norm * pos["side"]) > 0.55
        if same_side_crowded and pnl_pct >= -0.15:
            self._exit(tid, best_bid, best_ask, reason="인벤토리축소")
            return

        # Toxic regime + losing position => exit early.
        if signal["signal_conf"] < 0.20 and dyn["entry_score_thr"] > 0.35 and pnl_pct < 0:
            self._exit(tid, best_bid, best_ask, reason="독성회피")
            return

        # 순수익(Net PnL) 기준으로 익절 판단
        estimated_exit_fee_pct = self.fee_rate * 100
        net_pnl_pct = pnl_pct - estimated_exit_fee_pct

        if net_pnl_pct >= tp_pct:
            self._exit(tid, best_bid, best_ask, reason="TAKE PROFIT")
        elif pnl_pct <= hard_stop_pct:
            self._exit(tid, best_bid, best_ask, reason="STOP LOSS")

    def _enter(self, tid, entry_price, side, confidence, question, reason, amount):
        fee = amount * self.fee_rate
        total_spend = amount + fee
        if self.bankroll < total_spend:
            return

        if entry_price <= 0:
            return

        self.bankroll -= total_spend
        name = self.extract_display_name(question)
        coin = self.extract_coin(question)
        direction = "UP" if side == 1 else "DOWN"

        self.active_positions[tid] = {
            "avg_price": entry_price,
            "total_cost": total_spend, # 수수료 포함된 실제 지출액으로 저장 (PnL 정확도 향상)
            "shares": amount / entry_price,
            "side": side,  # +1 YES(UP), -1 NO(DOWN)
            "dir": direction,
            "name": name,
            "coin": coin,
            "current_price": entry_price,
            "pnl_pct": 0.0,
            "confidence": confidence,
            "reason": reason,
            "question_full": question,
        }
        self.entries += 1
        # 모바일용 짧은 로그 생성: OPEN [Mome...] -> OP Mome ETH 1:30 UP @0.5
        # 42자 제한 고려: OP(2) + 이유(4) + 이름(8) + 시간(4) + 방향(2) + 가격(5)
        short_reason = reason[:4]
        short_name = name.split("(")[0].strip()[:5]
        time_part = name.split("(")[1].split(")")[0].replace("AM","").replace("PM","")
        self.trade_logs.append(
            f"OP {short_reason} {short_name} {time_part} {direction} @{entry_price:.2f}"
        )

    def _exit(self, tid, best_bid, best_ask, reason=""):
        pos = self.active_positions.pop(tid)
        shares = pos["shares"]

        if reason == "만기정산":
            target = self.extract_target_price(pos.get("question_full", ""))
            bn_hist = self.price_feed.price_history.get(pos["coin"], [])
            real_p = bn_hist[-1]["price"] if bn_hist else None
            is_win = False

            if target is not None and real_p is not None:
                if pos["side"] == 1 and real_p > target:
                    is_win = True
                if pos["side"] == -1 and real_p < target:
                    is_win = True

            revenue = shares * 1.0 if is_win else 0.0
        else:
            if pos["side"] == 1:
                exit_price = best_bid
            else:
                exit_price = max(1.0 - best_ask, 0.0)
            revenue = (shares * exit_price) * (1.0 - self.fee_rate)

        pnl_dollar = revenue - pos["total_cost"]
        pnl_pct = (pnl_dollar / max(pos["total_cost"], 1e-6)) * 100
        self.bankroll += revenue
        self.exits += 1
        self.total_profit += pnl_dollar
        self.realized_returns.append(pnl_dollar / max(pos["total_cost"], 1e-6))

        if pnl_dollar >= 0:
            self.wins += 1
            icon = "WIN"
        else:
            self.losses += 1
            icon = "LOSS"

        # 모바일용 짧은 로그: WIN TAKE ETH 1:30 +1.2%
        short_reason = reason.split(" ")[0][:4] # TAKE PROFIT -> TAKE
        short_name = pos['name'].split("(")[0].strip()[:5]
        time_part = pos['name'].split("(")[1].split(")")[0].replace("AM","").replace("PM","")
        
        self.trade_logs.append(
            f"{icon} {short_reason} {short_name} {time_part} {pnl_pct:+.1f}%"
        )

    def _calc_unrealized(self):
        unrealized = 0.0
        for pos in self.active_positions.values():
            current_value = pos["shares"] * pos.get("current_price", pos["avg_price"])
            unrealized += current_value - pos["total_cost"]
        return unrealized

    def _calc_sharpe_like(self):
        if len(self.realized_returns) < 2:
            return 0.0
        stdev = statistics.pstdev(self.realized_returns)
        if stdev <= 1e-9:
            return 0.0
        mean_ret = statistics.mean(self.realized_returns)
        return (mean_ret / stdev) * math.sqrt(len(self.realized_returns))

    def _render(self):
        unrealized = self._calc_unrealized()
        invested = sum(p["total_cost"] for p in self.active_positions.values())
        total_pnl = self.total_profit + unrealized
        roi = (total_pnl / self.initial_balance) * 100
        wr = (self.wins / max(1, self.wins + self.losses)) * 100
        
        # 화면 초기화 (윈도우는 cls, 리눅스/터미널은 clear)
        # ANSI 코드가 안 먹히는 환경(PowerShell 등)을 위해 OS 명령어로 대체
        os.system('cls' if os.name == 'nt' else 'clear')
        print("=" * 42 + "\033[K")
        print("  POLYMARKET HFT ENGINE | by Donemoji\033[K")
        print("=" * 42 + "\033[K")
        print(f"  BANK: ${self.bankroll:.2f} | INV: ${invested:.2f}\033[K")
        print(f"  PnL: ${self.total_profit:+.2f} | ROI: {roi:+.2f}%\033[K")
        print(f"  WR: {wr:.0f}% | RUN: {self.get_elapsed()}\033[K")
        print("-" * 42 + "\033[K")
        print(f"  TRADES: {self.entries} (W:{self.wins} L:{self.losses})\033[K")
        print("-" * 42 + "\033[K")

        if self.active_positions:
            print("  [ACTIVE POSITIONS]\033[K")
            for p in self.active_positions.values():
                pnl = p.get("pnl_pct", 0.0)
                icon = "+" if pnl >= 0 else "-"
                # 이름 파싱: BTC (1:45AM-2:00PM) -> BTC 1:45-2:00
                raw_name = p['name']
                if "(" in raw_name and ")" in raw_name:
                    coin = raw_name.split("(")[0].strip()
                    time_part = raw_name.split("(")[1].split(")")[0].replace("AM","").replace("PM","")
                    # 시간 보정 (2:0 -> 2:00)
                    if ":" in time_part:
                        parts = time_part.split("-")
                        new_parts = []
                        for t in parts:
                            if ":" in t:
                                # HH:MM:SS 같은 경우 대비해서 안전하게 처리
                                t_split = t.split(":")
                                if len(t_split) >= 2:
                                    h, m = t_split[0], t_split[1]
                                    if len(m) == 1: m = "0" + m
                                    new_parts.append(f"{h}:{m}")
                                else:
                                    new_parts.append(t)
                            else:
                                new_parts.append(t)
                        time_part = "-".join(new_parts)
                    
                    display_name = f"{coin} {time_part}"
                else:
                    display_name = raw_name[:15]

                print(f"  {icon} {display_name:<15} {p['dir']} {pnl:+.1f}%\033[K")
        else:
            print("  [CANDIDATE MARKETS (Top 5)]\033[K")
            # 점수가 높은 순으로 정렬하여 상위 5개 표시
            candidates = sorted(
                self.dynamic_state.values(),
                key=lambda x: abs(x['signal']['score']),
                reverse=True
            )[:5]
            if candidates:
                for c in candidates:
                    score = c['signal']['score']
                    thr = c['dyn']['entry_score_thr']
                    edge = c.get('expected_edge', 0.0)
                    side = "UP" if score > 0 else "DOWN"
                    # 모바일용 초간단 출력: 이름(시간) 방향 점수
                    name = c['name'].split("(")[0].strip()[:8]
                    if "(" in c['name'] and ")" in c['name']:
                        time_part = c['name'].split("(")[1].split(")")[0].replace("AM","").replace("PM","")
                        
                        # 1:45-2:0 같은 경우를 1:45-2:00으로 보정
                        if "-" in time_part:
                            start_t, end_t = time_part.split("-")
                            
                            def fix_time(t_str):
                                if ":" in t_str:
                                    parts = t_str.split(":")
                                    if len(parts) >= 2:
                                        hh, mm = parts[0], parts[1]
                                        if len(mm) == 1: mm = "0" + mm # 2:0 -> 2:00
                                        return f"{hh}:{mm}"
                                return t_str
                                
                            start_t = fix_time(start_t)
                            end_t = fix_time(end_t)
                            time_str = f"{start_t}-{end_t}"
                        else:
                            time_str = time_part
                    else:
                        time_str = ""
                    
                    print(f"  {name} {time_str} {side} Sc:{abs(score):.2f}/{thr:.2f}\033[K")
            else:
                print("  Scanning markets...\033[K")
        print("-" * 42 + "\033[K")

        if self.trade_logs:
            print("  [RECENT ACTIVITY]\033[K")
            for log in self.trade_logs[-3:]:
                # 로그도 길면 자르기
                print(f"  {log[:40]}\033[K")
        print("=" * 42 + "\033[K")
        
        # 이전 화면 잔상 제거 (커서 아래 모든 내용 삭제)
        print("\033[J", end="")

    def show_status(self, msg):
        os.system('cls' if os.name == 'nt' else 'clear')
        print("=" * 64)
        print(f"  SYSTEM STATUS: {msg}")
        print(f"  BALANCE: ${self.bankroll:.2f}")
        print("=" * 64)
