# Polymarket HFT Bot

Binance 실시간 시세와 Polymarket 오라클 간의 시차를 이용한 고빈도 매매(HFT) 봇.
5분 단기 시장과 15분 이상 장기 시장에 대해 이원화된 전략을 수행함.

## 핵심 기능

*   **이원화 전략 엔진 (Dual Strategy)**
    *   **스캘핑 모드 (5분 시장)**: 골든타임(T-4분 ~ T-2분) 내 Binance 모멘텀 급변 시 진입.
    *   **스윙 모드 (15분 이상)**: Adaptive Maker 전략으로 스프레드 확보 및 유동성 공급.
*   **미세구조 분석 (Microstructure)**
    *   오더북 불균형(OB) 및 변동성(Toxicity) 분석을 통한 진입 필터링.
*   **리스크 관리**
    *   Kelly Betting: 승률과 수익비에 따른 포지션 규모 조절.
    *   Inventory Skew: 포지션 쏠림 방지를 위한 호가 조절.
    *   Circuit Breaker: 과도한 변동성 발생 시 매매 중단.

## 아키텍처 및 로직

```mermaid
graph TD
    A[시작] --> B{시장 탐색};
    B --> |5분 시장| C[스캘핑 모드];
    B --> |15분+ 시장| D[스윙 모드];
    
    C --> C1{골든타임?};
    C1 --> |Yes| C2{강한 모멘텀?};
    C1 --> |No| B;
    
    D --> D1{만기 3분전?};
    D1 --> |No| D2{안정적 추세?};
    D1 --> |Yes| B;
    
    C2 --> E[진입 판단];
    D2 --> E;
    
    E --> F{수수료 제외<br>수익 구간?};
    F --> |Yes| G[지정가 주문<br>(Maker)];
    F --> |No| B;
    
    G --> H[포지션 관리];
    H --> |수익 실현| I[TAKE PROFIT];
    H --> |손절매| J[STOP LOSS];
    H --> |만기 임박| K[강제 청산];
```

## 설치 방법

1. 저장소 복제
   ```bash
   git clone https://github.com/사용자명/polymarket-hft-bot.git
   cd polymarket-hft-bot
   ```

2. 의존성 설치
   ```bash
   pip install -r requirements.txt
   ```

3. 환경 설정
   * 루트 경로에 `.env` 파일 생성 및 API 키 입력.
   ```env
   PK=your_private_key
   CLOB_API_KEY=your_api_key
   CLOB_API_SECRET=your_api_secret
   CLOB_API_PASSPHRASE=your_passphrase
   ```

## 실행 방법

봇 실행:
```bash
python main.py
```

## 주의사항

본 소프트웨어는 교육 목적으로 제공됨. 실거래 손실에 대한 책임은 사용자에게 있음.
