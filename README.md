# ML Character Chat (Story + Persona RAG)

## 개요
스토리(world/story) 기반 컨텍스트와 최근 대화 맥락을 결합하고, 고정 Persona 정보를 포함한 프롬프트를 구성하여 캐릭터 특화 LLM (예: Pygmalion)에게 질의하는 구조의 Chat 서비스 스캐폴딩입니다. RunPod serverless 환경의 handler 패턴을 고려해 작성되었습니다.

## 디렉토리 구조
```
.
├── handler.py                # RunPod serverless 엔트리포인트
├── scripts/
│   ├── build_index.py        # 향후 벡터 인덱스 빌드 스크립트 (현재 placeholder)
│   └── run_local.py          # 로컬 E2E 테스트 (Mock LLM)
├── data/
│   ├── mock/
│   │   ├── persona.json
│   │   ├── story.json
│   │   └── chat_history.json
│   └── indexes/              # (미래) 벡터 인덱스 저장 위치
├── src/
│   ├── config/
│   │   └── config.py         # 환경/설정
│   ├── core/
│   │   ├── schemas.py        # Pydantic 스키마
│   │   ├── persona.py        # Persona 로드 유틸
│   │   └── prompt_builder.py # 프롬프트 조립
│   ├── rag/
│   │   ├── chat_rag.py       # 최근 대화 맥락 RAG (stub)
│   │   └── story_rag.py      # 스토리 컨텍스트 RAG (stub)
│   ├── llm/
│   │   └── mock_llm.py       # 경량 Mock LLM 어댑터
│   ├── service/
│   │   └── orchestrator.py   # 전체 오케스트레이션
│   └── utils/                # (추가 예정) 공용 유틸
├── tests/
│   └── test_prompt_builder.py
└── requirements.txt
```

## 처리 흐름
1. (RunPod) `handler(event)` 호출 → `event['input']` 파싱
2. `ChatRequest` 스키마 검증
3. `retrieve_chat_context`로 최근 대화 일부 추출 (stub)
4. `retrieve_story_context`로 스토리 일부 추출 (stub)
5. `build_prompt`가 persona + 두 컨텍스트 + history + user message를 하나의 prompt로 조립
6. Mock LLM (또는 실제 LLM 어댑터) 호출
7. 최종 `ChatResponse` 반환 (reply, 사용된 컨텍스트, 메타 정보 포함)

## 로컬 실행 (Mock)
사전 준비: 가상환경 생성 및 의존성 설치
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

E2E 모의 실행:
```bash
python scripts/run_local.py
```

인덱스 placeholder 생성:
```bash
python scripts/build_index.py
```

## RunPod 배포 개요
- 엔트리: `handler.py` 의 `handler` 함수
- 이미지 빌드시 `requirements.txt` 설치 후 `handler.py` 포함
- 실제 LLM을 사용하려면 `src/llm/`에 HuggingFace 또는 OpenAI 호환 어댑터 추가

## 확장 계획 (Next Steps)
- [ ] 실제 벡터 스토어 (FAISS / Chroma) + 임베딩 생성 파이프라인
- [ ] 대화 메모리(장기)와 에피소드 메모리(단기) 분리 전략
- [ ] 프롬프트 토큰 길이 관리(슬라이싱 / 요약) 로직 추가
- [ ] LLM 어댑터: HuggingFace Transformers / OpenAI / vLLM / Text Generation Inference 등 선택
- [ ] Observability: 구조화 로깅 + latency 측정 + 에러 태깅
- [ ] 캐시 전략: persona/system block 캐싱, story context 캐싱

## 테스트
기본 테스트 실행:
```bash
pytest -q
```

## 라이선스
(필요 시 추가)

---
문의나 수정 방향 요청 시 주석 혹은 Issue 형태로 남겨주세요.
