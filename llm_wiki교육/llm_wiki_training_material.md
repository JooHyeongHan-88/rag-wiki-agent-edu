# LLM Wiki 완전 정복 — AI Agent 시대의 지식 계층 구축·운영 실무 교안

> **RAG로는 부족했던 "구조화된 지식"을, Agent가 직접 만들고 읽고 고치게 만드는 기술**
>
> 본 문서는 **Python · 딥러닝 기초 · Transformer · Embedding · Vector DB · RAG · LangChain**(또는 유사 프레임워크)을 이미 사용해 본 **엔지니어**를 대상으로 합니다.
> LLM·RAG·임베딩의 *기초*는 다시 설명하지 않습니다. 그 위에 **LLM Wiki라는 지식 계층(knowledge layer)을 직접 구축·운영하는 실무**를 쌓습니다.
>
> 설명 : 실무 ≈ **40 : 60**. 모든 아키텍처는 하나의 구체적 코드베이스(`acme-billing`)와 "Claude Code를 페어로 쓰는 팀"의 실제 시나리오에 묶어서 설명합니다.

---

## 📚 이 문서를 시작하기 전에

### 학습 목표

이 교육을 마치면 여러분은 다음을 **직접 설명하고, 직접 구축할 수 있게** 됩니다.

- [ ] LLM Wiki가 **왜 등장했는가** (RAG의 한계는 정확히 무엇인가)
- [ ] RAG와 **무엇이 다른가** (검색 단위·중복·관계·운영비용)
- [ ] LLM Wiki의 **핵심 철학**: "AI가 읽기 위한 지식 계층"
- [ ] Wiki가 **단순 문서 저장소와 다른 이유**
- [ ] **Raw / Wiki / Schema** 3계층 구조와 각 계층의 입력·출력
- [ ] **Ingest / Query / Lint** 세 파이프라인의 단계별 동작
- [ ] Agent가 Wiki를 **생성하는 방식**과 **사용하는 방식**
- [ ] Wiki를 **운영**하고 **품질을 유지**하는 방법 (거버넌스·HITL·freshness)
- [ ] 실제 프로젝트(모노레포·엔터프라이즈 문서 통합 포함)에 **적용하는 방법**
- [ ] Claude Code / Codex / Cursor 같은 코딩 에이전트에서 Wiki를 **읽고·검색하고·수정**하게 만드는 법

### 이 문서를 읽는 법 — 약속된 기호

| 기호 | 의미 |
|------|------|
| 💡 | **비유 / 직관** — 추상 개념을 익숙한 것에 빗댐 (엔지니어 대상이어도 비유는 유지합니다) |
| 📌 | **핵심 포인트** — 반드시 기억할 것 |
| ⚠️ | **함정 / 안티패턴** — 현업에서 실제로 터지는 실패 |
| 🔧 | **실무 팁** — 운영·튜닝 시 참고 |
| 🧪 | **실무 시나리오** — `acme-billing`을 Claude Code로 운영하는 팀의 구체적 업무 흐름 |
| 💻 | **코드** — 거의 실행 가능한 실제 라이브러리 기반 스니펫 (pseudo 아님) |
| 🧭 | **한 줄 요약** — 각 섹션 결론 |

### 🧵 이 문서를 관통하는 단 하나의 예시 — `acme-billing`

RAG 교육이 "출산휴가 며칠인가요?"라는 **하나의 질문**을 끝까지 따라갔듯, 이 문서는 **하나의 코드베이스와 하나의 질문**을 20개 섹션 내내 관통합니다.

**코드베이스: `acme-billing`** — 구독 결제 백엔드 (SaaS의 청구 모듈)

```
acme-billing/                      # FastAPI + SQLAlchemy 2.0 + Alembic + Redis + Celery
├── app/
│   ├── api/
│   │   ├── subscriptions.py       # POST /subscriptions/{id}/renew, /cancel
│   │   └── webhooks.py            # POST /webhooks/stripe
│   ├── services/
│   │   ├── invoice.py             # InvoiceService.issue_invoice()
│   │   ├── subscription.py        # SubscriptionService.renew()
│   │   └── refund.py              # RefundPolicy.compute()
│   ├── tasks/
│   │   └── billing.py             # Celery: issue_invoice_task  (구 FastAPI BackgroundTasks)
│   └── models/
│       ├── subscription.py        # Subscription, SubscriptionStatus
│       ├── invoice.py             # Invoice
│       └── payment.py             # Payment, Customer
├── migrations/versions/
│   └── 0042_add_invoice_idempotency_key.py
└── docs/adr/
    ├── ADR-017-idempotent-invoice-issuance.md
    └── ADR-021-move-background-tasks-to-celery.md
```

**관통 질문 (이 문서의 "출산휴가 며칠인가요?")**

> 🧵 **"구독을 갱신할 때 인보이스가 중복 발행되지 않도록, 코드 어디에서 어떻게 보장하나?"**

이 질문이 좋은 이유: 정답이 **한 파일에 없습니다.** `엔드포인트 → 서비스 → Celery 태스크 → 모델 → 마이그레이션 → Redis 키 → ADR → Slack 사고 기록**에 흩어져 있고, 이들의 **관계를 따라가야** 답이 됩니다. 바로 이 "관계 추적"이 RAG와 LLM Wiki를 가르는 지점입니다. 같은 질문을 Section 1(왜 RAG로 부족한가), Section 9(Query), Section 13(Demo), Section 16(Claude Code)에서 반복해서 풀어 봅니다.

**팀 설정** — `acme-billing`을 운영하는 5인 백엔드 팀. 전원 Claude Code를 페어 프로그래밍 도구로 사용. 🧪 시나리오는 모두 이 팀 기준입니다.

### 전체 목차

| # | 섹션 | 핵심 |
|---|------|------|
| 1 | [LLM Wiki가 등장한 배경](#section-1) | RAG의 4대 한계 + Agent 시대의 지식 문제 |
| 2 | [LLM Wiki란 무엇인가](#section-2) | Compression·Distillation·Structured Knowledge |
| 3 | [RAG vs LLM Wiki 상세 비교](#section-3) | 8개 항목 비교표 |
| 4 | [Core Architecture — 3계층](#section-4) | Raw ↓ Wiki ↓ Schema |
| 5 | [Raw Layer](#section-5) | Agent가 읽는 원천 |
| 6 | [Wiki Layer](#section-6) | Knowledge/Concept Page·Relationship·Canonical |
| 7 | [Schema Layer](#section-7) | Entity·Relationship·Metadata·Constraints (JSON) |
| 8 | [Ingest Pipeline](#section-8) | Extract→Normalize→Merge→Dedup→Update |
| 9 | [Query Pipeline](#section-9) | Intent→Search→Traversal→Context→Answer |
| 10 | [Lint Pipeline](#section-10) | Broken Link·Stale·Duplicate·Schema Violation |
| 11 | [Agent-Driven Construction](#section-11) | 새 repo → Wiki 자동 생성 → PR |
| 12 | [Agent-Driven Maintenance](#section-12) | 커밋 → 영향분석 → 갱신 → Lint |
| 13 | [Demo 1 — FastAPI 백엔드](#section-13) | `acme-billing` Wiki 생성 전과정 |
| 14 | [Demo 2 — 대규모 모노레포](#section-14) | 코드·아키텍처·의존성 분석 |
| 15 | [Demo 3 — 엔터프라이즈 문서 통합](#section-15) | Notion·ADR·Slack·API Docs |
| 16 | [Claude Code / Codex / Cursor 활용](#section-16) | Agent가 Wiki를 읽고·검색하고·고친다 |
| 17 | [구현 예제 (Python)](#section-17) | Generator·Linter·Query Engine·Repo→Wiki |
| 18 | [운영 전략](#section-18) | HITL·Approval·Ownership·Governance·Freshness |
| 19 | [실패 사례](#section-19) | Wiki 폭증·중복·Hallucination·Schema Drift |
| 20 | [Best Practices](#section-20) | 기업 환경 기준 정리 |
| — | [부록](#appendix) | 용어·체크리스트·참고 |

---

<a id="section-1"></a>
## Section 1. LLM Wiki가 등장한 배경 — "왜 RAG만으로는 충분하지 않은가"

RAG는 지난 몇 년간 "LLM에 우리 지식을 주입"하는 사실상의 표준이었습니다. 여러분도 이미 `Loader → Splitter → Embedding → Vector Store → Retriever → Prompt → LLM` 파이프라인을 짜 봤을 겁니다. 그런데 그 RAG 시스템을 **Agent**(스스로 여러 턴에 걸쳐 코드를 읽고 고치는)에게 물려주는 순간, 네 가지 한계가 동시에 터집니다.

먼저 한계를 **관통 예시로** 체감해 봅시다. `acme-billing`의 RAG 인덱스는 코드·ADR·Slack·Notion을 전부 청크로 쪼개 벡터DB에 넣어 둔 상태입니다. 여기에 관통 질문을 던집니다.

> 🧵 "구독 갱신 시 인보이스 중복 발행을 어떻게 막나?"

RAG retriever가 `k=8`로 가져온 청크는 대략 이렇습니다.

```
[chunk 1] app/api/subscriptions.py  L40-78   renew_subscription() 일부
[chunk 2] ADR-017 초안 v1            "멱등키를 DB unique 제약으로…" (폐기된 안)
[chunk 3] ADR-017 최종              "멱등키를 Redis SETNX로…"      (채택된 안)
[chunk 4] Slack #billing-incidents  "또 중복 인보이스 떴어요" (2025-11, 사고 당시)
[chunk 5] app/services/invoice.py   issue_invoice() 일부 (단, Celery 이전의 옛 시그니처)
[chunk 6] tests/test_invoice.py     test_double_issue (스킵 처리된 테스트)
[chunk 7] Notion "결제팀 온보딩"     "인보이스는 BackgroundTasks로…" (ADR-021 이전 설명)
[chunk 8] app/models/invoice.py     Invoice 컬럼 정의 일부
```

이 8개 청크를 그대로 프롬프트에 넣으면 어떤 일이 생기는지가 Section 1의 주제입니다.

### 1-1. Context Explosion — 청크를 "더 많이 넣을수록" 나빠진다

Agent는 한 번 답하고 끝이 아니라 **수십 턴**을 돕니다. 매 턴마다 RAG가 8~20개 청크를 새로 끌어와 컨텍스트에 쌓습니다. 같은 `renew_subscription` 코드가 5번 다른 청크로 반복 등장하고, 폐기된 ADR 초안과 최종안이 **둘 다** 들어옵니다.

토큰으로 환산하면 체감이 됩니다. 실제로 재 봅시다.

> 💻 **RAG 청크 묶음이 매 턴 먹는 토큰 — `count_tokens`로 실측** (추정치 금지, 실제 API로)

```python
import anthropic

client = anthropic.Anthropic()

# RAG가 한 턴에 끌어온 8개 청크를 이어 붙인 컨텍스트 (예: 평균 350토큰 × 8 + 중복)
rag_context = "\n\n".join(open(f"chunks/{i}.txt").read() for i in range(8))

resp = client.messages.count_tokens(
    model="claude-opus-4-8",
    messages=[{"role": "user", "content": rag_context}],
)
print(resp.input_tokens)   # 예: ~3,200 토큰 / 턴

# 40턴 에이전트 세션이면 같은 류의 청크를 계속 재인출 → 누적 12만+ 토큰,
# 그중 상당수가 "동일 코드/폐기된 ADR"의 중복.
```

📌 **핵심:** 컨텍스트 윈도우가 1M으로 커진 게 문제를 해결하지 못합니다. (1) 입력 토큰은 곧 비용이고, (2) 길수록 느리며, (3) **Lost-in-the-middle**로 정작 핵심을 흘립니다. RAG는 "검색 단위가 청크"라서, Agent의 긴 세션에서 **같은 지식을 중복해서, 여러 형태로, 계속 다시** 퍼 올립니다. 이것이 Context Explosion입니다.

🧪 **시나리오:** 신입 D가 Claude Code로 "환불 로직 좀 봐줘"를 시킵니다. 에이전트는 세션마다 결제 도메인 청크를 처음부터 다시 긁고, 매번 "인보이스가 BackgroundTasks인지 Celery인지"를 코드에서 재추론합니다. 같은 80k 토큰의 재발견을 **모든 세션이 반복**합니다. 팀의 토큰 청구서가 선형으로 늘고, 답의 일관성은 세션마다 출렁입니다.

### 1-2. Knowledge Duplication — 같은 사실이 N벌로 흩어진다

`acme-billing`에서 "인보이스 발행은 멱등해야 한다"는 **하나의 사실**이 물리적으로 몇 군데에 있을까요?

```
① app/services/invoice.py 의 코드 (Redis SETNX)
② ADR-017 (채택안)        + ADR-017 초안 v1 (폐기안, 그러나 git/위키에 남음)
③ Slack 사고 스레드        (사고 당시 맥락)
④ Notion 온보딩 문서       (옛 설명: BackgroundTasks — 지금은 틀림)
⑤ tests/test_invoice.py    (검증 코드)
⑥ docstring / 주석         (또 다른 표현)
```

RAG는 이 **중복을 그대로** 인덱싱합니다. 임베딩이 비슷하니 검색 시 6개가 한꺼번에 딸려 오고, 그중 ②의 폐기안과 ④의 옛 설명은 **현재 사실과 모순**됩니다. Agent는 모순된 근거를 받고 "BackgroundTasks로 처리됩니다"라고 자신 있게 틀린 답을 합니다.

> 💡 **비유 — 정리 안 된 위키 vs 사수의 인수인계 노트**
>
> RAG는 도서관의 **복사기**입니다. 키워드/벡터가 맞는 페이지를 닥치는 대로 복사해 줍니다 — 초판·개정판·낙서·찢어진 메모까지 섞여서. LLM Wiki는 그 책들을 **다 읽고 정리한 사수의 인수인계 위키**입니다. 중복을 합치고, 모순을 해소해 "현재 맞는 한 문장(canonical)"으로 만들고, "이건 저것과 연결됨"이라는 지도까지 붙여 둡니다.

📌 **핵심:** RAG에는 **"이 사실의 정본(canonical)이 무엇인가"라는 개념이 없습니다.** 모든 청크가 동등하게 검색 후보일 뿐입니다.

### 1-3. Fragmented Retrieval — 관계를 따라가지 못한다

관통 질문의 진짜 정답은 이런 모양입니다.

```
renew_subscription (엔드포인트)
   └─ 호출→ SubscriptionService.renew()
              └─ enqueue→ issue_invoice_task (Celery)        ← ADR-021로 BackgroundTasks에서 이전됨
                            └─ 사용→ InvoiceService.issue_invoice()
                                       └─ 멱등 보장→ Redis SETNX  key=idem:invoice:{sub_id}:{period}
                                       └─ 쓰기→ Invoice 모델       ← migration 0042가 idempotency_key 컬럼 추가
   (근거: ADR-017,  사고 발단: Slack #billing-incidents 2025-11)
```

이건 **그래프 순회**입니다. 그런데 RAG의 검색 단위는 "독립된 청크"라서, `renew_subscription` 청크를 찾아도 거기서 **`issue_invoice_task`로 한 발 더 건너뛸 수단이 없습니다.** 벡터 유사도는 "renew와 의미가 비슷한 것"을 줄 뿐, "renew가 **호출하는** 것"을 주지 못합니다. 결국 Agent는 조각만 받고 관계는 매 턴 코드에서 다시 추론합니다.

⚠️ **함정:** "임베딩을 더 좋게 / 청크를 더 크게 / k를 더 크게" 튜닝해도 이 문제는 안 풀립니다. 본질이 **유사도 검색 ≠ 관계 순회**이기 때문입니다. (이래서 GraphRAG 같은 접근이 나왔고, LLM Wiki는 그 "관계"를 **1급 시민**으로 끌어올립니다 — Section 7.)

### 1-4. Long Context의 한계 — "전부 넣기"는 답이 아니다

"그럼 1M 컨텍스트에 레포 전체를 넣자"는 유혹이 있습니다. `acme-billing`은 작아서 가능할 수도 있죠. 하지만 (1) 모노레포(Section 14)는 수백만 줄이라 애초에 안 들어가고, (2) 들어가도 매 턴 풀 비용이며, (3) Lost-in-the-middle로 정확도가 오히려 떨어지고, (4) **모델이 매번 같은 통찰을 재유도**합니다 — "아, 인보이스는 멱등하구나"를 세션마다 처음부터 깨닫습니다.

📌 핵심: 필요한 건 "원문을 더 많이"가 아니라 **"이미 소화된 지식을 압축된 형태로"**. 이것이 Knowledge Compression(Section 2)이고, 프롬프트 캐싱과 결합하면 비용까지 잡힙니다(Section 9·16).

### 1-5. Agent 시대의 Knowledge Problem — 문제의 본질이 바뀌었다

RAG는 **"사람이 한 번 질문 → 한 번 답"** 을 위해 설계됐습니다. 그런데 코딩 에이전트의 작업 양상은 다릅니다.

| | 사람 Q&A (RAG가 가정한 세계) | 코딩 에이전트 (실제 세계) |
|---|---|---|
| 턴 수 | 1~2턴 | 수십~수백 턴 |
| 같은 지식 재사용 | 거의 없음 | **매 세션·매 에이전트가 반복** |
| 필요한 것 | "관련 문단" | "정본 사실 + 관계 + 출처" |
| 비용 구조 | 질문당 1회 | 누적·반복 → 폭증 |
| 답의 일관성 | 그때그때 | **팀 전체·시간에 걸쳐 일관**되어야 |

📌 **결론(왜 RAG로 부족한가):** RAG는 *검색* 문제를 풀지만, Agent 시대의 진짜 문제는 *지식 관리(knowledge management)* 입니다. 한 번 소화한 지식을 **정본화하고, 관계를 박제하고, 중복을 제거하고, 최신으로 유지하고, 모든 에이전트가 싸게 재사용**하게 만드는 것 — RAG에는 이 레이어가 통째로 빠져 있습니다. LLM Wiki는 바로 그 빠진 레이어입니다.

### 🧭 Section 1 한 줄 요약

> RAG는 *청크를 검색*한다. 하지만 Agent 시대에는 **Context Explosion·Duplication·Fragmented Retrieval·Long-context 한계**가 한꺼번에 터진다. 본질은 검색이 아니라 **지식 관리** — 소화된 지식을 정본화·관계화·정제·최신화해서 모든 에이전트가 싸게 재사용하게 만드는 계층이 필요하다.

---

<a id="section-2"></a>
## Section 2. LLM Wiki란 무엇인가 — "AI가 읽기 위한 지식 계층"

### 2-1. 한 문장 정의

> 📌 **LLM Wiki** = 원천 자료(Raw)를 **AI가 읽기 좋게 소화·압축·구조화한, 정본(canonical)·무중복·상호연결된 지식 계층.** 사람이 보조로 읽을 수 있지만, **1차 독자는 Agent**다.

핵심은 강조점입니다. 위키피디아는 *사람이 읽기 위한* 지식 계층입니다. LLM Wiki는 *Agent가 읽기 위한* 지식 계층입니다. 그래서 형식이 다릅니다 — 장황한 산문 대신 **정본 진술 + 명시적 관계 링크 + 출처(provenance)**, 그리고 그 위에 **기계가 순회·검증할 수 있는 그래프(Schema)**.

### 2-2. 세 가지 핵심 개념

**① Knowledge Compression (지식 압축)**

같은 사실의 N벌(코드+ADR+Slack+Notion)을 **한 페이지로 압축**합니다. RAG가 8청크 3,200토큰으로 답하던 것을, Wiki는 1페이지 ~400토큰으로 답합니다.

> 💡 **비유 — 소스 코드 vs 빌드 산출물.** Raw 자료는 *소스*입니다. Wiki는 그 소스를 "지식 컴파일러(=Agent + 사람 리뷰)"로 빌드한 **산출물**입니다. 손으로 쓰는 게 아니라 **재생성(regenerate)됩니다.** 소스가 바뀌면 다시 빌드하죠(Section 12). 이 "지식은 빌드된다"는 관점이 이 문서 전체의 멘탈 모델입니다.

**② Knowledge Distillation (지식 증류)**

압축이 "양을 줄이는 것"이라면 증류는 **"모순을 해소하고 정본을 세우는 것"** 입니다. ADR-017 초안(폐기)과 최종안이 충돌하면 → "현재 채택안은 Redis SETNX"라고 한 줄로 못 박고, 폐기안은 `superseded_by`로 표시합니다. Notion의 옛 "BackgroundTasks" 설명은 → "ADR-021로 Celery 이전됨, 옛 설명 폐기"로 갱신합니다.

**③ Structured Knowledge (구조화된 지식)**

Wiki는 자유 텍스트가 아니라 **구조**를 가집니다. 각 페이지는 타입(Concept/Knowledge), 정본 진술, **명시적 관계 링크(`[[...]]`)**, 출처 메타데이터를 가집니다. 이 구조의 *기계 표현*이 Schema Layer(Section 7)이고, 덕분에 **관계 순회**와 **Lint**(품질 자동 검증)가 가능해집니다.

### 2-3. `acme-billing`의 Wiki 페이지 — 실제 모양

말로만 하면 추상적이니, 관통 예시의 **정본 Concept 페이지**를 미리 봅니다. (Section 6에서 형식을 자세히 다룹니다.)

> 💻 `wiki/concepts/idempotent-invoice-issuance.md`

```markdown
---
id: concept.idempotent-invoice-issuance
type: concept
title: 멱등 인보이스 발행 (Idempotent Invoice Issuance)
status: canonical
owners: [team-billing]
last_verified: 2026-06-10
sources:                      # provenance — Raw로 역추적
  - code: app/services/invoice.py#issue_invoice
  - adr: docs/adr/ADR-017-idempotent-invoice-issuance.md
  - incident: slack://C0BILL/p1699...   # 2025-11 중복 인보이스 사고
related:                      # 관계 = 1급 시민
  - implemented_by: "[[InvoiceService]]"
  - triggered_by: "[[renew_subscription]]"
  - guarded_by: "[[redis-idempotency-key]]"
  - persisted_to: "[[Invoice]]"
  - superseded_decision: "[[ADR-017-draft-v1]]"   # DB unique 제약 안 (폐기)
---

## 정본 진술 (Canonical)
구독 갱신·웹훅 재시도로 `issue_invoice`가 중복 호출돼도 인보이스는 **기간(period)당 1건**만 생성된다.
멱등성은 **Redis `SETNX key=idem:invoice:{subscription_id}:{period}` (TTL 24h)** 로 보장한다.
DB unique 제약(초안 v1)은 경합 시 500을 유발해 **폐기**됐다(ADR-017).

## 동작 (요약)
1. `renew_subscription` → `SubscriptionService.renew()` → Celery `issue_invoice_task` enqueue (ADR-021)
2. 태스크가 `InvoiceService.issue_invoice(sub_id, period)` 호출
3. `SETNX` 성공 시에만 `Invoice` 생성. 실패(=이미 발행)면 no-op 반환.

## 주의 (Gotchas)
- `period`는 **UTC 월초** 기준. 타임존 버그 주의.
- BackgroundTasks 시절 설명(Notion 온보딩)은 **틀림** → Celery로 이전됨.
```

📌 한 페이지에 **정본 사실 + 관계 + 출처 + 함정**이 다 있습니다. Agent가 이 페이지 하나(~400토큰)만 읽으면, RAG가 8청크로 헤매던 답을 **모순 없이, 출처와 함께** 즉시 얻습니다.

### 2-4. 이건 새로운 게 아니다 — 이미 쓰고 있는 패턴의 일반화

LLM Wiki는 갑자기 등장한 단일 제품이 아니라, 업계에서 따로따로 자라던 패턴을 **하나의 아키텍처로 종합**한 것입니다. 이미 익숙할 실제 사례들:

| 실제 사례 | LLM Wiki의 어느 조각인가 |
|---|---|
| **`CLAUDE.md` / `AGENTS.md`** (에이전트가 읽는 레포 가이드) | 수작업 Wiki 페이지의 원시 형태 |
| **Cursor Rules / `.cursor/rules`** | 도메인별 정본 지식 주입 |
| **DeepWiki류 "레포 → 위키 자동 생성"** | Agent-Driven Construction (Section 11) |
| **Microsoft GraphRAG** (엔티티·관계 그래프 + 커뮤니티 요약) | Schema Layer + 관계 순회 (Section 7·9) |
| **Anthropic Skills** (필요 시 로드되는 전문 지식 폴더) | 점진적 공개(progressive disclosure)로 Wiki 일부만 로드 |

📌 즉, 여러분이 `CLAUDE.md`를 손으로 쓰고 있었다면 이미 LLM Wiki의 *맛보기*를 한 겁니다. 이 문서는 그것을 **Agent가 자동 생성·검증·유지하는 시스템**으로 끌어올립니다.

### 🧭 Section 2 한 줄 요약

> LLM Wiki는 **"AI가 읽기 위한" 정본·무중복·구조화된 지식 계층**이다. 세 기둥은 **압축(양↓)·증류(모순 해소·정본화)·구조화(관계+검증 가능)**. 멘탈 모델은 *"지식은 손으로 쓰는 게 아니라 빌드되는 산출물"*. `CLAUDE.md`·GraphRAG·DeepWiki·Skills가 그 조각들이다.

---

<a id="section-3"></a>
## Section 3. RAG vs LLM Wiki — 상세 비교

둘은 **경쟁이 아니라 계층**입니다. LLM Wiki는 종종 **검색에 RAG를 그대로 씁니다** — 단, "청크"가 아니라 "정본 Wiki 페이지"를 검색 대상으로 삼고, 그 위에 "관계 순회"를 얹습니다. 차이를 항목별로 못 박습니다.

### 3-1. 핵심 비교표

| 항목 | RAG | LLM Wiki |
|---|---|---|
| **저장 단위** | 원문 청크 (chunk) | **정본 지식 페이지** + 그래프 노드/엣지 |
| **검색 방식** | 벡터 유사도 Top-K | 페이지 하이브리드 검색 **+ Schema 그래프 순회** |
| **Context 사용량** | 중복 청크 다수 → 큼, 턴마다 재인출 | 정제된 1~수 페이지 → 작음, **캐시 친화적** |
| **중복 제거** | 없음 (동일 사실 N벌 그대로) | **Dedup/Merge가 1급 단계** (Section 8) |
| **관계 표현** | 없음 (청크는 독립) | **명시적 typed 관계** (calls/implements/supersedes…) |
| **유지보수** | 문서 추가/삭제 = 재인덱싱 | **Lint·증분 갱신·소유권**으로 운영 (Section 10·12·18) |
| **Agent 활용성** | 조각 제공, 관계는 매번 재추론 | 정본+관계 제공 → **재추론 불필요, 일관** |
| **Knowledge Evolution** | 정본/폐기 개념 없음, 모순 누적 | **status(canonical/deprecated/superseded)로 진화 관리** |
| **운영 비용** | 토큰 폭증, 세션마다 반복 | 압축+캐시로 ↓, 단 **빌드/Lint 비용은 새로 발생** |

⚠️ 공짜가 아닙니다. LLM Wiki는 RAG에 없던 **빌드·검증·거버넌스 비용**을 새로 만듭니다. 작은 1회성 Q&A엔 과합니다. 손익분기는 **"같은 지식을 여러 에이전트가 장기간 반복 사용"** 할 때 넘어갑니다 (Section 18에서 정량화).

### 3-2. 같은 질문, 두 시스템의 답

🧵 관통 질문에 대한 **RAG의 답** (앞서 본 8청크 기반):

> "인보이스는 BackgroundTasks로 발행되며… DB unique 제약으로 중복을 막습니다."
> → ❌ **틀림.** 폐기된 초안(②)과 옛 Notion(④)을 근거로 삼음. 출처 불명확. 3,200토큰 소모.

같은 질문에 대한 **LLM Wiki의 답** (Section 2-3 페이지 + 그래프 순회):

> "구독 갱신 시 `renew_subscription → SubscriptionService.renew → Celery issue_invoice_task → InvoiceService.issue_invoice` 경로로 발행되며, 중복은 **Redis `SETNX idem:invoice:{sub_id}:{period}`** 로 막습니다(ADR-017). DB unique 제약은 경합 500 때문에 폐기됐습니다. period는 UTC 월초 기준."
> → ✅ **정확·출처 포함·관계 포함.** ~450토큰. 다음 에이전트도 같은 답.

### 3-3. 결정 가이드 — 언제 무엇을

```
질문: 내 상황은?

 ├─ 1회성/저빈도 Q&A, 문서가 자주 안 바뀜 ─────────────► RAG로 충분
 │
 ├─ 코딩 에이전트가 같은 코드베이스를 장기 반복 작업 ───► LLM Wiki (RAG를 검색 엔진으로 내장)
 │     (예: acme-billing 유지보수, 모노레포)
 │
 ├─ "관계/의존성/영향범위"를 따라가야 답이 됨 ──────────► LLM Wiki (Schema 순회 필수)
 │
 └─ 정확성·출처·시간에 걸친 일관성이 핵심 ──────────────► LLM Wiki
```

📌 실무 정석: **"검색은 RAG, 지식 관리는 Wiki."** Wiki Layer 검색에 임베딩/벡터DB를 그대로 재사용하되, 인덱싱 대상을 *청크 → 정본 페이지*로 바꾸고 *관계 순회*를 얹는 것 — 이게 가장 흔한 실전 구성입니다.

### 🧭 Section 3 한 줄 요약

> RAG와 Wiki는 경쟁이 아니라 **계층**이다. RAG=청크 유사도 검색, Wiki=정본 페이지+관계 그래프. **중복 제거·관계·진화 관리·캐시 친화성**에서 Wiki가 이기지만, 빌드·Lint·거버넌스 비용을 새로 진다. 손익분기는 "여러 에이전트의 장기 반복 사용".

---
<a id="section-4"></a>
## Section 4. Core Architecture — Raw ↓ Wiki ↓ Schema (가장 중요)

LLM Wiki의 심장은 **3계층**입니다. 이 그림 하나가 문서 전체의 골격입니다.

```
                    ┌───────────────────────────────────────────────┐
   사람·도구가 생성  │                  RAW LAYER                     │  "원천 / 진실의 출처"
   (불변·중복·잡음)  │  코드 · PDF · ADR · GitHub · API Docs · Slack  │  Agent가 읽는 입력
                    │  · Notion                                      │  (Section 5)
                    └───────────────────────┬───────────────────────┘
                                            │  Ingest Pipeline (Section 8)
                                            │  Extract→Normalize→Merge→Dedup
                                            ▼
                    ┌───────────────────────────────────────────────┐
   Agent가 생성·     │                  WIKI LAYER                    │  "AI가 읽기 위한 산문"
   사람이 리뷰       │  Concept Page · Knowledge Page                 │  정본·무중복·[[관계링크]]
   (정본·압축)       │  Canonical 진술 + provenance                   │  (Section 6)
                    └───────────────────────┬───────────────────────┘
                                            │  같은 지식의 두 투영(projection)
                                            │  prose ↔ graph 동기화
                                            ▼
                    ┌───────────────────────────────────────────────┐
   기계가 순회·검증  │                 SCHEMA LAYER                   │  "기계가 읽기 위한 그래프"
   (그래프·제약)     │  Entity · Relationship · Metadata · Constraint │  순회·Lint·영향분석 가능
                    │  (JSON / graph)                                │  (Section 7)
                    └───────────────────────────────────────────────┘
                                            ▲
                          Query(9)·Lint(10)가 이 그래프를 사용
```

> 💡 **비유 — 책 / 백과사전 / 색인 카드.** Raw는 서가의 **원본 책들**(낡고 중복되고 모순됨). Wiki는 사서가 다 읽고 정리한 **백과사전 항목**(사람·AI가 읽음). Schema는 그 항목들을 **색인 카드 + 상호참조 화살표**로 만든 것(기계가 따라감). 같은 지식이지만 **독자가 다르고 형식이 다릅니다.**

### 4-1. 계층별 목적·입력·출력·예시 (관통 예시 기준)

| | **Raw Layer** | **Wiki Layer** | **Schema Layer** |
|---|---|---|---|
| **목적** | 진실의 원천 보존 | AI가 읽을 정본 지식 | 기계 순회·검증 |
| **독자** | (Ingest용) Agent | Agent(주) + 사람(보조) | 기계(Query·Lint) |
| **형식** | 무엇이든 (코드·PDF·로그) | Markdown + frontmatter + `[[링크]]` | JSON / 그래프 |
| **입력** | 사람·CI·툴이 생성 | Raw + Ingest 파이프라인 | Wiki 페이지에서 추출 |
| **출력** | (가공 전) | Concept/Knowledge 페이지 | entities/relationships/constraints |
| **변경 주체** | 개발자·외부 시스템 | Agent 생성 → 사람 승인 | Agent 생성 → Lint 검증 |
| **`acme-billing` 예시** | `invoice.py`, ADR-017, Slack 사고 | `concepts/idempotent-invoice-issuance.md` | `Invoice`–`guarded_by`→`redis-idempotency-key` 엣지 |
| **불변성** | 불변(append-only 취급) | 재생성 가능(빌드 산출물) | Wiki에서 재생성 |

### 4-2. 왜 Wiki와 Schema를 **분리**하나 — 같은 지식의 두 투영

이게 가장 자주 받는 질문입니다. "그냥 Markdown 하나면 되지, 왜 JSON 그래프를 또 만드나?"

답: **역할이 다릅니다.** Wiki(prose)는 *읽기* 최적, Schema(graph)는 *순회·검증* 최적입니다.

- 🧵 "renew가 무엇을 호출하나?"는 **그래프 순회** 질문 → Schema의 엣지를 BFS로 따라가야 빠르고 정확. Markdown 본문을 LLM이 매번 읽어 파싱하면 느리고 불안정.
- "이 페이지가 가리키는 `[[InvoiceService]]`가 실제로 존재하나?"는 **검증** 질문 → Schema 노드 존재 여부로 O(1) 확인 (Broken Link Lint, Section 10).
- 반대로 "왜 DB unique 제약을 폐기했나"의 *서사*는 Schema 엣지로 표현하기 나쁘고, Wiki 산문이 적합.

📌 **핵심:** Wiki 페이지의 frontmatter `related:`와 Schema의 `relationships`는 **같은 관계의 두 표현**입니다. 빌드 시 Wiki에서 Schema를 추출하고, Lint가 둘의 정합성을 검사합니다. 손으로 둘 다 쓰지 않습니다 — Wiki를 정본으로 두고 Schema를 **파생**시키는 게 보통입니다(역방향도 가능).

### 4-3. 디렉터리 레이아웃 — Wiki는 레포에 커밋된다

🧪 **시나리오:** `acme-billing` 팀은 Wiki를 **레포 안에** 둡니다. 코드와 함께 버전관리되고, PR로 리뷰되고, CI에서 Lint됩니다(Section 10). Agent는 이 디렉터리를 읽고 씁니다.

```
acme-billing/
├── app/ ...                         # Raw (코드)
├── docs/adr/ ...                    # Raw (ADR)
└── wiki/                            # ← LLM Wiki (레포에 커밋, PR 리뷰, CI Lint)
    ├── concepts/                    # Concept Page (개념·패턴)
    │   ├── idempotent-invoice-issuance.md
    │   └── dependency-injection.md
    ├── knowledge/                   # Knowledge Page (구체 엔티티)
    │   ├── InvoiceService.md
    │   ├── renew_subscription.md
    │   └── Invoice.md
    ├── schema/                      # Schema Layer (파생·검증 대상)
    │   ├── entities.json
    │   ├── relationships.json
    │   └── constraints.json
    └── .wikilintrc.yaml             # Lint 규칙 (Section 10)
```

> 🔧 **실무 팁:** Raw가 외부(Notion·Slack·Confluence)에 있으면 `wiki/`만 레포에 두고, Ingest 파이프라인이 외부 Raw를 끌어옵니다(Section 15). 핵심은 **Wiki/Schema는 버전관리 + 코드리뷰 + CI 대상**이라는 것 — "운영되는 시스템"이지 일회성 산출물이 아닙니다.

### 🧭 Section 4 한 줄 요약

> 3계층: **Raw**(진실의 원천, 불변·잡음) → **Wiki**(AI용 정본 산문, 재생성됨) → **Schema**(기계용 그래프, 순회·검증). Wiki와 Schema는 *같은 지식의 두 투영*(읽기용/순회용)이며 Wiki를 정본으로 Schema를 파생한다. `wiki/`는 레포에 커밋되어 PR·CI 대상이 된다.

---

<a id="section-5"></a>
## Section 5. Raw Layer — Agent가 무엇을 읽는가

Raw Layer는 **진실의 원천(source of truth)** 입니다. 가공되지 않은, 중복·모순·잡음투성이의 원본. LLM Wiki는 이것을 *바꾸지 않고*, 읽어서 위층을 빌드합니다.

### 5-1. Raw 소스 유형과 `acme-billing` 매핑

| Raw 유형 | 무엇을 담나 | `acme-billing` 예 | 로더/추출기 |
|---|---|---|---|
| **GitHub Repository (코드)** | 구조·구현의 1차 진실 | `app/**`, `migrations/**` | tree-sitter / `ast` / git |
| **Markdown** | 핸드북·README | `README.md`, `CONTRIBUTING.md` | 파일 직접 |
| **ADR** | *왜* 그렇게 결정했나 | `docs/adr/ADR-017`, `ADR-021` | 파일 직접 |
| **API Docs** | 외부 계약 | OpenAPI `openapi.json` (FastAPI 자동생성) | OpenAPI 파서 |
| **Slack Logs** | 사고·암묵지·맥락 | `#billing-incidents` 2025-11 사고 | Slack API |
| **Notion** | 온보딩·정책 | "결제팀 온보딩" | Notion API |
| **PDF** | 계약·규정·외부 명세 | 결제대행사(PSP) 연동 명세 | `pypdf`/멀티모달 |

📌 **각 Raw 유형은 "다른 종류의 진실"을 담습니다.** 코드는 *무엇을/어떻게*, ADR은 *왜*, Slack은 *맥락/사고*, API Docs는 *계약*. 좋은 Wiki 페이지는 이들을 **교차 인용**합니다 — Section 2-3 페이지가 코드+ADR+Slack을 한 페이지에 묶었듯이.

### 5-2. Raw의 두 가지 규칙: 불변성 + 출처추적

**① 불변(immutable)으로 취급한다.** Wiki는 Raw를 수정하지 않습니다. 코드를 고치는 건 개발자(또는 코딩 에이전트의 별도 작업)이지, Wiki 빌드가 아닙니다. Wiki는 **읽기 전용 소비자**입니다.

**② 모든 Wiki 진술은 Raw로 역추적(provenance)된다.** Section 2-3 페이지의 `sources:`가 그것입니다. 출처 없는 Wiki 진술은 **환각 의심 대상**이고 Lint가 잡아냅니다(Section 10, 19). 이게 RAG의 "출처 표기"를 계층 차원으로 끌어올린 것입니다.

> 💻 **Raw 로딩 — 코드에서 "읽을 단위" 뽑기** (`ast`로 함수/클래스 경계 추출)

```python
import ast, pathlib

def iter_code_units(root: str):
    """파일을 통째로 청크내지 않고, 함수/클래스 단위로 의미 경계를 뽑는다."""
    for path in pathlib.Path(root).rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                yield {
                    "raw_id": f"{path}#{node.name}",          # provenance 키
                    "kind": type(node).__name__,
                    "name": node.name,
                    "file": str(path),
                    "lineno": node.lineno,
                    "src": ast.get_source_segment(path.read_text(), node),
                    "doc": ast.get_docstring(node),
                }

# 예: renew_subscription, SubscriptionService.renew, InvoiceService.issue_invoice …
```

> 🔧 **실무 팁:** 코드 Raw는 "임의 N자 청킹"보다 **AST 경계 청킹**이 압도적으로 낫습니다 — `issue_invoice` 함수가 두 청크로 잘리는 일이 없고, `raw_id`가 안정적이라 provenance·증분 갱신(Section 12)에 그대로 쓰입니다.

### 🧭 Section 5 한 줄 요약

> Raw Layer는 **진실의 원천**(코드·ADR·API Docs·Slack·Notion·PDF)이며 유형마다 다른 진실(무엇/왜/맥락/계약)을 담는다. 두 규칙: **불변으로 취급**, **모든 Wiki 진술은 Raw로 역추적**. 코드 Raw는 임의 청킹이 아니라 AST 경계로 읽는다.

---

<a id="section-6"></a>
## Section 6. Wiki Layer — 정본 지식의 형식 (가장 중요)

Wiki Layer는 **AI가 읽기 위한 정본 산문**입니다. 두 종류의 페이지로 이루어집니다.

### 6-1. 두 페이지 타입 — Concept vs Knowledge

| | **Concept Page** | **Knowledge Page** |
|---|---|---|
| 담는 것 | **개념·패턴·정책** (추상) | **구체 엔티티** (코드의 실체) |
| 1:1 대응 | 코드에 직접 대응 안 됨 | 보통 하나의 클래스/함수/모델/엔드포인트 |
| `acme-billing` 예 | `idempotent-invoice-issuance`, `dependency-injection` | `InvoiceService`, `renew_subscription`, `Invoice` |
| 수명 | 길다 (설계가 바뀌어야 변함) | 코드와 함께 변함 (증분 갱신 대상) |

> 💡 **비유:** Knowledge Page는 위키피디아의 *"에펠탑"* 항목(구체적 실체), Concept Page는 *"고딕 건축"* 항목(여러 실체를 관통하는 개념). 둘은 `[[링크]]`로 연결됩니다 — "에펠탑은 [[고딕 건축]]의 예가 아니다" 같은 관계까지.

### 6-2. Knowledge Page 해부 — `InvoiceService`

> 💻 `wiki/knowledge/InvoiceService.md`

```markdown
---
id: knowledge.InvoiceService
type: knowledge
entity_kind: service_class
title: InvoiceService
status: canonical
owners: [team-billing]
last_verified: 2026-06-10
sources:
  - code: app/services/invoice.py#InvoiceService
related:
  - defined_in: "[[app/services/invoice.py]]"
  - implements: "[[idempotent-invoice-issuance]]"      # → Concept Page
  - called_by: "[[issue_invoice_task]]"
  - writes: "[[Invoice]]"
  - depends_on: "[[redis-idempotency-key]]"
  - uses_pattern: "[[dependency-injection]]"
---

## 무엇인가
구독에 대한 인보이스 생성을 책임지는 서비스 클래스. 멱등 발행의 **구현체**.

## 공개 API
- `issue_invoice(subscription_id: int, period: date) -> Invoice | None`
  - 반환 `None` = 이미 발행됨(멱등 no-op). 예외 아님.
- `void_invoice(invoice_id: int) -> None`

## 핵심 불변식 (Invariants)
- 같은 `(subscription_id, period)`로 몇 번 호출돼도 `Invoice`는 1건. (→ [[idempotent-invoice-issuance]])
- `period`는 UTC 월초.

## 협력자 (Collaborators)
- `redis` (멱등 키) · `db: AsyncSession` (FastAPI [[dependency-injection]]로 주입)
```

📌 Knowledge Page의 본질: **"코드를 읽지 않고도 이 엔티티를 안전하게 쓸 수 있을 만큼"** 의 계약·불변식·협력자·관계를 담습니다. 구현 세부(루프·변수)는 넣지 않습니다 — 그건 코드(Raw)가 진실이니까. Wiki는 *코드에서 자명하지 않은* 것(불변식, "None은 멱등 no-op이다", 관계)을 담습니다.

🧪 **시나리오 (불변식이 버그를 막는다):** 개발자 J가 Claude Code로 결제 재시도 핸들러를 짭니다. `issue_invoice`가 `None`을 반환하자, Wiki가 없었다면 에이전트는 코드만 보고 "발행 실패"로 오인해 `raise HTTPException(500)` 같은 잘못된 처리를 넣었을 겁니다(중복 호출마다 500 → 알림 폭주). 하지만 `CLAUDE.md`(Section 16) 규칙대로 `InvoiceService` Knowledge Page를 먼저 읽어 **"None = 이미 발행됨(멱등 no-op), 예외 아님"** 불변식을 알게 됨 → `None`이면 기존 인보이스를 조회해 정상 응답하는 올바른 코드를 작성. **코드에서 자명하지 않은 한 줄의 불변식**이 운영 사고를 막은 것 — 이게 Knowledge Page가 존재하는 이유입니다.

### 6-3. Concept Page 해부 — FastAPI Dependency Injection

스펙이 요구한 FastAPI 예시(`BackgroundTasks`, `Dependency Injection`, `APIRouter`)를 Concept Page로 표현해 봅니다.

> 💻 `wiki/concepts/dependency-injection.md`

```markdown
---
id: concept.dependency-injection
type: concept
title: 의존성 주입 (FastAPI Depends) in acme-billing
status: canonical
owners: [team-billing, team-platform]
last_verified: 2026-06-09
sources:
  - code: app/api/deps.py
  - code: app/api/subscriptions.py
related:
  - provides: "[[get_db]]"          # AsyncSession 주입자
  - provides: "[[get_redis]]"
  - consumed_by: "[[renew_subscription]]"
  - consumed_by: "[[InvoiceService]]"
---

## 정본 진술
`acme-billing`의 모든 엔드포인트는 DB 세션·Redis·현재 사용자(인증)를
**`Depends()`** 로 주입받는다. 직접 전역 객체를 import 하지 않는다.
이유: 테스트에서 `app.dependency_overrides`로 가짜 세션을 끼우기 위함.

## 표준 패턴
- `get_db()` → `AsyncSession` (요청 단위, 끝나면 자동 close/rollback)
- `get_redis()` → `redis.asyncio.Redis` (커넥션 풀 공유)
- `get_current_customer()` → 인증된 `Customer`

## 관계
- `[[renew_subscription]]` 같은 [[APIRouter]] 핸들러가 이 패턴의 소비자.
- 백그라운드 작업([[issue_invoice_task]])은 요청 스코프 밖이라 **DI를 못 쓴다**
  → 세션을 직접 생성. (흔한 함정: 태스크에서 요청 세션을 재사용하려다 깨짐.)
```

여기서 **관계가 어떻게 지식을 묶는지** 보세요. `renew_subscription`(Knowledge) → `uses_pattern` → `dependency-injection`(Concept) → `provides` → `get_db`(Knowledge). Agent가 "renew 핸들러에 DB 세션 어떻게 들어와?"를 물으면 이 관계 한 홉으로 답이 나옵니다.

### 6-4. Relationship — 관계는 1급 시민

RAG에 없던, Wiki의 핵심 자산. 관계는 **타입을 가진 엣지**입니다. `acme-billing`에서 쓰는 표준 관계 타입:

| 관계 타입 | 의미 | 예 (`acme-billing`) |
|---|---|---|
| `calls` / `called_by` | 호출 | `renew_subscription` → `SubscriptionService.renew` |
| `implements` / `implemented_by` | 개념의 구현 | `InvoiceService` → `idempotent-invoice-issuance` |
| `triggers` / `triggered_by` | 흐름 시작 | `renew_subscription` → `issue_invoice_task` |
| `writes` / `reads` | 데이터 접근 | `InvoiceService` → `Invoice` |
| `depends_on` | 인프라/외부 의존 | `InvoiceService` → `redis-idempotency-key` |
| `guarded_by` | 불변식 보장 수단 | `Invoice` → `redis-idempotency-key` |
| `supersedes` / `superseded_by` | 결정의 진화 | `ADR-021` → `BackgroundTasks 방식` |
| `documented_in` | 출처 결정 | `idempotent-invoice-issuance` → `ADR-017` |

⚠️ **함정 (Wrong Relationship):** 관계 타입을 아무거나 만들면 그래프가 의미를 잃습니다. **타입 vocabulary를 Schema에 고정**하고(Section 7), Lint가 미등록 타입을 거부하게 하세요(Section 10). 자유 텍스트 관계는 Schema Drift의 시작입니다(Section 19).

### 6-5. Canonical Knowledge — 정본과 진화

각 페이지·진술은 `status`를 가집니다. 이게 RAG에 없던 **지식 진화 관리**입니다.

```
status 라이프사이클:
  draft ──(사람 승인)──► canonical ──(사실 변경)──► deprecated
                              │
                              └──(대체 결정 등장)──► superseded_by: [[새 페이지]]
```

🧵 관통 예시의 진화: 처음엔 `idempotent-invoice-issuance`가 "DB unique 제약"으로 canonical이었습니다. ADR-017로 "Redis SETNX"가 채택되며 옛 진술은 `superseded`로, 새 진술이 canonical로 바뀌었습니다. **둘 다 그래프에 남되 status로 구분**됩니다 — 그래서 Agent는 "예전엔 왜 DB 제약이었나"도 답할 수 있고(서사 보존), "지금 뭐가 맞나"도 틀리지 않습니다(정본 우선).

> 🔧 **실무 팁 (Knowledge Compression의 실제 효과):** Section 2-3 정본 페이지(~400토큰)를 **프롬프트 캐싱**으로 고정하면, 같은 도메인을 만지는 모든 에이전트 턴이 그 페이지를 **캐시 read(≈0.1배 비용)** 로 재사용합니다. 정본 페이지는 자주 안 바뀌므로 캐시 적중률이 높습니다 — Wiki가 비용까지 잡는 지점(Section 9·16에서 코드로).

### 🧭 Section 6 한 줄 요약

> Wiki Layer = **Concept Page**(개념·패턴) + **Knowledge Page**(구체 엔티티), 둘 다 정본 진술·불변식·`[[관계]]`·출처를 담는다. **관계는 타입을 가진 1급 엣지**(calls/implements/guarded_by/supersedes…)이고 vocabulary는 Schema에 고정한다. `status`로 정본↔폐기↔대체를 관리해 **지식의 진화**를 잃지 않는다.

---

<a id="section-7"></a>
## Section 7. Schema Layer — 기계가 읽는 그래프

Schema Layer는 Wiki를 **기계가 순회·검증할 수 있는 형태**로 만든 것입니다. 네 가지 요소: **Entity · Relationship · Metadata · Constraints**.

### 7-1. Entity — 타입을 가진 노드

> 💻 `wiki/schema/entities.json` (발췌)

```json
{
  "entities": [
    {
      "id": "knowledge.InvoiceService",
      "kind": "service_class",
      "title": "InvoiceService",
      "status": "canonical",
      "wiki_page": "wiki/knowledge/InvoiceService.md",
      "source": { "code": "app/services/invoice.py#InvoiceService" },
      "metadata": { "owners": ["team-billing"], "last_verified": "2026-06-10" }
    },
    {
      "id": "knowledge.renew_subscription",
      "kind": "http_endpoint",
      "title": "POST /subscriptions/{id}/renew",
      "status": "canonical",
      "wiki_page": "wiki/knowledge/renew_subscription.md",
      "source": { "code": "app/api/subscriptions.py#renew_subscription" },
      "metadata": { "method": "POST", "auth": "customer" }
    },
    {
      "id": "concept.idempotent-invoice-issuance",
      "kind": "concept",
      "title": "멱등 인보이스 발행",
      "status": "canonical",
      "wiki_page": "wiki/concepts/idempotent-invoice-issuance.md",
      "source": { "adr": "docs/adr/ADR-017-idempotent-invoice-issuance.md" }
    },
    {
      "id": "knowledge.redis-idempotency-key",
      "kind": "infra_resource",
      "title": "Redis 멱등 키",
      "status": "canonical",
      "metadata": { "key_pattern": "idem:invoice:{subscription_id}:{period}", "ttl_sec": 86400 }
    }
  ]
}
```

`kind`는 **고정 vocabulary**입니다: `service_class`, `http_endpoint`, `db_model`, `background_task`, `infra_resource`, `concept`, `decision(ADR)`, `external_contract` 등. 임의 kind는 Lint가 거부합니다.

### 7-2. Relationship — 타입을 가진 엣지

> 💻 `wiki/schema/relationships.json` (발췌) — 관통 질문의 정답 경로가 그래프로

```json
{
  "relationships": [
    { "from": "knowledge.renew_subscription", "type": "triggers",
      "to": "knowledge.issue_invoice_task" },
    { "from": "knowledge.issue_invoice_task", "type": "calls",
      "to": "knowledge.InvoiceService", "note": "issue_invoice()" },
    { "from": "knowledge.InvoiceService", "type": "implements",
      "to": "concept.idempotent-invoice-issuance" },
    { "from": "knowledge.InvoiceService", "type": "writes",
      "to": "knowledge.Invoice" },
    { "from": "knowledge.Invoice", "type": "guarded_by",
      "to": "knowledge.redis-idempotency-key" },
    { "from": "decision.ADR-021", "type": "supersedes",
      "to": "knowledge.fastapi-backgroundtasks-invoicing",
      "note": "BackgroundTasks → Celery 이전" }
  ]
}
```

이 엣지들을 `from`/`to`로 이으면 정확히 Section 1-3의 정답 경로가 됩니다. Query 엔진은 이걸 **BFS로 순회**합니다(Section 9).

### 7-3. Metadata — 운영을 가능케 하는 부가정보

엔티티/관계에 붙는 운영 메타데이터. 이게 있어야 **거버넌스·freshness·접근제어**가 됩니다.

| 메타데이터 | 용도 | 어디서 쓰나 |
|---|---|---|
| `owners` | 변경 승인자 | Approval Workflow (Section 18) |
| `last_verified` | 신선도 | Stale Lint (Section 10) |
| `source` | 출처 | Provenance·Broken-source Lint |
| `status` | 정본/폐기 | Query 우선순위·Dedup |
| `visibility` | 접근 권한 | 권한별 검색 필터 (Section 18) |
| `confidence` | Agent 생성 신뢰도 | 사람 리뷰 우선순위 (Section 11) |

### 7-4. Constraints — 그래프가 지켜야 할 규칙

Schema의 백미. **그래프 자체의 무결성 규칙**을 선언하고, Lint가 강제합니다.

> 💻 `wiki/schema/constraints.json`

```json
{
  "entity_kinds": ["service_class","http_endpoint","db_model","background_task",
                   "infra_resource","concept","decision","external_contract"],
  "relationship_types": ["calls","called_by","triggers","triggered_by","implements",
                         "implemented_by","writes","reads","depends_on","guarded_by",
                         "supersedes","superseded_by","documented_in"],
  "rules": [
    { "id": "no-dangling-edge",
      "desc": "관계의 from/to는 반드시 존재하는 entity여야 한다" },
    { "id": "endpoint-must-have-owner",
      "applies_to_kind": "http_endpoint",
      "require_metadata": ["owners"] },
    { "id": "canonical-needs-source",
      "when": { "status": "canonical" },
      "require_metadata": ["source"],
      "desc": "정본 진술은 출처가 없으면 환각 의심" },
    { "id": "single-canonical-per-concept",
      "desc": "한 개념에 canonical 진술은 하나만 (나머지는 superseded/deprecated)" },
    { "id": "no-unknown-relationship-type",
      "desc": "relationship_types에 없는 타입 금지 (Schema Drift 방지)" }
  ]
}
```

> 💻 **Constraints를 코드로 강제** — `jsonschema` + 커스텀 규칙 (Linter의 일부, Section 10·17에서 확장)

```python
import json, jsonschema

ENTITY_SCHEMA = {
    "type": "object",
    "required": ["id", "kind", "title", "status"],
    "properties": {
        "id": {"type": "string"},
        "kind": {"enum": json.load(open("wiki/schema/constraints.json"))["entity_kinds"]},
        "status": {"enum": ["draft", "canonical", "deprecated", "superseded"]},
    },
    "additionalProperties": True,
}

def validate_entities(entities: list[dict]) -> list[str]:
    errors = []
    for e in entities:
        try:
            jsonschema.validate(e, ENTITY_SCHEMA)
        except jsonschema.ValidationError as ex:
            errors.append(f"{e.get('id','?')}: {ex.message}")
    return errors
```

📌 **핵심:** Constraints가 LLM Wiki를 "그냥 마크다운 폴더"와 가르는 결정적 차이입니다. 그래프에 **타입 시스템과 무결성 규칙**이 있어서, Agent가 생성한 지식을 **기계가 검증**할 수 있습니다. 이게 없으면 Wiki는 곧 Schema Drift로 썩습니다(Section 19).

### 🧭 Section 7 한 줄 요약

> Schema Layer = **Entity(타입 노드)·Relationship(타입 엣지)·Metadata(운영정보)·Constraints(무결성 규칙)** 의 JSON 그래프. kind와 relationship_type은 **고정 vocabulary**, constraints는 `jsonschema`+커스텀 규칙으로 **기계 강제**. 이 타입 시스템이 순회(Query)·검증(Lint)·거버넌스를 가능케 하며, "마크다운 폴더"와 "운영되는 지식 그래프"를 가른다.

---
<a id="section-8"></a>
## Section 8. Ingest Pipeline — Raw에서 Wiki를 빌드하기

Ingest는 **"지식 컴파일러"** 입니다. Raw를 입력받아 Wiki+Schema를 빌드합니다.

```
   RAW                                                              WIKI + SCHEMA
   코드/ADR/Slack ─► Extract ─► Normalize ─► Merge ─► Deduplicate ─► Wiki Update ─► Schema Update
                    (LLM이      (정본 ID·    (기존     (중복 사실     (페이지         (그래프
                     엔티티·    vocabulary   페이지에   탐지·삭제)     write)          재생성)
                     관계 추출)  매핑)        합치기)
```

각 단계를 관통 예시로 따라갑니다.

### 8-1. Extract — Raw 단위에서 엔티티·관계 뽑기 (LLM + 구조화 출력)

Section 5의 `iter_code_units`가 뽑은 `issue_invoice` 함수 소스를 LLM에 주고, **구조화 출력**으로 엔티티/관계를 받습니다. 자유 텍스트가 아니라 스키마 강제가 핵심입니다.

> 💻 구조화 출력으로 Extract (`output_config.format`)

```python
import anthropic, json

client = anthropic.Anthropic()

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "kind": {"enum": ["service_class","http_endpoint","db_model",
                                  "background_task","infra_resource","concept"]},
                "title": {"type": "string"},
                "canonical_statement": {"type": "string"},
                "invariants": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["id","kind","title","canonical_statement"],
            "additionalProperties": False,
        }},
        "relationships": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "from": {"type": "string"},
                "type": {"enum": ["calls","triggers","implements","writes","reads",
                                  "depends_on","guarded_by"]},
                "to": {"type": "string"},
            },
            "required": ["from","type","to"],
            "additionalProperties": False,
        }},
    },
    "required": ["entities","relationships"],
    "additionalProperties": False,
}

def extract(unit: dict) -> dict:
    resp = client.messages.create(
        model="claude-opus-4-8",        # 추출 정확도가 Wiki 품질을 좌우 → Opus.
        max_tokens=4000,                # 대규모 레포면 비용상 sonnet/haiku 검토(Section 14)
        system="너는 코드에서 정본 지식과 타입 관계를 뽑는 추출기다. "
               "코드에 명시된 사실만 뽑고, 추측 금지. id는 'kind.Name' 형식.",
        messages=[{"role": "user", "content":
                   f"file: {unit['file']}\nname: {unit['name']}\n\n{unit['src']}"}],
        output_config={"format": {"type": "json_schema", "schema": EXTRACT_SCHEMA}},
    )
    return json.loads(next(b.text for b in resp.content if b.type == "text"))

# extract(issue_invoice_unit) →
# entities: [{id:"service_class.InvoiceService", canonical_statement:"기간당 1건만…", invariants:[...]}]
# relationships: [{from:"...InvoiceService", type:"guarded_by", to:"infra_resource.redis-idempotency-key"}]
```

⚠️ **함정 (Agent Hallucination):** 추출 LLM이 코드에 없는 관계를 "그럴듯하게" 지어낼 수 있습니다(Section 19). 방어: (1) **"명시된 사실만"** 시스템 프롬프트, (2) 모든 추출에 `source`(raw_id) 강제, (3) Lint의 `canonical-needs-source` 규칙. ADR·Slack 같은 *서사* Raw는 추출 신뢰도가 낮으니 `confidence`를 낮게 달고 사람 리뷰로 보냅니다.

### 8-2. Normalize — 정본 ID와 vocabulary로 맞추기

추출기는 같은 것을 `InvoiceService` / `invoice_service` / `InvoiceSvc`로 제각각 부를 수 있습니다. Normalize가 **정본 ID**(`knowledge.InvoiceService`)로 통일하고, `kind`/`relationship type`을 Schema vocabulary(Section 7-4)에 매핑합니다. vocabulary에 없는 타입은 가장 가까운 표준으로 매핑하거나 리뷰 큐로 보냅니다.

### 8-3. Merge — 기존 페이지에 새 사실 합치기

`acme-billing`은 이미 `InvoiceService` Knowledge Page가 있습니다. 새 커밋에서 `void_invoice` 메서드가 추가됐다면, **페이지를 새로 만들지 않고** 기존 페이지의 "공개 API" 섹션에 합칩니다. 핵심 판단: *"이건 새 엔티티인가, 기존 엔티티의 새 사실인가?"* — `id` 일치로 1차 판별, 모호하면 임베딩 유사도(다음 단계)로.

### 8-4. Deduplicate — 같은 사실의 N벌을 하나로

Section 1-2에서 본 중복(코드+ADR+Slack+Notion이 같은 "멱등" 사실)을 여기서 제거합니다. **임베딩 유사도**로 "이 진술은 이미 정본에 있다"를 탐지합니다.

> 💻 정본 진술과의 중복 탐지 (코사인 유사도)

```python
import numpy as np

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def is_duplicate(new_stmt_emb, canonical_embs: list, threshold=0.92) -> bool:
    """새 진술이 기존 정본 진술과 사실상 동일하면 True → 흡수(merge), 새 페이지 X."""
    return any(cosine(new_stmt_emb, ce) >= threshold for ce in canonical_embs)

# Notion "인보이스는 BackgroundTasks로…" 진술:
#   - idempotent-invoice-issuance 정본과 sim 0.88 (관련은 있으나 동일 아님)
#   - 단, 사실이 '모순' → Dedup이 아니라 '갱신 후보'로 분기.
# 따라서 Dedup은 (a) 동일→흡수, (b) 모순→리뷰/갱신, (c) 신규→추가 로 3분기.
```

⚠️ **임계값 함정:** threshold가 낮으면 다른 사실을 합쳐 정보 손실, 높으면 중복이 남아 Wiki 폭증(Section 19). 0.9~0.95에서 시작해 도메인별 튜닝하고, **모순 케이스(b)는 절대 자동 흡수하지 말고** 사람에게 보냅니다 — "옛 사실 vs 새 사실"의 판단은 위험합니다.

### 8-5. Wiki Update + Schema Update — 쓰고, 그래프 재생성

정제된 결과로 Markdown 페이지를 write하고(Wiki), 거기서 `entities.json`/`relationships.json`을 **재생성**합니다(Schema). Wiki가 정본이고 Schema는 파생이므로, 페이지의 frontmatter `related:`를 파싱해 relationships를 다시 만듭니다. 마지막에 **Lint를 돌려**(Section 10) 무결성을 확인한 뒤에야 커밋합니다.

🧪 **시나리오 (전체 Ingest 1회):** 팀이 `make wiki-ingest`를 돌리면 → 변경된 코드 유닛만 Extract → Normalize → 기존 페이지와 Merge/Dedup → `wiki/` 갱신 → Schema 재생성 → `wiki-lint` 통과 시 PR 생성. 사람은 PR diff(추가/변경된 정본 진술)만 리뷰합니다. **지식 빌드가 코드 빌드처럼** 돌아갑니다.

### 🧭 Section 8 한 줄 요약

> Ingest = 지식 컴파일러: **Extract**(LLM 구조화 추출, 출처 강제) → **Normalize**(정본 ID·vocabulary) → **Merge**(기존 페이지에 합침) → **Deduplicate**(임베딩 유사도; 동일은 흡수, **모순은 사람에게**) → **Wiki/Schema Update**(쓰고 그래프 재생성 후 Lint). 환각·중복·모순을 막는 게 각 단계의 존재 이유.

---

<a id="section-9"></a>
## Section 9. Query Pipeline — Wiki로 정확하게 답하기

Query는 Ingest의 역방향입니다. 질문을 받아 **정본 페이지 검색 + 그래프 순회**로 정밀한 컨텍스트를 조립합니다.

```
   질문 ─► Intent Analysis ─► Wiki Search ─► Schema Traversal ─► Context Construction ─► Answer
          (lookup? 관계?       (정본 페이지   (그래프 BFS로        (정본+이웃 합치고        (출처 포함,
           how-to?)            하이브리드     관계 이웃 수집)      중복 제거·캐시)          관계 포함)
                               검색)
```

관통 질문 🧵 "구독 갱신 시 인보이스 중복 발행을 어떻게 막나?"로 끝까지 따라갑니다.

### 9-1. Intent Analysis — 질문 유형 분류

질문이 *단순 조회(lookup)* 인지, *관계 추적(traversal)* 인지, *방법(how-to)* 인지에 따라 검색 전략이 다릅니다. 관통 질문은 "막는 **경로/수단**"을 묻는 → **traversal**.

> 💻 인텐트 분류 (구조화 출력)

```python
def analyze_intent(question: str) -> dict:
    resp = client.messages.create(
        model="claude-haiku-4-5",       # 분류는 가볍고 빈번 → Haiku로 비용↓
        max_tokens=300,
        messages=[{"role": "user", "content": question}],
        system="질문을 분류하라.",
        output_config={"format": {"type": "json_schema", "schema": {
            "type": "object",
            "properties": {
                "intent": {"enum": ["lookup", "traversal", "how_to", "impact"]},
                "seed_entities": {"type": "array", "items": {"type": "string"}},
                "relation_focus": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["intent", "seed_entities"],
            "additionalProperties": False,
        }}},
    )
    return json.loads(next(b.text for b in resp.content if b.type == "text"))

# analyze_intent("구독 갱신 시 인보이스 중복 발행을 어떻게 막나?") →
# {"intent":"traversal","seed_entities":["renew_subscription","invoice"],
#  "relation_focus":["triggers","calls","guarded_by"]}
```

### 9-2. Wiki Search — 정본 페이지를 하이브리드 검색

여기서 RAG 기술을 **그대로 재사용**합니다. 단, 인덱싱 대상이 *청크*가 아니라 *정본 페이지*이고, **`status=canonical` 우선·`deprecated` 제외** 필터가 붙습니다. seed로 `idempotent-invoice-issuance` 정본 페이지가 잡힙니다.

```python
def wiki_search(seed_entities, k=3):
    # 벡터 검색(임베딩) + 키워드(BM25) 하이브리드. canonical만.
    hits = vector_store.search(seed_entities, k=k,
                               filter={"status": "canonical"})
    return [h.entity_id for h in hits]   # 예: ["concept.idempotent-invoice-issuance"]
```

### 9-3. Schema Traversal — RAG가 못 하는 그 단계

검색으로 찾은 seed 노드에서 **관계 그래프를 BFS**로 따라가 이웃을 모읍니다. 이게 Fragmented Retrieval(Section 1-3)을 푸는 핵심입니다.

> 💻 그래프 순회 (`networkx`)

```python
import json, networkx as nx

def build_graph():
    G = nx.DiGraph()
    rels = json.load(open("wiki/schema/relationships.json"))["relationships"]
    for r in rels:
        G.add_edge(r["from"], r["to"], type=r["type"], note=r.get("note"))
    return G

def traverse(G, seeds, focus_types, max_hops=2):
    """seed에서 focus 관계를 따라 max_hops까지 이웃 수집."""
    collected, frontier = set(seeds), list(seeds)
    for _ in range(max_hops):
        nxt = []
        for node in frontier:
            for _, to, data in G.out_edges(node, data=True):
                if not focus_types or data["type"] in focus_types:
                    if to not in collected:
                        collected.add(to); nxt.append(to)
        frontier = nxt
    return collected

G = build_graph()
nodes = traverse(G, ["knowledge.renew_subscription"],
                 focus_types=["triggers","calls","implements","writes","guarded_by"])
# → {renew_subscription, issue_invoice_task, InvoiceService,
#    idempotent-invoice-issuance, Invoice, redis-idempotency-key}
#   = 정확히 Section 1-3의 정답 경로!
```

📌 **여기가 분기점입니다.** RAG는 "renew와 비슷한 청크"만 줬습니다. Wiki는 "renew가 **triggers** 하는 것 → 그게 **calls** 하는 것 → 그게 **guarded_by** 하는 것"을 **결정론적으로** 따라갑니다. 유사도가 아니라 관계입니다.

### 9-4. Context Construction — 정본+이웃을 압축 조립 (+ 캐시)

순회로 모은 노드들의 **정본 페이지만** 이어 붙입니다(중복·deprecated 없음). 그리고 자주 안 바뀌는 정본 묶음을 **프롬프트 캐싱**으로 고정해 비용을 죽입니다.

> 💻 컨텍스트 조립 + 프롬프트 캐싱 (Wiki가 비용을 잡는 지점)

```python
def build_context(node_ids) -> str:
    pages = [load_canonical_page(nid) for nid in node_ids]   # status=canonical만
    return "\n\n---\n\n".join(pages)                          # 보통 수백~1.5k 토큰

def answer(question, node_ids):
    wiki_context = build_context(node_ids)
    resp = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1500,
        system=[{
            "type": "text",
            "text": "다음 Wiki 지식만 근거로 답하라. 없으면 모른다고 하라. "
                    "각 주장에 [출처: ...]를 달아라.\n\n" + wiki_context,
            "cache_control": {"type": "ephemeral"},   # ← 정본 묶음을 캐시
        }],
        messages=[{"role": "user", "content": question}],
    )
    u = resp.usage
    print(f"cache_read={u.cache_read_input_tokens} input={u.input_tokens}")
    # 같은 도메인 후속 질문은 cache_read로 ≈0.1배 비용 → 에이전트 장기세션에서 결정적
    return next(b.text for b in resp.content if b.type == "text")
```

> 🔧 **캐시 적중의 조건(중요):** 정본 페이지가 **byte 단위로 안정**해야 캐시가 맞습니다. Wiki 페이지에 타임스탬프·UUID를 본문에 박지 마세요(frontmatter에만). `last_verified` 같은 변동 필드는 캐시 프리픽스 *뒤*로 보내거나 본문에서 빼야 합니다 — 안 그러면 매 턴 캐시 미스(silent invalidator).

### 9-5. Answer — 출처와 관계를 포함해

최종 답(Section 3-2의 ✅ 답)은 **정본 진술 + 순회 경로 + 출처**를 담습니다. RAG의 3,200토큰·모순 답 대비, ~450토큰·정확·일관.

🧪 **시나리오:** 신입 D의 에이전트가 같은 질문을 매 세션 던져도, 매번 **같은 정본 답**을 ~450토큰·캐시 read로 받습니다. Section 1-1의 "세션마다 80k 재발견"이 사라집니다.

### 🧭 Section 9 한 줄 요약

> Query = **Intent**(lookup/traversal/how-to/impact 분류) → **Wiki Search**(정본 페이지 하이브리드 검색, canonical 우선) → **Schema Traversal**(그래프 BFS로 관계 이웃 결정론적 수집 ← RAG가 못 하는 단계) → **Context Construction**(정본만 압축 + 프롬프트 캐싱) → **Answer**(출처·관계 포함). 정확·일관·저비용.

---

<a id="section-10"></a>
## Section 10. Lint Pipeline — Wiki를 썩지 않게

Wiki는 **운영되는 시스템**이지 일회성 산출물이 아닙니다. 코드에 린터/CI가 있듯, Wiki에도 있어야 합니다. **Lint가 없는 Wiki는 6개월 안에 신뢰를 잃습니다**(Section 19의 모든 실패가 여기서 예방됩니다).

```
   wiki/ + schema/ ─► [Broken Link] [Missing Entity] [Duplicate] [Stale] [Schema Violation]
                                        │
                                        ▼
                         통과 → 머지 허용 / 실패 → PR 차단·리뷰 큐
```

### 10-1. 다섯 가지 검사 — 왜 필요한가 + 코드

**① Broken Link — `[[...]]`가 존재하지 않는 엔티티를 가리킴**
왜: 코드가 리팩터돼 `InvoiceService`가 `BillingService`로 바뀌면, 그를 가리키던 링크가 깨집니다. Agent가 죽은 링크를 따라가면 헛돕니다.

```python
import re, json, pathlib

def check_broken_links(entities: dict) -> list[str]:
    ids = {e["id"] for e in entities}
    titles = {e["title"]: e["id"] for e in entities}
    errors = []
    for page in pathlib.Path("wiki").rglob("*.md"):
        for m in re.finditer(r"\[\[([^\]]+)\]\]", page.read_text(encoding="utf-8")):
            ref = m.group(1)
            if ref not in ids and ref not in titles:
                errors.append(f"{page}: broken link [[{ref}]]")
    return errors
```

**② Missing Entity — 관계가 가리키는 노드가 그래프에 없음** (`no-dangling-edge`)
왜: relationships.json의 `to`가 entities.json에 없으면 순회가 끊깁니다.

```python
def check_missing_entities(entities, relationships) -> list[str]:
    ids = {e["id"] for e in entities}
    return [f"dangling: {r['from']} -{r['type']}-> {r['to']}"
            for r in relationships if r["from"] not in ids or r["to"] not in ids]
```

**③ Duplicate Concept — 같은 개념의 정본이 둘 이상** (`single-canonical-per-concept`)
왜: Dedup이 놓친 중복이 쌓이면 Section 1-2의 모순 답이 부활합니다. 임베딩 유사도로 정본끼리 비교.

```python
def check_duplicate_concepts(canonical_pages, embs, thr=0.93) -> list[str]:
    errs = []
    for i in range(len(embs)):
        for j in range(i+1, len(embs)):
            if cosine(embs[i], embs[j]) >= thr:
                errs.append(f"중복 의심: {canonical_pages[i]} ≈ {canonical_pages[j]}")
    return errs
```

**④ Stale Knowledge — Raw는 바뀌었는데 Wiki가 안 바뀜**
왜: 가장 위험. `invoice.py`가 어제 커밋됐는데 Wiki `last_verified`가 한 달 전이면, Wiki가 **현실과 어긋났을 가능성**. git mtime과 비교.

```python
import subprocess
from datetime import datetime, timezone

def git_last_modified(path: str) -> datetime:
    ts = subprocess.check_output(["git","log","-1","--format=%cI","--",path], text=True).strip()
    return datetime.fromisoformat(ts)

def check_stale(entities) -> list[str]:
    errs = []
    for e in entities:
        src = e.get("source", {}).get("code", "").split("#")[0]
        lv = e.get("metadata", {}).get("last_verified")
        if src and lv and pathlib.Path(src).exists():
            if git_last_modified(src) > datetime.fromisoformat(lv).replace(tzinfo=timezone.utc):
                errs.append(f"STALE: {e['id']} — {src}가 last_verified({lv}) 이후 변경됨")
    return errs
```

**⑤ Schema Violation — Constraints 위반** (Section 7-4)
왜: 미등록 kind/관계 타입, 출처 없는 canonical 등 → Schema Drift의 싹. `validate_entities`(7-4) + 규칙 검사.

### 10-2. CI 게이트로 — Wiki는 PR로 들어온다

🧪 **시나리오:** `acme-billing`의 `.github/workflows/wiki-lint.yml`이 모든 PR에서 `wiki-lint`를 돌립니다.
- 개발자가 `invoice.py`를 고치고 Wiki를 안 고치면 → **Stale**로 PR이 **빨간불**. "Wiki도 갱신하라"는 신호.
- 유지보수 에이전트(Section 12)가 Wiki를 갱신했는데 죽은 링크를 남기면 → **Broken Link**로 차단.
- 누가 임의 관계 타입 `mentions`를 쓰면 → **Schema Violation**으로 거부.

```yaml
# .github/workflows/wiki-lint.yml
name: wiki-lint
on: [pull_request]
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }       # git mtime(Stale 검사)용 전체 히스토리
      - run: pip install -r wiki/requirements.txt
      - run: python -m wikilint wiki/   # 위 5검사 통합 (Section 17에 전체 코드)
```

📌 **핵심:** Lint는 "Wiki를 코드처럼 다룬다"는 철학의 집행관입니다. **사람과 에이전트 양쪽이 만든 지식을 같은 기준으로 검증**합니다. Lint 없는 Wiki는 Section 19의 모든 실패(폭증·중복·환각·drift)에 무방비입니다.

> ⚠️ **Stale은 경고로 시작하라.** 처음부터 Stale을 hard-fail로 걸면 정상 코드 변경마다 Wiki 강제 갱신이라 팀이 지칩니다. **신규 코드 변경은 경고(annotation), 핵심 엔티티(`owners`+높은 중심성)만 차단**으로 단계 도입하세요.

### 🧭 Section 10 한 줄 요약

> Lint = Wiki의 CI. **Broken Link·Missing Entity·Duplicate·Stale·Schema Violation** 5종을 검사해 PR 게이트로 건다. Wiki를 *코드처럼* 다뤄, 사람·에이전트가 만든 지식을 같은 기준으로 검증한다. Lint 없는 Wiki는 반드시 썩는다(Section 19). Stale은 경고→핵심만 차단으로 단계 도입.

---
<a id="section-11"></a>
## Section 11. Agent-Driven Wiki Construction — Agent가 Wiki를 만든다

지금까지의 파이프라인을 **에이전트가 자율로 돌려** Wiki를 처음부터 짓는 과정입니다.

```
   새 Repository ─► Agent 분석 ─► Knowledge 추출 ─► Wiki 생성 ─► Schema 생성 ─► Lint ─► PR 생성
   (또는 기존)     (구조·진입점·   (Section 8       (Concept/     (entities/    (Sec.10) (사람 리뷰)
                   의존성 파악)     Extract)        Knowledge)    relationships)
```

### 11-1. 왜 Agent가 만드나 — 규모와 지속성

수백 개 엔티티의 Wiki를 사람이 손으로 쓰는 건 불가능하고, 써도 곧 낡습니다. **Agent가 초안을 짓고 사람이 정본을 승인**하는 분업이 정석입니다. Agent의 강점은 "코드 전체를 읽고 관계를 뽑는 노동"이고, 사람의 강점은 "이게 진짜 정본인가, 출처가 맞나"의 판단입니다.

### 11-2. 단계별 — `acme-billing` 부트스트랩

**① Agent 분석:** 진입점(`app/main.py`의 라우터 등록), 디렉터리 구조, 의존성(`pyproject.toml`), ADR 목록을 먼저 훑어 *지도*를 만듭니다. "이건 FastAPI 결제 서비스, 핵심 흐름은 구독·인보이스·웹훅."

**② Knowledge 추출:** Section 8의 Extract를 모든 코드 유닛 + ADR + 핵심 문서에 적용. 엔티티·관계·정본 진술 후보 생성.

**③~⑤ Wiki/Schema 생성 + Lint:** 페이지를 write, 그래프 생성, Lint 통과 확인.

**⑥ PR 생성:** `wiki/` 전체를 PR로. **`confidence`가 낮은 페이지에 리뷰 표시**를 달아 사람의 시선을 유도.

> 💻 Construction 오케스트레이션 (드라이버; 추출·Lint는 Section 8·17 재사용)

```python
def bootstrap_wiki(repo_root: str):
    units = list(iter_code_units(repo_root)) + list(iter_adrs(repo_root))
    all_entities, all_rels = [], []
    for u in units:
        ex = extract(u)                          # Section 8-1 (LLM 구조화 추출)
        for e in ex["entities"]:
            e.setdefault("metadata", {})["confidence"] = score_confidence(u, e)
            e["source"] = {u["src_kind"]: u["raw_id"]}     # provenance 강제
        all_entities += ex["entities"]
        all_rels += ex["relationships"]

    entities = normalize_and_dedup(all_entities)  # Section 8-2~8-4
    write_wiki_pages(entities, all_rels)          # Section 8-5
    regenerate_schema(entities, all_rels)
    errors = run_wiki_lint("wiki/")               # Section 10
    if errors:
        raise SystemExit("\n".join(errors))       # 깨진 Wiki는 PR 금지
    open_pr(branch="wiki/bootstrap",
            title="Bootstrap LLM Wiki",
            review_flags=[e["id"] for e in entities
                          if e["metadata"]["confidence"] < 0.7])  # 사람 시선 유도
```

🧪 **시나리오 (Claude Code로):** 팀 리드가 Claude Code에서 `/wiki bootstrap`(팀이 만든 커스텀 워크플로) 또는 평문으로 *"이 레포를 분석해서 `wiki/` 초안을 만들고, 확신 낮은 페이지는 표시해서 PR 올려"* 라고 시킵니다. Claude Code는 `glob`/`grep`/`read`로 레포를 훑고, 위 단계를 수행해 PR을 엽니다. 리드는 **20개 페이지 중 confidence<0.7인 4개만** 집중 리뷰합니다.

⚠️ **함정 (첫 부트스트랩의 환각):** 처음엔 ADR·Slack 같은 서사에서 Agent가 관계를 과잉 생성합니다. 방어: (1) 1차 부트스트랩은 **코드 Raw만**으로(사실 밀도 높음), 서사 Raw는 2차에 낮은 confidence로, (2) `single-canonical-per-concept` Lint로 과잉 정본 차단, (3) **사람 승인 없이는 어떤 페이지도 `canonical`이 안 됨**(생성 시 `draft`).

### 🧭 Section 11 한 줄 요약

> Construction = **분석→추출→Wiki/Schema 생성→Lint→PR**. Agent가 초안(`draft`)을 짓고 사람이 정본(`canonical`)을 승인하는 분업. confidence로 리뷰 우선순위를 매기고, 출처를 강제하고, Lint를 PR 전 게이트로 둬 환각을 막는다.

---

<a id="section-12"></a>
## Section 12. Agent-Driven Wiki Maintenance — 커밋이 Wiki를 갱신한다

Construction이 1회성이라면, Maintenance는 **매 커밋마다** 도는 지속 과정입니다. Wiki가 낡지 않는 비결.

```
   새 Commit ─► Wiki 영향 분석 ─► Page 업데이트 ─► Schema 수정 ─► Lint ─► 승인(머지)
   (PR)        (어느 페이지가     (변경분만        (그래프       (Sec.10)
               영향받나?)         재-Ingest)       갱신)
```

### 12-1. 핵심은 "영향 분석" — 전체 재빌드 금지

수만 엔티티 Wiki를 매 커밋마다 통째로 다시 빌드하면 비싸고 느립니다. **바뀐 Raw가 건드리는 페이지만** 골라 갱신해야 합니다. 이를 위해 **역 출처 인덱스(reverse provenance index)** 를 유지합니다: `raw_id → 영향받는 wiki 페이지`.

> 💻 커밋 diff → 영향받는 Wiki 페이지 (증분 갱신)

```python
import subprocess, json

def changed_files(base="origin/main", head="HEAD") -> list[str]:
    out = subprocess.check_output(["git","diff","--name-only",f"{base}...{head}"], text=True)
    return [l for l in out.splitlines() if l]

def affected_wiki_pages(changed: list[str]) -> set[str]:
    # 빌드 시 만들어 둔 역인덱스: { "app/services/invoice.py": ["wiki/knowledge/InvoiceService.md", ...] }
    index = json.load(open("wiki/.provenance_index.json"))
    pages = set()
    for f in changed:
        pages.update(index.get(f, []))
    return pages

# 예: app/tasks/billing.py 변경 + ADR-021 추가 →
#     {InvoiceService.md, renew_subscription.md, idempotent-invoice-issuance.md}
```

### 12-2. 관통 예시의 진짜 갱신 — BackgroundTasks → Celery (ADR-021)

🧵 `acme-billing`이 인보이스 발행을 **FastAPI `BackgroundTasks` → Celery로 이전**하는 PR이 올라옵니다(ADR-021). 이게 Wiki에 어떻게 전파되는지가 Maintenance의 정수입니다.

🧪 **시나리오 (Claude Code 유지보수 에이전트):**
1. **영향 분석:** diff에 `app/tasks/billing.py`(신규)·`app/api/subscriptions.py`(수정)·`ADR-021`(신규). 역인덱스로 `renew_subscription`, `InvoiceService`, `idempotent-invoice-issuance` 페이지가 영향.
2. **Page 업데이트:** 에이전트가 변경분을 재-Extract해서:
   - `renew_subscription` 관계: `triggers → issue_invoice_task(Celery)` 로 변경 (구 BackgroundTasks 관계 제거).
   - 새 Knowledge Page `issue_invoice_task.md` 생성(`draft`).
   - `idempotent-invoice-issuance` 페이지의 "동작" 섹션을 Celery 흐름으로 갱신.
   - **옛 진술을 지우지 않고** `superseded_by`로 표시 + ADR-021을 `supersedes` 엣지로.
3. **Schema 수정:** relationships.json에서 `renew → BackgroundTasks 핸들러` 엣지 제거, `renew → issue_invoice_task` 추가, `ADR-021 supersedes 옛방식` 추가.
4. **Lint:** Stale 해소 확인, Broken Link(옛 핸들러 참조 잔존?) 검사. Notion 온보딩의 "BackgroundTasks" 설명이 이제 **모순**으로 떠 리뷰 큐로.
5. **승인:** 에이전트가 **코드 PR과 같은 PR에 Wiki diff를 포함**시켜 올림. 리뷰어는 코드와 지식을 함께 승인. 머지 시 `draft → canonical` 승격.

📌 **결정적 패턴: "코드 변경과 Wiki 변경은 같은 PR."** 이래야 둘이 영원히 동기화됩니다. CI의 Stale Lint(Section 10-1④)가 "코드만 고치고 Wiki 안 고친" PR을 빨간불로 막아 이 습관을 **강제**합니다.

> 🔧 **실무 팁:** 유지보수 에이전트를 **PR 훅**(GitHub Actions의 `pull_request` 이벤트)으로 자동 기동하되, **자동 머지는 금물**. 에이전트는 Wiki diff를 *제안*하고, 사람이 *승인*합니다(Section 18 HITL). 자동 머지는 Schema Drift·환각이 무인 누적되는 지름길입니다(Section 19).

### 🧭 Section 12 한 줄 요약

> Maintenance = 매 커밋의 **영향분석(역 출처 인덱스로 변경 페이지만)→증분 갱신→Schema 수정→Lint→승인**. 옛 사실은 지우지 말고 `superseded`로 진화 기록. 철칙: **"코드 변경과 Wiki 변경은 같은 PR"**, Stale Lint가 이를 강제, 자동 머지는 금지(사람 승인).

---

<a id="section-13"></a>
## Section 13. 실무 Demo 1 — FastAPI 백엔드(`acme-billing`)에 Wiki 깔기

지금까지의 전부를 **하나의 흐름**으로 봅니다. 스택: **FastAPI + SQLAlchemy + Alembic + Redis + Claude Code.**

### 13-1. Day 0 — 부트스트랩 (Section 11 적용)

🧪 팀 리드가 Claude Code에서:

```
> 이 레포(acme-billing)를 분석해서 wiki/ 초안을 만들어줘.
  - app/** 코드는 AST 단위로, docs/adr/** 는 결정으로 추출
  - Concept/Knowledge 페이지 분리, 관계는 schema vocabulary만 사용
  - 모든 정본 진술에 source 강제, 확신 낮은 건 draft로
  - 끝나면 wiki-lint 돌리고 PR 올려
```

Claude Code가 만든 산출물(발췌) — 실제 Knowledge Page:

> 💻 `wiki/knowledge/renew_subscription.md` (생성 결과)

```markdown
---
id: knowledge.renew_subscription
type: knowledge
entity_kind: http_endpoint
title: POST /subscriptions/{id}/renew
status: draft            # ← 사람 승인 전
confidence: 0.86
owners: [team-billing]
last_verified: 2026-06-18
sources:
  - code: app/api/subscriptions.py#renew_subscription
related:
  - uses_pattern: "[[dependency-injection]]"
  - triggers: "[[issue_invoice_task]]"
  - reads: "[[Subscription]]"
---

## 무엇인가
구독을 다음 결제 주기로 갱신하는 엔드포인트. 인증된 Customer만 호출 가능.

## 시그니처
`POST /subscriptions/{id}/renew` → 202 Accepted (인보이스 발행은 비동기)

## 동작
1. `get_db`,`get_current_customer`를 [[dependency-injection]]로 주입받음
2. `SubscriptionService.renew(sub)` 로 다음 period 계산·상태 갱신
3. `issue_invoice_task.delay(sub.id, period)` 로 [[issue_invoice_task]] enqueue → 202 반환

## 주의
- 인보이스는 **동기 반환되지 않음**(202). 중복 발행 방지는 [[idempotent-invoice-issuance]] 참조.
```

리드는 PR에서 confidence 0.86(괜찮음)을 확인하고, `draft` 4개만 손봐 승인 → `canonical` 승격.

### 13-2. Day 30 — Alembic 마이그레이션이 Wiki를 건드림

🧵 누가 `migration 0042`로 `Invoice`에 `idempotency_key` 컬럼을 추가합니다. **추적·감사용**(어느 멱등 키로 발행됐는지 기록)이지 **unique 제약이 아닙니다** — 경합 차단은 여전히 Redis SETNX의 몫이고, DB unique는 ADR-017에서 폐기됐습니다.

🧪 Maintenance 에이전트(Section 12)가 PR 훅으로 기동:
- 영향: `Invoice.md`(컬럼 추가), `idempotent-invoice-issuance.md`(저장 방식 명확화).
- `Invoice.md`의 "스키마" 섹션에 `idempotency_key: str | None (non-unique, 추적용)` 추가, `migration: 0042`를 source에 추가. **정본 진술에 "이 컬럼은 가드가 아니라 기록"이라고 명시** → 미래의 에이전트가 unique 제약으로 "개선"하려다 ADR-017의 경합 500을 재현하는 일을 막음.
- Lint: Stale 해소. PR에 코드+Wiki diff 동봉.

### 13-3. Day 45 — 관통 질문 재방문 (성과 측정)

신입 D가 Claude Code에 🧵 "구독 갱신 시 인보이스 중복 발행 어떻게 막아?" 를 물음. 에이전트는 `wiki query`(Section 9)로:

```
Intent: traversal (seed: renew_subscription, invoice)
Traverse: renew_subscription -triggers-> issue_invoice_task -calls-> InvoiceService
          -implements-> idempotent-invoice-issuance -guarded_by-> redis-idempotency-key
Context: 위 5개 정본 페이지 (~1.2k 토큰, 캐시 read)
Answer: "renew는 202로 비동기 발행을 트리거하고, 멱등은 Redis SETNX
         idem:invoice:{sub_id}:{period}로 보장됩니다(ADR-017). period는 UTC 월초…"
```

**측정 (Before/After):**

| | RAG (Day 0) | LLM Wiki (Day 45) |
|---|---|---|
| 답 정확성 | ❌ 폐기안·옛 설명 혼입 | ✅ 정본·관계·출처 |
| 토큰/질문 | ~3,200 | ~1,200 (+ 캐시 read) |
| 세션 간 일관성 | 출렁임 | 동일 |
| 신입 온보딩 | 코드 재독 | Wiki 1페이지 |

### 🧭 Section 13 한 줄 요약

> FastAPI 백엔드 Demo: **Day0 부트스트랩(draft→승인)**, **Day30 마이그레이션이 증분 갱신을 유발**, **Day45 관통 질문이 정본·관계·저토큰으로 해결**. 코드와 Wiki가 같은 PR로 함께 진화하며, RAG 대비 정확성·일관성·비용이 측정 가능하게 개선된다.

---

<a id="section-14"></a>
## Section 14. 실무 Demo 2 — 대규모 GitHub 모노레포

`acme-billing`은 작았습니다. 이제 **수백만 줄·수십 패키지**의 모노레포(예: `acme-platform/` 안에 billing·auth·notifications·web·shared-libs)를 봅니다. 규모가 모든 것을 바꿉니다.

### 14-1. 규모가 만드는 3대 문제와 해법

| 문제 | 해법 |
|---|---|
| **컨텍스트 초과** — 레포가 1M에 안 들어감 | 패키지 단위로 **Wiki 샤딩** (`wiki/billing/`, `wiki/auth/`, …) + 상위 인덱스 |
| **추출 비용 폭증** — 수만 유닛 × LLM 호출 | **모델 계층화**(상세 추출 Opus, 대량 1차 추출 Sonnet/Haiku) + **Batch API**(50%↓) + **증분만**(Section 12) |
| **교차 패키지 의존** — billing이 auth를 import | 관계 엣지가 **패키지 경계를 넘음** → 의존성 그래프가 Wiki의 핵심 산출물 |

> 💻 대량 추출을 **Batch API**로 (지연 비민감 1차 추출, 50% 비용)

```python
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

reqs = [
    Request(custom_id=u["raw_id"],
            params=MessageCreateParamsNonStreaming(
                model="claude-sonnet-4-6",      # 대량 1차 추출은 Sonnet으로 비용·속도 균형
                max_tokens=2000,
                system="코드에서 엔티티/관계만 추출. 추측 금지.",
                messages=[{"role":"user","content": u["src"]}],
            ))
    for u in iter_code_units("acme-platform/billing")   # 수천 유닛
]
batch = client.messages.batches.create(requests=reqs)
# 완료 후 결과를 normalize→dedup→shard별 wiki write (Section 8)
```

> 🔧 모델 선택 원칙: **정확도가 Wiki 품질을 좌우**하므로 핵심 엔티티·관계는 `claude-opus-4-8`, 광범위 1차 스캔은 `claude-sonnet-4-6`, 단순 분류(intent 등)는 `claude-haiku-4-5`. 비용은 Batch + 증분으로 잡습니다.

### 14-2. 아키텍처·의존성 분석 — Wiki가 빛나는 지점

모노레포에서 Agent가 만드는 가장 가치 있는 산출물은 **아키텍처 Concept Page + 의존성 그래프**입니다.

> 💻 패키지 의존성 그래프를 Schema 엣지로 (예: import 분석)

```python
def package_dependency_edges(root):
    edges = []
    for unit in iter_code_units(root):
        pkg = unit["file"].split("/")[1]              # billing / auth / ...
        for imp in find_imports(unit["src"]):         # ast.Import / ImportFrom
            dep_pkg = resolve_package(imp)
            if dep_pkg and dep_pkg != pkg:
                edges.append({"from": f"package.{pkg}", "type": "depends_on",
                              "to": f"package.{dep_pkg}"})
    return dedup(edges)
# → package.billing -depends_on-> package.auth, package.shared-libs ...
```

🧪 **시나리오 (영향 범위 질문):** 플랫폼 팀이 `shared-libs`의 `Money` 타입을 바꾸려 합니다. Claude Code에 *"Money 타입을 바꾸면 어디가 깨지나?"* (intent=**impact**). Query 엔진이 `package.* -depends_on-> shared-libs` + `* -reads/writes-> Money`를 **역방향 순회**해 영향 패키지·엔티티 목록을 출력. RAG로는 절대 못 하는, 모노레포에서 가장 자주 필요한 질문입니다.

⚠️ **함정 (Wiki 폭증):** 모노레포에서 "모든 함수에 페이지"를 만들면 수만 페이지로 폭증합니다(Section 19). **선택적 추출**: 공개 API·교차 패키지 경계·핵심 도메인 엔티티만 Knowledge Page로, 내부 헬퍼는 페이지 없이 코드(Raw)에 맡깁니다. "Wiki는 코드의 복제가 아니라 *자명하지 않은 것*의 정본"이라는 Section 6 원칙을 규모에서 지키는 게 관건.

### 🧭 Section 14 한 줄 요약

> 모노레포는 규모가 문제: **샤딩(패키지별 Wiki)**, **모델 계층화+Batch+증분**으로 비용·컨텍스트 해결, **의존성 그래프**가 핵심 산출물(→ impact 질문). 폭증을 막으려면 *모든 것*이 아니라 **공개 API·경계·핵심 도메인만** 페이지화한다.

---

<a id="section-15"></a>
## Section 15. 실무 Demo 3 — 엔터프라이즈 문서 통합 (Notion·ADR·Slack·API Docs)

이번엔 **코드가 아니라 회사의 흩어진 문서**를 하나의 Wiki로 통합합니다. 가장 어렵고 가장 가치 있는 케이스 — 지식이 6개 시스템에 중복·모순된 채 흩어져 있으니까요.

### 15-1. 소스 커넥터 — 이질적 Raw를 한 형식으로

각 소스를 공통 Raw 단위(`{raw_id, kind, text, source_meta}`)로 끌어옵니다.

> 💻 멀티소스 커넥터 (공통 인터페이스)

```python
def load_notion(db_id):    # notion-client
    return [{"raw_id": f"notion://{p['id']}", "kind": "doc",
             "text": page_to_md(p), "source_meta": {"url": p["url"], "edited": p["last_edited_time"]}}
            for p in notion.databases.query(db_id)["results"]]

def load_slack(channel):   # slack_sdk; 사고·결정 스레드만 선별
    return [{"raw_id": f"slack://{channel}/{t['ts']}", "kind": "thread",
             "text": thread_text(t), "source_meta": {"channel": channel, "ts": t["ts"]}}
            for t in important_threads(channel)]

def load_openapi(path):    # API 계약
    spec = json.load(open(path))
    return [{"raw_id": f"openapi://{p}", "kind": "external_contract",
             "text": endpoint_md(p, op), "source_meta": {"path": p}}
            for p, op in iter_paths(spec)]

raw = load_notion(BILLING_DB) + load_slack("#billing-incidents") + load_openapi("openapi.json")
```

### 15-2. 교차 소스 Dedup·모순 해소 — 이 데모의 핵심

🧵 같은 "멱등 인보이스" 사실이 **코드(진실)·ADR-017(결정)·Slack(사고)·Notion(옛 온보딩, 틀림)** 에 있습니다. Ingest의 Dedup(Section 8-4)이 3분기합니다:

- 코드 + ADR-017 → **동일/보완** → 한 정본 페이지로 흡수, 둘 다 `source`에.
- Slack 사고 → **맥락 추가** → 정본 페이지의 `incident` 출처로.
- Notion "BackgroundTasks" → **모순** → 자동 흡수 금지, **리뷰 큐**. 사람이 "Notion이 낡음" 확인 → Notion 페이지에 `deprecated` + 정본으로 링크. (이상적으로는 Notion 원문도 갱신하도록 알림.)

📌 **출처의 권위 순위(authority ranking)** 를 정해 두면 모순 해소가 빨라집니다. 보통 **코드 > ADR > API Docs > Slack > Notion** (실행되는 것 > 결정 > 계약 > 대화 > 위키). 이건 회사마다 다르니 거버넌스로 합의(Section 18).

### 15-3. 접근 권한 — 엔터프라이즈의 필수

회사 문서는 부서·등급별 접근 권한이 있습니다. Wiki에 `visibility` 메타데이터를 박고, Query 시 **사용자 권한으로 필터**합니다(Section 7-3).

> 💻 권한 필터링된 검색

```python
def wiki_search_scoped(seeds, user, k=3):
    return vector_store.search(seeds, k=k, filter={
        "status": "canonical",
        "visibility": {"$in": visible_scopes(user)},   # 예: ["public","team-billing"]
    })
# 인사·재무 정본 페이지는 권한 없는 에이전트 세션에 노출되지 않음
```

⚠️ **함정:** 접근제어를 Query 단계에만 두면, 컨텍스트 조립·캐싱에서 새어나갈 수 있습니다. **검색·순회·컨텍스트·캐시 키 전 단계**에서 `visibility`를 일관 적용하세요. 권한 다른 사용자가 같은 캐시 프리픽스를 공유하지 않게 캐시 키에 scope를 포함.

🧪 **시나리오:** 결제팀 에이전트가 "환불 정책"을 물으면 `team-billing`+`public` 정본만 검색되고, 재무팀 전용 "수익인식 정책"은 안 보입니다. 같은 질문을 재무팀 에이전트가 하면 둘 다 보입니다 — **같은 Wiki, 권한별 다른 뷰.**

### 🧭 Section 15 한 줄 요약

> 엔터프라이즈 통합 = **멀티소스 커넥터(Notion·Slack·OpenAPI·ADR)로 공통 Raw화 → 교차소스 Dedup/모순해소(권위 순위: 코드>ADR>API>Slack>Notion, 모순은 사람) → 권한(`visibility`)을 전 단계 일관 적용**. 가장 어렵지만 "흩어진 사내 지식의 단일 정본화"라는 최대 가치를 낸다.

---
<a id="section-16"></a>
## Section 16. Claude Code / Codex / Cursor 에서의 활용 (가장 중요)

이론을 실전 도구에 연결합니다. 코딩 에이전트가 Wiki를 **읽고·검색하고·수정하고·활용해 더 정확한 코드를 쓰는** 네 가지를 `acme-billing`/Claude Code 기준으로 봅니다.

### 16-1. Agent가 Wiki를 **읽는** 법 — 진입점 + 점진적 공개

에이전트는 Wiki 전체를 컨텍스트에 욱여넣지 않습니다(그럼 Context Explosion 부활). 대신 **진입점 파일**이 "Wiki가 여기 있고, 이렇게 쓰라"고 알려주고, 에이전트가 **필요한 페이지만** 읽습니다(progressive disclosure).

> 💻 `CLAUDE.md` (레포 루트) — Claude Code의 진입점

```markdown
## 지식 베이스 (LLM Wiki)
이 레포의 정본 지식은 `wiki/`에 있다. 코드를 추론하기 전에 관련 정본 페이지를 먼저 읽어라.

- 개념/패턴: `wiki/concepts/` (예: idempotent-invoice-issuance, dependency-injection)
- 구체 엔티티: `wiki/knowledge/` (서비스/엔드포인트/모델별 1페이지)
- 관계 그래프: `wiki/schema/relationships.json` — "무엇이 무엇을 호출/트리거하나"는 여기서 순회
- 검색이 필요하면 `wiki query "<질문>"` (MCP 도구, 아래) 를 써라. 코드 전체를 다시 읽지 마라.

규칙: 정본 페이지와 코드가 충돌하면 코드(Raw)가 진실이다. 그 경우 Wiki Stale을 보고하라.
```

- **Codex** → `AGENTS.md`에 같은 내용. **Cursor** → `.cursor/rules/wiki.mdc`에 같은 규칙. 진입점 파일명만 다를 뿐 패턴은 동일합니다.

📌 핵심: 진입점은 **"전체 지식"이 아니라 "지식으로 가는 지도 + 사용 규칙"**. 실제 페이지는 에이전트가 그때그때 `read`로 끌어옵니다. 이게 Knowledge Compression이 런타임에 작동하는 방식.

### 16-2. Agent가 Wiki를 **검색하는** 법 — MCP 도구로 Query 엔진 노출

Section 9의 Query 엔진을 **MCP 서버**로 감싸면, Claude Code/Cursor가 `wiki_query`를 일반 도구처럼 호출합니다. 에이전트가 직접 벡터검색·그래프순회를 하는 게 아니라, **검증된 Query 엔진에 위임**합니다.

> 💻 Wiki Query를 MCP 도구로 (에이전트가 호출) — 핵심 핸들러

```python
# wiki_mcp_server.py  (MCP 서버; Claude Code가 도구로 인식)
# tool: wiki_query(question) -> {answer, sources, traversed_path}
def handle_wiki_query(question: str) -> dict:
    intent = analyze_intent(question)                 # Section 9-1
    seeds  = wiki_search(intent["seed_entities"])      # 9-2
    nodes  = traverse(G, seeds, intent.get("relation_focus", []))  # 9-3
    ctx    = build_context(nodes)                      # 9-4 (정본만)
    return {
        "answer": answer(question, nodes),             # 9-5
        "sources": [page_source(n) for n in nodes],    # provenance
        "traversed_path": list(nodes),                 # 관계 경로(설명가능성)
    }
```

🧪 **시나리오:** Claude Code가 `acme-billing`에서 환불 기능을 구현하다 *"환불 시 인보이스 상태를 어떻게 바꿔야 하지?"* 가 필요하면, 코드를 200파일 뒤지는 대신 `wiki_query("환불 시 인보이스 상태 전이")`를 호출 → 정본 답 + `Invoice` 상태머신 페이지 + 출처를 ~1k 토큰에 받음.

> 🔧 대안(도구 없이): MCP가 부담이면 `wiki query`를 단순 CLI로 만들고 `CLAUDE.md`에서 "Bash로 `python -m wiki query …`를 호출하라"고 안내해도 됩니다. 에이전트는 bash로 호출하고 결과를 읽습니다.

### 16-3. Agent가 Wiki를 **수정하는** 법 — 코드와 같은 PR

Section 12 그대로입니다. 에이전트가 코드를 고치면 **같은 PR에 Wiki diff**를 넣습니다. `CLAUDE.md`에 이 의무를 박아 둡니다:

```markdown
## Wiki 갱신 의무
정본 엔티티(`wiki/knowledge|concepts`에 페이지가 있는 것)의 동작/관계/불변식을 바꾸는 코드를
수정하면, 같은 PR에서 해당 Wiki 페이지도 갱신하라. 옛 사실은 삭제하지 말고 superseded로 표시.
PR 전 `python -m wikilint wiki/` 가 통과해야 한다.
```

CI의 Stale Lint(Section 10)가 이 의무를 **강제**합니다 — 안 지키면 PR 빨간불.

### 16-4. Agent가 Wiki로 **더 정확한 코드를 쓰는** 법

이게 최종 목적입니다. 정본 불변식을 알고 코딩하면 환각·버그가 줄어듭니다.

🧵 **Before (Wiki 없음):** Claude Code가 환불 기능을 짜며 인보이스를 `db.add(Invoice(...))`로 **직접 생성** → 멱등 우회 → Section 1-2의 중복 인보이스 버그 재발.

**After (Wiki 있음):** `CLAUDE.md` 규칙대로 먼저 `wiki_query`/`idempotent-invoice-issuance` 페이지를 읽음 → "인보이스는 반드시 `InvoiceService.issue_invoice`로, period당 1건" 불변식을 알게 됨 → `InvoiceService`를 통해 생성하는 올바른 코드를 작성. **정본 지식이 가드레일**로 작동.

📌 **핵심:** Wiki는 단순 검색 가속이 아니라 **에이전트의 코드 품질 가드레일**입니다. "이 코드베이스에서 옳은 방법"을 정본으로 박아 두면, 에이전트가 매번 그것을 따릅니다 — 팀의 컨벤션·불변식이 모든 에이전트 세션에 일관 적용됩니다.

### 🧭 Section 16 한 줄 요약

> 코딩 에이전트는 **(1) 진입점(`CLAUDE.md`/`AGENTS.md`/cursor rules)으로 Wiki를 발견·필요 페이지만 읽고, (2) `wiki_query` MCP/CLI로 검증된 Query 엔진에 검색을 위임하고, (3) 코드와 같은 PR로 Wiki를 수정(Stale Lint가 강제), (4) 정본 불변식을 가드레일 삼아 더 정확한 코드를 쓴다.** Wiki는 검색 가속이자 코드 품질 가드레일.

---

<a id="section-17"></a>
## Section 17. 구현 예제 (Python) — 바닥부터 조립

앞 섹션들에서 조각으로 본 코드를 **네 개의 실행 가능한 프로그램**으로 묶습니다. 프레임워크 없이 표준 라이브러리 + `anthropic` + `networkx`/`jsonschema`로 — Section 9의 RAG 교육이 마지막에 "프레임워크 없이 바닥부터"를 보여줬던 것과 같은 의도입니다.

### 예제 1 — 간단한 Wiki Generator (Raw 1개 → 페이지 1개)

```python
# wiki_gen.py — 코드 유닛 하나에서 Knowledge Page 한 장 생성
import anthropic, json, pathlib
client = anthropic.Anthropic()

def generate_page(unit: dict) -> str:
    ex = extract(unit)                     # Section 8-1 (구조화 추출)
    e = ex["entities"][0]
    related = "\n".join(f'  - {r["type"]}: "[[{r["to"].split(".")[-1]}]]"'
                        for r in ex["relationships"] if r["from"] == e["id"])
    page = f"""---
id: {e['id']}
type: knowledge
entity_kind: {e['kind']}
title: {e['title']}
status: draft
confidence: {score_confidence(unit, e):.2f}
last_verified: {today()}
sources:
  - code: {unit['raw_id']}
related:
{related}
---

## 무엇인가
{e['canonical_statement']}

## 불변식
""" + "\n".join(f"- {inv}" for inv in e.get("invariants", []))
    out = pathlib.Path(f"wiki/knowledge/{e['title']}.md")
    out.write_text(page, encoding="utf-8")
    return str(out)
```

> **이 코드가 하는 일:** Raw 유닛 → LLM 구조화 추출 → frontmatter(정본/관계/출처) + 본문(정본 진술·불변식) 페이지로 직렬화. `status:draft`/`confidence`로 **사람 승인 대기**, `sources`로 **provenance 강제**.

### 예제 2 — Wiki Linter (Section 10 통합 실행기)

```python
# wikilint/__main__.py — python -m wikilint wiki/
import sys, json, pathlib

def main(wiki_dir: str):
    entities = json.load(open(f"{wiki_dir}/schema/entities.json"))["entities"]
    rels     = json.load(open(f"{wiki_dir}/schema/relationships.json"))["relationships"]
    errors = []
    errors += validate_entities(entities)              # Schema Violation (7-4)
    errors += check_missing_entities(entities, rels)   # Missing Entity (10-1②)
    errors += check_broken_links(entities)             # Broken Link (10-1①)
    errors += check_duplicate_concepts(*load_canonical_embeddings(wiki_dir))  # Duplicate (③)
    errors += check_stale(entities)                    # Stale (④)
    if errors:
        print("\n".join(f"✗ {e}" for e in errors)); sys.exit(1)
    print("✓ wiki-lint passed"); sys.exit(0)

if __name__ == "__main__":
    main(sys.argv[1])
```

> **이 코드가 하는 일:** 5개 검사를 한 번에 돌려 비-0 종료로 **CI를 빨간불** 만듦. Section 10-2의 GitHub Actions가 이걸 호출. *사람과 에이전트가 만든 지식*을 동일 기준으로 검증하는 게이트.

### 예제 3 — Agent Query Engine (Section 9 통합)

```python
# wiki_query.py — python -m wiki_query "구독 갱신 시 인보이스 중복 막는 법?"
import sys, json, networkx as nx

def query(question: str) -> dict:
    intent = analyze_intent(question)                          # 9-1
    seeds  = wiki_search(intent["seed_entities"])              # 9-2
    G      = build_graph()                                     # 9-3
    nodes  = traverse(G, seeds, intent.get("relation_focus", []))
    text   = answer(question, nodes)                           # 9-4~9-5 (+캐시)
    return {"answer": text, "path": list(nodes),
            "sources": [n for n in nodes]}

if __name__ == "__main__":
    print(json.dumps(query(sys.argv[1]), ensure_ascii=False, indent=2))
```

> **이 코드가 하는 일:** intent 분류 → 정본 검색 → **그래프 순회**(RAG와의 결정적 차이) → 캐시된 정본 컨텍스트로 출처 포함 답. Section 16-2의 MCP 서버가 이 함수를 감쌈.

### 예제 4 — Repository → Wiki 자동 생성 (Section 11·8 통합 + PR)

```python
# build_wiki.py — 레포 전체를 Wiki로 (부트스트랩/전체 재빌드)
import subprocess

def build_wiki(repo_root: str, branch="wiki/auto"):
    units = list(iter_code_units(repo_root)) + list(iter_adrs(repo_root))  # 5-1
    ents, rels = [], []
    for u in units:                                   # 대규모면 batches로 (14-1)
        ex = extract(u)                               # 8-1
        for e in ex["entities"]:
            e["source"] = {u["src_kind"]: u["raw_id"]}
            e.setdefault("metadata", {})["confidence"] = score_confidence(u, e)
        ents += ex["entities"]; rels += ex["relationships"]

    ents = normalize_and_dedup(ents)                  # 8-2~8-4
    write_wiki_pages(ents, rels)                      # 8-5 (예제1 일반화)
    regenerate_schema(ents, rels)                     # entities/relationships.json
    build_provenance_index(ents)                      # 12-1 (증분 갱신용 역인덱스)

    if run_wiki_lint("wiki/"):                        # 예제2
        raise SystemExit("lint failed — PR 생성 중단")

    subprocess.run(["git","checkout","-B",branch]); subprocess.run(["git","add","wiki/"])
    subprocess.run(["git","commit","-m","build: regenerate LLM Wiki"])
    subprocess.run(["gh","pr","create","--fill",
                    "--title","Update LLM Wiki",
                    "--body","자동 생성. confidence<0.7 페이지는 리뷰 요망."])
```

> **이 코드가 하는 일:** Raw 수집 → 추출 → 정규화/Dedup → 페이지·Schema·**역 출처 인덱스** 생성 → **Lint 게이트** → 통과 시 PR. Section 11(구축)·12(유지보수의 역인덱스)·8(파이프라인)·10(Lint)을 한 드라이버로 종합. 이것이 "지식을 빌드하는" 실체.

> 📌 네 예제의 공통 신경: **(1) 모든 진술에 source, (2) 생성물은 draft, (3) Lint 게이트 통과 전엔 PR 금지, (4) 사람 승인으로 canonical 승격.** 이 네 가드가 Section 19의 실패를 코드 레벨에서 예방합니다.

### 🧭 Section 17 한 줄 요약

> 네 프로그램 — **Generator**(Raw→페이지)·**Linter**(5검사 CI 게이트)·**Query Engine**(검색+그래프순회+캐시)·**Repo→Wiki**(전체 빌드+PR). 프레임워크 없이 `anthropic`+`networkx`+`jsonschema`로 조립되며, *source 강제·draft 생성·Lint 게이트·사람 승인*이라는 공통 가드를 코드에 박는다.

---

<a id="section-18"></a>
## Section 18. 운영 전략 — Wiki를 "살아 있는 시스템"으로

Wiki는 한 번 만들고 끝이 아닙니다. 운영의 5요소.

### 18-1. Human in the Loop (HITL)
Agent는 **초안(`draft`)** 까지, 정본 승격은 **사람**이. 특히 모순 해소(Section 8-4 b분기)·정본 변경·관계 추가는 사람 승인 필수. 🧪 `acme-billing` 팀은 Wiki PR을 **2분 룰**로 다룹니다 — "코드 PR 리뷰 끝에 Wiki diff도 본다." 별도 의식이 아니라 코드리뷰에 흡수.

### 18-2. Approval Workflow
`owners` 메타데이터(Section 7-3) → **CODEOWNERS와 연동**. `wiki/knowledge/Invoice*`는 `@team-billing`이, `wiki/concepts/dependency-injection`은 `@team-platform`이 승인해야 머지. 도메인 정본을 함부로 못 바꾸게.

### 18-3. Ownership
모든 정본 페이지에 **소유 팀**. 소유 없는 페이지는 Lint 경고. "이 지식은 누구 책임인가"가 없으면 아무도 안 고치고 썩습니다.

### 18-4. Governance
- **관계/엔티티 vocabulary 변경**은 플랫폼 팀 승인(Schema는 전사 공용).
- **출처 권위 순위**(Section 15-2: 코드>ADR>API>Slack>Notion) 합의·문서화.
- **Wiki에 무엇을 페이지화할지 기준**(공개 API·경계·핵심 도메인만; Section 14) 합의 → 폭증 방지.

### 18-5. Knowledge Freshness
- **Stale Lint**(Section 10)가 1차 방어선.
- **주기적 재검증:** 핵심 정본 페이지는 분기마다 `last_verified` 갱신(에이전트가 코드와 대조 후 "여전히 맞음" 확인).
- **freshness 대시보드:** `status`별·`last_verified` 분포를 보면 "썩어가는 영역"이 보입니다.

### 18-6. 언제 투자하나 — 손익분기
⚠️ Wiki는 빌드·Lint·거버넌스 비용을 새로 만듭니다(Section 3). 도입 판단:

```
✅ 한다:  장수 코드베이스 · 여러 에이전트/사람이 반복 작업 · 온보딩 잦음 ·
          "관계/영향범위" 질문 빈번 · 정확성·일관성이 비용보다 중요
❌ 미룬다: 프로토타입 · 1회성 분석 · 곧 버릴 코드 · 1인 단기 프로젝트
```

📌 정량 신호: "에이전트가 **같은 도메인 지식을 주당 N회 이상 재발견**한다"면(Section 1-1) Wiki가 토큰·시간을 회수합니다. `acme-billing` 팀은 결제 도메인 질문이 주 50+회라 즉시 흑자.

### 🧭 Section 18 한 줄 요약

> 운영 5요소: **HITL**(draft는 Agent, canonical은 사람)·**Approval**(owners↔CODEOWNERS)·**Ownership**(주인 없는 페이지 금지)·**Governance**(vocabulary·권위순위·페이지화 기준 합의)·**Freshness**(Stale Lint+주기 재검증+대시보드). 손익분기는 "같은 지식의 반복 사용 빈도".

---

<a id="section-19"></a>
## Section 19. 실패 사례 — 원인과 해결책

운영하면 반드시 마주치는 5대 실패. 각각 **원인 → 증상 → 해결**.

| 실패 | 원인 | 증상 | 해결 |
|---|---|---|---|
| **Wiki 폭증** | "모든 함수에 페이지" | 수만 페이지, 검색 노이즈, 유지 불가 | 페이지화 기준(공개 API·경계·핵심만, 14·18), 내부 헬퍼는 코드에 위임 |
| **Duplicate Knowledge** | Dedup 부재/임계값 낮음 | 같은 사실 N벌, 모순 답 부활(1-2) | Dedup 단계(8-4)+Duplicate Lint(10③), `single-canonical` 규칙 |
| **Agent Hallucination** | 추출 LLM이 없는 사실/관계 지어냄 | 코드에 없는 "정본" | source 강제·"명시된 것만" 프롬프트·`canonical-needs-source` Lint·draft+사람승인(8·11·17) |
| **Wrong Relationship** | 잘못된/임의 관계 타입 | 순회가 엉뚱한 곳으로, impact 질문 오답 | vocabulary 고정(7-4)·`no-unknown-relationship-type` Lint·관계도 사람 리뷰 |
| **Schema Drift** | 규칙 없이 스키마가 제멋대로 진화 | kind/타입 난립, Lint 무력화 | Constraints(7-4)·Schema 거버넌스(18-4)·CI 강제(10) |

### 19-1. 가장 흔하고 위험한 것 — Stale(낡음)

위 표에 없지만 실전 1위는 **Stale**입니다. 코드는 바뀌는데 Wiki가 안 따라가면, **틀린 정본이 가드레일을 오염**시켜 에이전트가 자신 있게 틀린 코드를 씁니다(환각보다 위험 — "출처 있는 거짓").

🧵 사례: `acme-billing`이 멱등을 Redis SETNX → DB unique 제약으로 **되돌렸는데**(가상) Wiki를 안 고침. 에이전트는 정본을 믿고 SETNX 코드를 계속 생성 → 미묘한 버그.

해결: (1) **"코드+Wiki 같은 PR" + Stale Lint**가 근본 방어, (2) 코드와 Wiki 충돌 시 **"코드가 진실"** 규칙을 `CLAUDE.md`에 명시(16-1) → 에이전트가 모순을 *보고*하게, (3) 주기 재검증(18-5).

### 19-2. 메타 교훈
📌 5대 실패의 공통 해법은 **"Wiki를 코드처럼 운영"** 입니다 — 타입(Schema)·테스트(Lint)·코드리뷰(Approval)·CI(게이트)·소유권(owners). 이 엔지니어링 규율을 지식에 적용하는 게 LLM Wiki의 본질이고, 빼면 어김없이 위 실패가 옵니다.

### 🧭 Section 19 한 줄 요약

> 5대 실패(**폭증·중복·환각·잘못된 관계·Schema Drift**)와 숨은 1위(**Stale**)는 전부 **"Wiki를 코드처럼 운영"**(Schema=타입, Lint=테스트, Approval=리뷰, CI=게이트, owners=소유권)으로 예방된다. 특히 Stale은 "출처 있는 거짓"이라 환각보다 위험 — 코드+Wiki 동일 PR과 "코드가 진실" 규칙으로 막는다.

---

<a id="section-20"></a>
## Section 20. Best Practices — 기업 환경 기준 정리

지금까지를 **실전 체크리스트**로 압축합니다.

**아키텍처**
- [ ] Raw/Wiki/Schema **3계층 분리**. Wiki는 정본 산문, Schema는 순회·검증용 그래프(4·7).
- [ ] Wiki를 **레포에 커밋** — 버전관리·PR·CI 대상(4-3).
- [ ] Wiki는 코드 복제가 아니라 **자명하지 않은 것(불변식·관계·정본)의 정본**(6).

**파이프라인**
- [ ] Ingest의 **Dedup·모순 분기**를 반드시 둔다(동일=흡수, 모순=사람)(8-4).
- [ ] Query는 **검색 + 그래프 순회** 둘 다(관계 질문은 순회 없이 못 푼다)(9).
- [ ] **모든 진술에 source**, 생성물은 **draft**, 정본 승격은 **사람**(8·11·18).

**품질·운영**
- [ ] **Lint를 CI 게이트로**(Broken/Missing/Duplicate/Stale/Schema)(10).
- [ ] **"코드 변경과 Wiki 변경은 같은 PR"** — Stale Lint로 강제(12·16).
- [ ] **owners↔CODEOWNERS**, vocabulary·권위순위 **거버넌스 합의**(18).
- [ ] **status로 진화 관리** — 옛 사실은 삭제 말고 superseded(6-5·12).

**에이전트 통합**
- [ ] 진입점(`CLAUDE.md`/`AGENTS.md`/cursor rules)에 **Wiki 위치·사용 규칙·"코드가 진실"** 명시(16).
- [ ] Query 엔진을 **MCP/CLI 도구**로 노출, 에이전트가 위임(16-2).
- [ ] 정본 페이지를 **프롬프트 캐싱**으로 재사용(9-4·6-5).

**비용·규모**
- [ ] 모델 계층화(추출 Opus / 대량 Sonnet / 분류 Haiku) + **Batch + 증분**(14).
- [ ] 모노레포는 **샤딩**, 페이지화는 **공개 API·경계·핵심만**(14·19).
- [ ] **손익분기 확인** 후 투자(반복 사용 빈도)(18-6).

### 🧭 Section 20 한 줄 요약

> 베스트 프랙티스의 한 문장: **"Raw/Wiki/Schema 3계층을, source·draft·Lint·사람승인·동일PR·거버넌스라는 엔지니어링 규율로 운영하고, 에이전트에는 진입점+Query도구+캐시로 연결하라."**

---

<a id="appendix"></a>
## 부록

### 부록 A. 용어 사전

| 용어 | 뜻 |
|---|---|
| **LLM Wiki** | AI가 읽기 위한 정본·무중복·구조화된 지식 계층 |
| **Raw / Wiki / Schema Layer** | 원천 / AI용 정본 산문 / 기계용 그래프 |
| **Concept Page / Knowledge Page** | 개념·패턴 페이지 / 구체 엔티티 페이지 |
| **Canonical** | 현재 맞는 정본 진술 (status) |
| **Provenance** | 모든 진술의 Raw 출처 역추적 |
| **Relationship (typed edge)** | calls/implements/guarded_by/supersedes… 타입 관계 |
| **Ingest / Query / Lint** | 빌드 / 질의 / 검증 파이프라인 |
| **Dedup·Merge** | 같은 사실 흡수 / 기존 페이지에 합치기 |
| **Schema Traversal** | 관계 그래프 BFS 순회 (RAG와의 결정적 차이) |
| **Stale** | Raw는 변했는데 Wiki가 안 변한 상태 (실전 최대 위험) |
| **Schema Drift** | 규칙 없이 스키마가 썩는 것 |
| **Reverse Provenance Index** | `raw_id → 영향 페이지`, 증분 갱신용 |
| **HITL** | Agent 초안 + 사람 정본 승인 분업 |

### 부록 B. 구축 순서 체크리스트 (0→1)

1. 페이지화 **기준**과 관계 **vocabulary** 합의(거버넌스).
2. `iter_code_units` 등 **Raw 로더**(AST 경계) 마련.
3. **Extract**(구조화 출력) → Normalize → Dedup 파이프라인(8).
4. **Wiki/Schema 생성** + **역 출처 인덱스**(11·12).
5. **Linter** 5검사 + **CI 게이트**(10).
6. **Query 엔진**(검색+순회+캐시) + **MCP/CLI**(9·16).
7. 진입점(`CLAUDE.md`)에 **사용 규칙** 박기(16).
8. **부트스트랩 PR** → 사람 승인 → canonical(11).
9. **PR 훅 유지보수 에이전트** 가동(12).
10. **freshness 대시보드**·주기 재검증(18).

### 부록 C. "이것만은" — 최종 7문장 요약

1. RAG는 *청크 검색*, LLM Wiki는 *지식 관리* — Agent 시대의 진짜 문제는 후자다.
2. 3계층: **Raw**(원천)→**Wiki**(AI용 정본 산문)→**Schema**(기계용 그래프).
3. 핵심 무기는 RAG에 없던 **정본화·무중복·타입 관계·관계 순회**.
4. 지식은 손으로 쓰는 게 아니라 **빌드되는 산출물** — Ingest가 컴파일러다.
5. **모든 진술에 출처, 생성물은 draft, Lint 게이트, 사람이 canonical 승격.**
6. **"코드 변경과 Wiki 변경은 같은 PR"** — Stale이 환각보다 위험하다.
7. 에이전트에는 **진입점 + Query 도구 + 캐시**로 연결하면, Wiki는 검색 가속이자 **코드 품질 가드레일**이 된다.

### 부록 D. 참고 / 관련 기술

본 교안의 아키텍처는 다음 실제 사례·기술의 패턴을 **하나의 방법론으로 종합**한 것입니다(단일 표준 제품이 아니라 합성 아키텍처임을 밝힙니다).

- **CLAUDE.md / AGENTS.md** — 에이전트가 읽는 레포 가이드(수작업 Wiki 페이지의 원형).
- **Cursor Rules (`.cursor/rules`)** — 도메인 정본 지식 주입.
- **DeepWiki류 자동 위키 생성** — 레포 → 위키 자동화(Agent-Driven Construction의 실사례).
- **Microsoft GraphRAG** — 엔티티·관계 그래프 + 커뮤니티 요약(Schema Layer·관계 순회의 근거).
- **Anthropic Skills / MCP** — 점진적 공개·도구화(진입점·Query 위임의 실사례).
- **Anthropic Prompt Caching** — 정본 컨텍스트 재사용으로 비용 절감(9-4).

> ※ 코드 샘플은 2026년 기준 `anthropic` Python SDK·`claude-opus-4-8`/`sonnet-4-6`/`haiku-4-5` 모델, `networkx`·`jsonschema` 기준입니다. 실제 도입 시 자사 레포·문서 분포에서 임계값(Dedup 0.9~0.95 등)·모델 선택·페이지화 기준을 PoC로 검증하길 권장합니다.

---

*본 문서는 LLM/RAG/Embedding 기초를 갖춘 엔지니어 대상 "AI Agent 시대의 LLM Wiki 구축·운영 실무 교안"으로 작성되었습니다. 설명:실무 ≈ 40:60, `acme-billing`(FastAPI 결제 백엔드)과 관통 질문("구독 갱신 시 인보이스 중복 발행 방지")을 20개 섹션 내내 관통시켰습니다. 이후 단일 자기완결 HTML 교육 콘텐츠의 원본(Markdown)으로 사용합니다.*

