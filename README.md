# DHL Stakeholder QA — Agentic RAG System

> An end-to-end AI system built over the **DHL Group Stakeholder Engagement Guideline**.  
> Demonstrates RAG pipelines, multi-agent orchestration, prompt versioning, agent tracing, and LLM evaluation.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Setup](#setup)
- [System 1 — RAG Q&A Pipeline](#system-1--rag-qa-pipeline)
- [System 2 — Stakeholder Triage (Multi-Agent)](#system-2--stakeholder-triage-multi-agent)
- [Prompt Versioning](#prompt-versioning)
- [Evaluation](#evaluation)
- [LangSmith Tracing](#langsmith-tracing)
- [Results](#results)


---

## Overview

This project implements two independent AI systems grounded in DHL's official Stakeholder Engagement Guideline (Sections 3, 5, and 6).

**System 1** answers general questions about DHL's stakeholder engagement policies using a RAG pipeline with two versioned prompts evaluated by RAGAS.

**System 2** automates stakeholder issue triage using a LangGraph multi-agent system with conditional routing — classifying issues, assessing risk, and recommending a course of action grounded in the guideline. Agent 3's prompt is versioned and evaluated using LLM-as-Judge.

---

## Architecture

### System 1 — RAG Q&A Pipeline

```
DHL PDF
   │
   ▼
PyPDFLoader → RecursiveCharacterTextSplitter → BGE-small-en embeddings
   │
   ▼
ChromaDB (vector store)
   │
   ▼
Query → Retriever (top-3 chunks) → Versioned Prompt → Groq LLM → Answer
   │
   ▼
ReAct Agent (tool routing)
   ├── Tool 1: dhl_document_search  (question in the document)
   └── Tool 2: web_search           (question outside the document)
```

### System 2 — Stakeholder Triage (LangGraph)

```
Incoming stakeholder message
           │
           ▼
   ┌───────────────┐
   │  CLASSIFIER   │  Agent 1 — identifies issue type,
   │    AGENT      │  stakeholder group, owning department
   └───────┬───────┘  (grounded in §3)
           │
           ▼
   ┌───────────────┐
   │ RISK ASSESSOR │  Agent 2 — assesses urgency level
   │    AGENT      │  and potential ramifications
   └───────┬───────┘  (grounded in §6)
           │
           ▼ [CONDITIONAL EDGE]
           │
     ┌─────┴──────┐
     │            │
High/Critical   Low/Medium
     │            │
     ▼            ▼
┌─────────┐  ┌──────────┐
│ ACTION  │  │PLAYBOOK  │  Agent 3a: LLM reasoning
│ ADVISOR │  │MATCHER   │  Agent 3b: deterministic lookup
│ AGENT   │  │          │  (both grounded in §5 + §6)
└────┬────┘  └────┬─────┘
     │            │
     └─────┬──────┘
           ▼
   ┌───────────────┐
   │    REPORT     │  Structured triage report
   │   COMPILER    │  with full audit trail
   └───────────────┘
```

---

## Project Structure

```
dhl-stakeholder-rag/
│
├── DHL_RAG_Project.ipynb          # Main notebook — run cells top to bottom
│
├── prompts/                       # Versioned prompt files (YAML)
│   ├── rag_v1.0.yaml              # RAG prompt — baseline (deprecated)
│   ├── rag_v2.0.yaml              # RAG prompt — structured + citations (active)
│   ├── action_advisor_v1.0.yaml   # Action Advisor — direct (deprecated)
│   └── action_advisor_v2.0.yaml   # Action Advisor — chain-of-thought (active)
│
├── requirements.txt               # All Python dependencies
├── .env.example                   # API key template (copy to .env)
├── .gitignore                     # Excludes .env, chroma_db, PDFs
└── README.md
```

---

## Tech Stack

| Component | Tool | Purpose |
|---|---|---|
| LLM | Groq `llama-3.1-8b-instant` | Fast, free inference |
| Embeddings | `BGE-small-en-v1.5` | Semantic search |
| Vector store | ChromaDB | Document retrieval |
| RAG framework | LangChain | Chain + ReAct agent |
| Multi-agent | LangGraph | Stateful agent orchestration |
| Prompt versioning | YAML + PyYAML | Tracked prompt iterations |
| RAG evaluation | RAGAS | Faithfulness, Answer Relevance, Context Precision |
| Agent evaluation | LLM-as-Judge (Groq) | Guideline Faithfulness, Urgency Appropriateness, Action Concreteness |
| Tracing | LangSmith | Full agent execution traces |
| Web search | Tavily | Fallback for out-of-document queries |
| Environment | python-dotenv | Secure API key management |

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/dhl-stakeholder-rag.git
cd dhl-stakeholder-rag
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API keys

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

```
GROQ_API_KEY=your_groq_api_key_here
LANGCHAIN_API_KEY=your_langsmith_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

Get free keys here:
- **Groq**: https://console.groq.com
- **LangSmith**: https://smith.langchain.com
- **Tavily**: https://app.tavily.com

### 4. Upload the DHL document

Upload `stakeholder-engagement-guideline.pdf` when prompted in Cell 3 of the notebook.

### 5. Run in Google Colab

Upload the notebook to [Google Colab](https://colab.research.google.com), upload your `.env` file using the file browser (left panel), then run cells top to bottom.

---

## System 1 — RAG Q&A Pipeline

Answers general questions about DHL's stakeholder engagement policies by retrieving relevant chunks from the guideline document.

**Key design decisions:**
- **Chunk size 500 / overlap 100** — preserves context across section boundaries
- **BGE-small-en-v1.5** — same embedding model used in production deployments; normalized embeddings for better cosine similarity
- **Top-3 retrieval** — balances context richness vs. prompt size
- **ReAct agent** — routes to document search or web search depending on whether the question is covered in the document

**Example queries:**
```
"What are DHL's guiding principles for stakeholder engagement?"
→ Uses document retrieval tool (answer found in §4)

"Who is the current CEO of DHL Group?"
→ Routes to web search (not in the document)
```

---

## System 2 — Stakeholder Triage (Multi-Agent)

Automates the triage workflow described in §6 of the DHL guideline:
> *"This department will ascertain the level of urgency of the respective issue and its potential ramifications and suggest an appropriate course of action."*

### Agents

| Agent | Input | Output | Guideline section |
|---|---|---|---|
| Classifier | Raw message | Issue type, stakeholder group, owning dept | §3 |
| Risk Assessor | Classified message | Urgency level, risk type, ramifications | §6 |
| Action Advisor *(High/Critical)* | Full triage context | Engagement format, steps, timeline | §5 + §6 |
| Playbook Matcher *(Low/Medium)* | Stakeholder + issue type | Pre-defined standard response | §5 |

### Conditional routing

After the Risk Assessor, a **conditional edge** routes the workflow:
- `High` or `Critical` urgency → **Action Advisor** (LLM reasoning — novel situations)
- `Low` or `Medium` urgency → **Playbook Matcher** (deterministic — no LLM call, faster and cheaper)

This demonstrates that good agent orchestration knows **when not to call an LLM**.

---

## Prompt Versioning

Prompts are stored as YAML files in the `prompts/` folder. Each file tracks:

```yaml
version: "2.0"
created: "2025-04-22"
author: "Mariem"
status: "active"          # active | deprecated | draft
change_log: >
  Added chain-of-thought reasoning after v1.0 showed low
  Guideline Faithfulness in LLM-as-Judge evaluation.
eval_scores: null         # filled automatically after evaluation runs
template: |
  You are an action planning specialist...
```

**Workflow for adding a new prompt version:**

```bash
# 1. Copy the current active version
cp prompts/action_advisor_v2.0.yaml prompts/action_advisor_v3.0.yaml

# 2. Edit the new file — update version, change_log, status, template
# 3. Set the old version to deprecated in its YAML file
# 4. Commit and tag
git add prompts/
git commit -m "prompt: action_advisor v3.0 — added few-shot examples"
git tag prompt-action-advisor-v3.0
```

After evaluation runs, scores are written back to the YAML automatically:

```python
update_eval_scores("action_advisor_v2.0", {
    "guideline_faithfulness": 4.2,
    "urgency_appropriateness": 4.5,
    "action_concreteness": 3.8,
    "overall_score": 4.2
})
```

---

## Evaluation

### System 1 — RAGAS metrics

Evaluated on a 5-question test set against the DHL guideline.

| Metric | rag_v1.0 | rag_v2.0 | Change |
|---|---|---|---|
| Answer Relevancy | — | — | — |
| Faithfulness | — | — | — |
| Context Precision | — | — | — |

> Scores are filled automatically when the RAGAS evaluation cell runs and written back to the YAML files via `update_eval_scores()`.

### System 2 — LLM-as-Judge

The Action Advisor (Agent 3) is evaluated independently across 3 test scenarios using a separate Groq judge LLM. No ground truth required — the DHL guideline document is the reference.

| Dimension | What it measures | Scale |
|---|---|---|
| Guideline Faithfulness | Are steps traceable to the document? | 1–5 |
| Urgency Appropriateness | Is the response proportional to urgency? | 1–5 |
| Action Concreteness | Are steps specific and actionable? | 1–5 |

| Metric | action_advisor_v1.0 | action_advisor_v2.0 | Change |
|---|---|---|---|
| Guideline Faithfulness | — | — | — |
| Urgency Appropriateness | — | — | — |
| Action Concreteness | — | — | — |
| Overall Score | — | — | — |

> Scores filled automatically after the LLM-as-Judge cell runs.

---

## LangSmith Tracing

Every agent execution is traced in LangSmith under the project **DHL-Stakeholder-RAG**.

Custom `@traceable` spans are added per agent:

| Span name | run_type | What it shows |
|---|---|---|
| `Agent1_Classifier` | chain | Classification decision + retrieved chunks |
| `Agent2_RiskAssessor` | chain | Urgency assessment + routing decision |
| `Agent3a_ActionAdvisor` | chain | LLM reasoning trace + prompt version used |
| `Agent3b_PlaybookMatcher` | tool | Playbook key matched — confirms no LLM call |
| `LLM_Judge_ActionAdvisor` | llm | Judge scores + reasoning per test case |

Opening two traces side by side (High urgency vs Low urgency) clearly shows the conditional routing in action: the High urgency trace has a deeper tree with the Action Advisor LLM call; the Low urgency trace shows the Playbook Matcher as a tool span with near-zero latency.

---


## Author

**Mariem Kallel** — AI Engineer  
📧 mariemkallel36@gmail.com  
🌐 [Portfolio](https://mariemportfolio.vercel.app/)
