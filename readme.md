# ReflectRAG: Self-Reflective Retrieval-Augmented Generation

## What is ReflectRAG?

Traditional RAG pipelines blindly retrieve documents and pass them to an LLM regardless of whether those documents are actually relevant. **ReflectRAG** takes a smarter approach by grading the retrieved documents, evaluating the generated answer, and rewriting the query to retry if the result is not good enough.

Introduced in [Asai et al., 2023](https://arxiv.org/abs/2310.11511), ReflectRAG uses *reflection tokens* that allow a model to decide when to retrieve, assess document relevance, and verify that its generated answer is factually grounded. This implementation approximates that framework using LLM-based graders at each decision point rather than fine-tuned reflection tokens, and models the full loop as a stateful directed graph using **LangGraph**.

---

ReflectRAG is an AI-powered question-answering system that goes beyond traditional RAG by actively evaluating and critiquing its own outputs. Built with **LangGraph**, **LangChain**, **ChromaDB**, and **Google Gemini**, it delivers grounded, hallucination-resistant responses through a stateful, self-correcting pipeline.

---

## Features

- 🔍 Dynamic document retrieval from a ChromaDB vector store
- 🧠 LLM-powered relevance grading so only useful documents move forward
- ✅ Hallucination detection where generated answers are verified against source documents
- 🔁 Automatic query rewriting and retry when retrieval quality is poor
- 🚦 Transparent decision flow modeled as a directed graph with conditional edges
- 🛑 Recursion limit safeguard to prevent infinite rewrite loops

---

## Tech Stack

| Layer | Technology |
|---|---|
| Graph Orchestration | LangGraph |
| LLM & Chains | LangChain + LangChain Google GenAI |
| Vector Store | ChromaDB |
| Embeddings | Google Gemini Embeddings |
| Document Loading | LangChain Community + BeautifulSoup4 |
| Prompt Hub | LangChain Hub |
| Environment Management | python-dotenv |

---

## How It Works

The pipeline runs each query through a stateful graph with four intelligent decision nodes:

```
User Question
      │
      ▼
 [Retrieve] ──→ Fetch relevant documents from ChromaDB vector store
      │
      ▼
 [Grade Documents] ──→ Filter out irrelevant docs using an LLM grader
      │
      ├── Relevant docs found ──→ [Generate] ──→ [Grade Generation]
      │                                                  │
      │                                    ┌────────────┴────────────┐
      │                                    ▼                         ▼
      │                             Grounded &               Not grounded /
      │                             answers question ──→ ✅   off-topic ──→ 🔄
      │
      └── No relevant docs ──→ [Rewrite Query] ──→ loop back to Retrieve
                                     (up to recursion limit)
```

**Key decision nodes:**

- **Document Grader** Scores each retrieved document for relevance to the question.
- **Answer Grader** Verifies the generated answer is grounded in retrieved documents with no hallucinations.
- **Question Grader** Confirms the generation actually addresses what was asked.
- **Query Rewriter** Rephrases the question to improve retrieval if initial results fall short.

---

## Project Structure

```
ReflectRAG/
├── main.py              # Entry point, runs two demo scenarios
├── requirements.txt     # Python dependencies
└── src/
    └── graph.py         # LangGraph graph definition (nodes, edges, state)
```

---

## Demo Scenarios
 
Running `main.py` demonstrates two contrasting cases:
 
| Scenario | Query | Expected Behavior |
|---|---|---|
| 1 | *"What are large language models and how do they work?"* | Finds relevant docs, generates a grounded and cited answer ✅ |
| 2 | *"What is the best pizza recipe?"* | No relevant docs found, rewrites query repeatedly, stops gracefully at recursion limit 🛑 |
 
---

## Requirements

```
langgraph
langchain
langchain-google-genai
langchain_community
langchainhub
chromadb
langchain-text-splitters
beautifulsoup4
tiktoken
python-dotenv
```

## Setup Guide

### 1. Clone the repository

```bash
git clone https://github.com/Sahil-kumar2/ReflectRAG.git
cd ReflectRAG
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key

Create a `.env` file in the root directory:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

> Get your API key from [Google AI Studio](https://aistudio.google.com/).

### 5. Run the demo

```bash
python main.py
```

---

## Key Concepts

The `recursion_limit` parameter in `main.py` controls how many times the graph can loop before halting. This acts as a useful guardrail when no relevant documents exist in the store, preventing infinite rewrite cycles.

---

## 📚 References

- [ReflectRAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511) by Asai et al., 2023
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain ReflectRAG Tutorial](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_self_rag/)

---

## 🧑 Author

**Sahil Kumar**