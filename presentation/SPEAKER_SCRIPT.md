# Speaker Script - Financial Document RAG

**Total Time: 15 minutes**
**Format: Live presentation + demo, followed by 10-15 min code/design Q&A**

---

## Slide 1: Title (0:30)

> **What to say:**
> "Hi everyone, thanks for having me. Over the next 15 minutes, I'll walk you through a project I built - a Retrieval-Augmented Generation system for financial document Q&A. I'll cover the architecture, the design decisions I made, the evaluation framework I built around it, and where I'd take it to production. Then I'll do a quick live demo and we can dig into the code afterwards."

**Key signal:** Confident framing, sets clear expectations.

---

## Slide 2: The Problem (1:00)

> **What to say:**
> "The motivation here is simple. Financial analysts spend hours searching through 10-K filings, earnings reports, and SEC documents - each one running 100+ pages. Traditional keyword search misses semantic meaning - if a document says 'sales' but you searched for 'revenue', you miss it. And LLMs alone aren't the answer either - their training data is stale, their context windows are limited, and they hallucinate financial numbers, which is a non-starter for finance.
>
> RAG solves this elegantly: we keep the LLM's reasoning ability but ground it in real, retrievable, citable documents."

**Key signal:** Clear problem framing, shows you understand both why it matters and why naive solutions fail.

---

## Slide 3: My Approach - RAG Architecture (1:00)

> **What to say:**
> "I built two distinct pipelines:
>
> The **offline pipeline** runs once when documents are added - it parses PDFs, chunks them, embeds them, and stores them in a vector database. This is the slow, batch-style work.
>
> The **online pipeline** runs on every user query - it embeds the question, retrieves similar chunks, reranks them for precision, then sends them to an LLM to generate a grounded answer with citations.
>
> Separating these is critical - in production, you don't want to re-parse documents every time someone asks a question."

**Key signal:** Shows production thinking, separation of concerns.

---

## Slide 4: System Architecture (1:30)

> **What to say:**
> "Here's the full architecture. Five core components, each independent and swappable:
>
> 1. **Parser** - extracts text and structure from PDFs
> 2. **Chunker** - splits documents into searchable pieces while preserving page metadata for citations
> 3. **Embedder** - converts text into vectors that capture meaning
> 4. **Vector Store** - ChromaDB with HNSW index for fast similarity search
> 5. **Reranker + LLM** - the final precision and synthesis layer
>
> I used dependency injection throughout, so any component can be swapped - I can replace ChromaDB with Pinecone or Vertex AI Vector Search by changing one class. Same with the embedder, the LLM, everything."

**Key signal:** Modularity, swappability, production mindset.

**If asked:** "Why ChromaDB?" → "Open source, runs locally, easy to develop against. For production I'd move to Vertex AI Vector Search or Pinecone for scale."

---

## Slide 5: Indexing Pipeline Deep Dive (1:30)

> **What to say:**
> "Let me walk through some non-obvious decisions in the indexing pipeline.
>
> For the **parser**, I use the unstructured library which handles complex PDF layouts well, but I built a PyMuPDF fallback for environments where unstructured can't be installed. That's a real production concern.
>
> For the **chunker**, the interesting decision is preserving page numbers in metadata. This is what enables citations in the final answer - 'According to NVIDIA's 10-K, page 15...'. Without page metadata, you lose traceability.
>
> For the **embedder**, I chose all-MiniLM-L6-v2 - it's only 80MB, generates 384-dim vectors, runs locally with no API costs, and gives competitive quality. For a higher-quality option I'd consider OpenAI's text-embedding-3 or BGE models."

**Key signal:** Pragmatic decisions, awareness of trade-offs, production thinking.

---

## Slide 6: Retrieval - The Two-Stage Pattern (1:30)

> **What to say:**
> "This is one of the most important design decisions in the system - the two-stage retrieval pattern.
>
> Stage one: vector search using a **bi-encoder**. The query and documents are embedded separately and compared with cosine similarity. This is fast - we can search 10,000+ chunks in milliseconds - but it's approximate.
>
> Stage two: reranking with a **cross-encoder**. Here, the query and each candidate document are processed together by the model, which captures word-level interactions and gives much better relevance scoring. But it's slow - it can't scale to a full database.
>
> So we use them together: bi-encoder retrieves the top 20, cross-encoder reranks to top 5. This is a standard pattern in production search systems - Bing, Google, every major search engine does something similar."

**Key signal:** Deep understanding of retrieval, production patterns.

**If asked:** "Why top 20 → top 5? Why not 50 → 5?" → "Trade-off between recall and latency. 20 catches most relevant docs while keeping reranking fast. I'd tune this based on the use case."

---

## Slide 7: Generation - Preventing Hallucination (1:00)

> **What to say:**
> "The 'G' in RAG is generation, but the real challenge here is preventing hallucination - especially in finance, where made-up numbers are a serious risk.
>
> I use four strategies:
>
> 1. **System prompt engineering** - explicit instruction to use ONLY the provided context
> 2. **Permission to say 'I don't know'** - the prompt explicitly allows refusal when context is insufficient
> 3. **Low temperature** - 0.1 keeps responses deterministic and factual
> 4. **Citation requirements** - every answer cites the source document and page
>
> I'm using Gemini 2.5 Flash for fast inference, but the wrapper class makes it trivial to swap to Claude, GPT-4, or any other LLM."

**Key signal:** Awareness of LLM risks, prompt engineering skill.

---

## Slide 8: Evaluation Framework (1:30)

> **What to say:**
> "An ML system without evaluation is just guessing. So I built a complete evaluation framework with four metric categories:
>
> - **Retrieval metrics**: Precision@K and Recall@K - of the chunks I retrieved, how many are actually relevant?
> - **Generation metrics**: Faithfulness using LLM-as-judge - does the answer stay grounded in the context?
> - **Latency**: end-to-end response time - critical for production
> - **Experiment tracking**: every run is logged with config and results, exportable to CSV
>
> I created a curated test set with 10 questions across NVIDIA and Apple 10-K filings, with manually verified ground truth chunk IDs and expected answers. This is what lets me make data-driven decisions."

**Key signal:** ML rigor, MLOps awareness.

---

## Slide 9: Experimental Results (1:30)

> **What to say:**
> "Here are real results from running the evaluation across three chunk size configurations.
>
> The headline finding: smaller chunks - 400 to 800 characters - outperform larger 1200-character chunks on both precision and recall, AND they're faster. With 1200 chars, latency jumps from 3 seconds to over 5 seconds.
>
> But there's a story behind this. When I first ran experiments, I noticed chunk size wasn't affecting results at all. Turns out my chunking code had a bug - it was treating each PDF page as a single chunk, ignoring the chunk_size parameter entirely. I rewrote the chunker to properly split large pages, and that's when I started getting meaningful comparisons. Without the evaluation framework, I would have never found that bug."

**Key signal:** Engineering rigor, debugging skill, intellectual honesty about mistakes.

---

## Slide 10: Live Demo (2:30)

> **What to say:**
> "Let me show you this running. [SWITCH TO STREAMLIT APP]
>
> "First, I'll ask a specific factual question - 'What was NVIDIA's revenue for fiscal year 2024?' [Run query]. You can see the answer with the specific number, and below it, the source citations - it's pulling from NVIDIA's 10-K, page 15. I can expand each source to verify the answer.
>
> "Now a multi-document question: 'What are the main risk factors for both companies?' [Run query]. Notice it pulls from both NVIDIA and Apple documents, synthesizes them in the answer, and cites both.
>
> "And here's the important one - watch what happens when I ask something not in the documents: 'When was the CEO born?' [Run query]. The system correctly says it cannot find this information, instead of hallucinating an answer. This is faithfulness in action."

**Key signal:** Working system, transparency, well-handled failure mode.

**Demo prep checklist:**
- [ ] Streamlit app already running
- [ ] Browser window ready to switch to
- [ ] Test all 3 questions beforehand to ensure they work
- [ ] Have a fallback if API fails

---

## Slide 11: Production & GCP Migration (1:00)

> **What to say:**
> "Today this runs locally - tomorrow it could run on GCP. The architecture maps cleanly:
>
> - ChromaDB → **Vertex AI Vector Search** for managed scaling
> - Sentence Transformers → **Vertex AI Embeddings API** if we want managed inference
> - Streamlit → **Cloud Run** for stateless deployment with auto-scaling
> - Local Gemini API → **Vertex AI Gemini** for enterprise-grade LLM access
>
> The migration is straightforward because of the modular design - I'd swap implementations behind the same interfaces. I'd also add caching with Memorystore, monitoring with Cloud Logging, and an A/B testing framework so we can run experiments in production safely."

**Key signal:** GCP fluency, production roadmap, MLOps awareness.

---

## Slide 12: Lessons Learned (0:45)

> **What to say:**
> "Three things I'd highlight from building this:
>
> 1. **Build evaluation first.** Without it, you can't tell if changes are improvements. The chunker bug I mentioned earlier - I only found it because I was running rigorous experiments.
>
> 2. **The two-stage retrieval pattern matters.** Skipping reranking is the difference between mediocre and good results.
>
> 3. **Prompt engineering is engineering.** Treating the system prompt as a critical, versioned piece of infrastructure - with explicit constraints, fallback behaviors, and citation requirements - is what separates a hallucinating system from a trustworthy one."

**Key signal:** Reflection, senior-level insights.

---

## Slide 13: What I'd Do Next (0:45)

> **What to say:**
> "If I had another sprint on this, my priorities would be:
>
> 1. **Hybrid retrieval** - combining semantic search with BM25 keyword search for better coverage on specific terminology
> 2. **Query rewriting** - using a smaller LLM to expand or clarify ambiguous queries before retrieval
> 3. **Caching layer** - common questions are common; we shouldn't re-embed and re-call the LLM every time
> 4. **Fine-tuning the reranker** on financial domain data for even better precision
> 5. **Multi-modal support** - financial documents have charts and tables that pure text extraction misses"

**Key signal:** Roadmap thinking, awareness of state-of-the-art techniques.

---

## Slide 14: Known Limitations (0:45)

> **What to say:**
> "I want to be honest about the limitations:
>
> 1. **Single-tenant** - this isn't multi-user safe yet; you'd need user-scoped collections
> 2. **No table extraction** - financial documents have tables that PDF parsing handles poorly
> 3. **No conversation history** - each query is independent; we'd need session state for follow-ups
> 4. **Evaluation set is small** - 10 questions; production would need hundreds with continuous test set growth
> 5. **No production observability** - I'd add Datadog or Cloud Monitoring before deploying for real
>
> These are all addressable - I designed the architecture so each can be solved without major refactoring."

**Key signal:** Maturity, honesty, awareness of what's missing.

---

## Slide 15: Closing (0:30)

> **What to say:**
> "To summarize: I built an end-to-end RAG system with proper evaluation, experiment tracking, and a clear path to production. The architecture is modular, the decisions are documented, and the code is ready for the deep-dive Q&A.
>
> Happy to dig into any component - the chunking strategy, the retrieval pipeline, the evaluation framework, the deployment story. What would you like to explore first?"

**Key signal:** Confident close, invites Q&A on your terms.

---

# Q&A Prep - Likely Questions

## Architecture Questions

**Q: Why RAG over fine-tuning?**
> A: Three reasons: (1) Fine-tuning requires retraining when data changes - RAG just re-indexes. (2) Fine-tuning loses citations - RAG preserves them, which is critical for finance. (3) Cost - fine-tuning is expensive; RAG uses smaller models with retrieval.

**Q: Why ChromaDB instead of Pinecone/Weaviate?**
> A: For prototyping, ChromaDB is faster to start with - no API keys, runs locally, free. For production with many concurrent users, I'd move to Vertex AI Vector Search or Pinecone for managed scaling.

**Q: How do you handle very long documents?**
> A: The chunker splits them - any document gets broken into ~800 char chunks with overlap. The vector store doesn't care about original document length; it cares about chunks.

## Retrieval Questions

**Q: How do you choose chunk size?**
> A: Empirically. I ran experiments with 400, 800, and 1200 char chunks. 400-800 won on retrieval metrics and latency. There's no universal answer - depends on document type and query type.

**Q: What's HNSW?**
> A: Hierarchical Navigable Small World - a graph-based approximate nearest neighbor algorithm. It builds a layered graph that lets search traverse intelligently in O(log n) instead of brute-force O(n). Standard for vector DBs.

**Q: Why retrieve 20 and rerank to 5?**
> A: Vector search is approximate, so casting a wider net (20) catches relevant docs that might have lower embedding similarity. Reranking with cross-encoder is more precise but slower - 20 is the sweet spot.

## Generation Questions

**Q: How do you prevent hallucination?**
> A: Four-pronged: explicit prompt constraints, allow "I don't know" responses, temperature=0.1, citation requirements. Plus evaluation - I measure faithfulness with LLM-as-judge.

**Q: Why temperature 0.1 instead of 0?**
> A: 0 is fully deterministic but can get stuck on poor outputs. 0.1 allows tiny variation while staying factual. For creative tasks I'd use 0.7+.

## Evaluation Questions

**Q: How did you build ground truth?**
> A: Manually for the curated test set - I read the documents and identified relevant chunks for each question. For scaling, I'd use LLM-assisted ground truth generation with human review.

**Q: What if chunk IDs change when chunk size changes?**
> A: Great question - they do, which broke my initial evaluation. Solution: regenerate ground truth per chunk configuration, OR use page-based metrics that are invariant to chunking. I built the former.

## Production Questions

**Q: How would you scale this to 10M documents?**
> A: Vertex AI Vector Search for managed ANN, sharded by document type or date. GPU embedding generation. Caching layer. Multi-region for latency.

**Q: How do you handle document updates?**
> A: Delete old chunks by ID, re-ingest. ChromaDB supports upserts. For real-time, you'd want change data capture from the source system.

**Q: What about cost?**
> A: Embedding generation is the cheap part - one-time. The expensive part is the LLM call per query. Strategies: cache common queries, use cheaper models for simple questions, batch where possible.

## Code Questions

**Q: Walk me through what happens when a user asks a question.**
> A: Question hits `RAGPipeline.query()`. We optionally apply a metadata filter, then call `vector_store.search()` which embeds the query and runs ChromaDB's HNSW search for top-20. Then `reranker.rerank()` runs the cross-encoder on (query, doc) pairs and returns top-5. Finally `llm.generate()` formats the chunks with source citations, builds the prompt, and calls Gemini.

**Q: What design patterns are you using?**
> A: Wrapper pattern (each external library wrapped in a clean class), Dependency Injection (components passed into pipeline), Factory pattern (`create_pipeline()`), Strategy/Null Object pattern (`NoOpReranker` and `MockLLM`), Lazy Loading throughout.

---

# Final Tips

1. **Start strong** - first 30 seconds set the tone. Sound confident.
2. **Don't read slides** - use them as anchors, talk naturally.
3. **Time check** - aim for 13 minutes spoken + 2 min demo = 15 total.
4. **In Q&A** - it's okay to say "I'd need to think about that" for hard questions.
5. **For code Q&A** - reference the actual files: "If we open `src/pipeline.py`, you'll see..."
6. **Energy** - they want to see passion. You built this end-to-end. Own it.

You got this.
