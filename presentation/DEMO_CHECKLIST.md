# Pre-Interview Checklist (Last 50 minutes)

## CRITICAL - Do These First (15 min)

### 1. Test the demo end-to-end (10 min)
```powershell
# In your venv terminal:
streamlit run app.py
```

Test these 3 questions and confirm they all work:
- [ ] "What was NVIDIA's revenue for fiscal year 2024?" → Should give specific number with citation
- [ ] "Compare risk factors for both companies" → Should pull from both NVIDIA and Apple docs
- [ ] "When was the CEO born?" → Should say "I cannot find this information"

If any fail:
- Check `.env` has valid `GOOGLE_API_KEY`
- Verify documents are ingested: should see chunk count in sidebar
- Worst case: enable Mock Mode for demo (still shows the system structure)

### 2. Open the deck (2 min)
- Path: `presentation\financial_rag_deck.pptx`
- Open it in PowerPoint, run through slides once
- Make sure all transitions work

### 3. Have your code ready (3 min)
Open these files in tabs/windows BEFORE the interview:
- `src/pipeline.py` - the orchestrator
- `src/retrieval/vector_store.py` - vector search
- `src/retrieval/reranker.py` - two-stage retrieval
- `src/generation/llm.py` - prompt engineering
- `compare_chunk_sizes.py` - the experiment runner

## NICE TO HAVE (Next 15 min)

### 4. Skim the speaker script
- File: `presentation\SPEAKER_SCRIPT.md`
- Read through Q&A prep section especially
- Don't memorize - know the structure

### 5. Mental rehearsal
- Run through your opening 30 seconds in your head
- Practice the 1-line pitch: "I built a RAG system for financial document Q&A with rigorous evaluation and a clear path to production"

## INTERVIEW DAY TIPS

### Opening
- Smile, eye contact, confident tone
- "Thanks for having me. Let me walk you through..."

### Slide pacing
- Don't rush, but don't linger
- ~1 min per slide on average
- Skip ahead if you're behind on time
- Slide 9 (results) and slide 10 (demo) are the highlights

### During the demo
- Talk while loading - don't let silence happen
- If something fails, stay calm: "Let me show you in mock mode" or "let me try another question"
- Have a backup demo question ready

### Q&A
- Pause before answering hard questions
- "That's a great question - let me think..."
- It's OK to say "I'd need to think about that more" 
- Refer to the actual code: "If we open `src/pipeline.py`..."
- Acknowledge limits honestly

### Trade-offs you should be ready to discuss
- ChromaDB vs Pinecone vs pgvector
- Bi-encoder vs cross-encoder (when to use each)
- Chunk size: smaller (precision) vs larger (context)
- Top-K retrieval (recall vs latency)
- Temperature (factual vs creative)
- Local vs API embeddings
- Fine-tuning vs RAG

## COMMON QUESTIONS - YOUR ANSWERS

**"Walk me through your code"**
> "Sure - the entry point is `RAGPipeline.query()` in `src/pipeline.py`. It calls vector_store.search to get top 20 candidates, then reranker.rerank to narrow to top 5, then llm.generate to synthesize an answer. Each component is dependency-injected so it's easy to swap."

**"How would you deploy this?"**
> "Streamlit on Cloud Run for the UI, ChromaDB → Vertex AI Vector Search for production, Cloud Logging for observability, Memorystore for caching. The architecture is already modular enough that this is mostly a deployment exercise, not a refactor."

**"What was the hardest part?"**
> "Honestly, the chunker bug. My initial code treated each PDF page as a single chunk regardless of chunk_size parameter. I only found it because my evaluation experiments showed identical results across configurations - that's when I dug in and fixed it. Without evaluation, I'd never have caught it."

**"What would you change if you started over?"**
> "Three things: (1) Build evaluation BEFORE the pipeline, not after - it informs every design decision. (2) Use a hybrid retrieval (semantic + BM25) from the start. (3) Plan for production observability from day one."

## YOU GOT THIS

Remember:
- You built this end-to-end. You know it better than they do.
- Senior engineers value HONESTY about trade-offs over false confidence
- They WANT you to succeed - they're invested in this interview going well
- Be curious about their problems, not just selling yourself
