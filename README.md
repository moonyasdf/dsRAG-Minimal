# dsRAG-Minimalist

*The original creators of [dsRAG](https://github.com/D-Star-AI/dsRAG) are Zach and Nick McCormick.*

## What is dsRAG?
dsRAG is a retrieval engine for unstructured data. It is especially good at handling challenging queries over dense text, like financial reports, legal documents, and academic papers. dsRAG achieves substantially higher accuracy than vanilla RAG baselines on complex open-book question answering tasks. On one especially challenging benchmark, [FinanceBench](https://arxiv.org/abs/2311.11944), dsRAG gets accurate answers 96.6% of the time, compared to the vanilla RAG baseline which only gets 32% of questions correct.

There are three key methods used to improve performance over vanilla RAG systems:
1. Semantic sectioning
2. AutoContext
3. Relevant Segment Extraction (RSE)

#### Semantic sectioning
Semantic sectioning uses an LLM to break a document into sections. It works by annotating the document with line numbers and then prompting an LLM to identify the starting and ending lines for each “semantically cohesive section.” These sections should be anywhere from a few paragraphs to a few pages long. The sections then get broken into smaller chunks if needed. The LLM is also prompted to generate descriptive titles for each section. These section titles get used in the contextual chunk headers created by AutoContext, which provides additional context to the ranking models (embeddings and reranker), enabling better retrieval.

#### AutoContext (contextual chunk headers)
AutoContext creates contextual chunk headers that contain document-level and section-level context, and prepends those chunk headers to the chunks prior to embedding them. This gives the embeddings a much more accurate and complete representation of the content and meaning of the text. In our testing, this feature leads to a dramatic improvement in retrieval quality. In addition to increasing the rate at which the correct information is retrieved, AutoContext also substantially reduces the rate at which irrelevant results show up in the search results. This reduces the rate at which the LLM misinterprets a piece of text in downstream chat and generation applications.

#### Relevant Segment Extraction
Relevant Segment Extraction (RSE) is a query-time post-processing step that takes clusters of relevant chunks and intelligently combines them into longer sections of text that we call segments. These segments provide better context to the LLM than any individual chunk can. For simple factual questions, the answer is usually contained in a single chunk; but for more complex questions, the answer usually spans a longer section of text. The goal of RSE is to intelligently identify the section(s) of text that provide the most relevant information, without being constrained to fixed length chunks.

For example, suppose you have a bunch of SEC filings in a knowledge base and you ask "What were Apple's key financial results in the most recent fiscal year?" RSE will identify the most relevant segment as the entire "Consolidated Statement of Operations" section, which will be 5-10 chunks long. Whereas if you ask "Who is Apple's CEO?" the most relevant segment will be identified as a single chunk that mentions "Tim Cook, CEO."

---

## What's New in dsRAG-Minimalist?

This version represents a significant refactoring of the original dsRAG library, guided by the following principles:

1.  **Minimalism:** Drastically reduced codebase focusing *only* on core RAG functionality. Removed features not essential for the main retrieval and generation pipeline.
2.  **Performance:** Optimized critical code paths for speed, particularly document ingestion and query processing (< 3s target for context preparation). Readability was sometimes sacrificed for performance, compensated by extensive comments (`#`).
3.  **Local-First Focus:** Prioritized and streamlined local processing using tools like Ollama, SentenceTransformers, Jina AI, SQLite, and local file storage.
4.  **Flexibility via Injection:** While local-first, the design allows users to *inject* their preferred models (local or API-based) during runtime initialization or function calls.
5.  **Clear Structure & Documentation:** Maintained a clear internal structure and added comprehensive comments to aid understanding by both humans and LLMs, despite potential code complexity.
6.  **Externalized Prompts:** Moved all internal LLM prompts to external `.txt` files in the `/prompts` directory for easier inspection and customization.

## Key Changes & Simplifications

*   **Databases:**
    *   **Vector DB:** Hardcoded to use **Qdrant** exclusively. All support for Chroma, Pinecone, FAISS (within BasicVectorDB), Weaviate, Milvus, Postgres has been removed.
    *   **Chunk DB:** Hardcoded to use **SQLite** exclusively. Support for BasicChunkDB, Postgres, DynamoDB has been removed.
    *   **Chat History DB:** Hardcoded to use **SQLite** exclusively. Support for BasicChatThreadDB has been removed.
*   **File Parsing (dsparse):**
    *   **No VLM:** All Vision Language Model (VLM) parsing capabilities have been **removed**. Parsing relies solely on text extraction from PDFs (PyPDF2), DOCX (docx2txt), TXT, and MD files.
    *   **Simplified `parse_and_chunk`:** The main function no longer accepts `file_parsing_config` related to VLM.
    *   **Semantic Sectioning:** Still supported, but now explicitly requires an `LLM` instance to be passed via `semantic_sectioning_config`. Examples use an injected Ollama model.
*   **File System:**
    *   Hardcoded to use **`LocalFileSystem`**. Support for S3FileSystem has been removed.
*   **Model Integration:**
    *   **Injection:** `KnowledgeBase`, `get_chat_thread_response`, etc., now rely on users *injecting* instances of `Embedding`, `LLM`, and `Reranker` classes. No default API clients are instantiated internally.
    *   **Local Model Examples:** Examples (`01_create_kb_local.py`, `03_query_kb.py`, `04_chat_basic.py`) demonstrate using Ollama, `sentence-transformers`, and `jina-reranker` for a fully local setup. Specific models requested are used (`intfloat/multilingual-e5-large-instruct`, `jina-reranker-v2-base-multilingual`, `gemma:9b` via Ollama).
    *   **API Flexibility:** Users can still use APIs (OpenAI, Anthropic, OpenRouter, LM Studio, etc.) by creating or using compatible `LLM` or `Embedding` wrapper classes and injecting those instances. Example `04_chat_basic.py` includes commented-out sections for LM Studio/OpenRouter.
*   **Code Structure:** Refactored into clearer sub-packages (`core`, `database`, `chat`, `dsparse`, `utils`).
*   **External Prompts:** All LLM prompts are loaded from `.txt` files in `/prompts`.

## Limitations

*   **No VLM Parsing:** Cannot process image-only PDFs or leverage visual understanding for complex layouts, tables, or figures. Relies purely on text extraction.
*   **Single Database Backend:** Tied to Qdrant for vectors and SQLite for chunks/chat history.
*   **Single File System:** Only supports local file storage.
*   **Reduced Configuration:** Some advanced or less common configuration options from the original dsRAG might have been removed for simplicity.

## Capabilities Preserved (and Enhanced)

*   **Core RAG Logic:** Semantic Sectioning, AutoContext, and Relevant Segment Extraction (RSE) remain central.
*   **Local Processing:** Optimized for running entirely on local hardware using Ollama and local embedding/reranking models.
*   **Model Flexibility:** Easily swap local or API models via dependency injection.
*   **Chat Functionality:** Includes chat history management and retrieval-augmented response generation.
*   **Performance:** Designed for significantly faster document processing and query preparation compared to setups relying heavily on external APIs for every step.

## Performance Focus

A primary goal was to achieve high performance, targeting sub-3-second context preparation for queries. This involved:
*   Using efficient local databases (Qdrant, SQLite).
*   Optimizing data handling and processing steps.
*   Potentially sacrificing some code readability for speed, offset by detailed comments (`#`).
Actual performance depends heavily on hardware, the specific local models used, and KB size.
