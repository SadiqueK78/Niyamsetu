# backend/config.py
# ─────────────────────────────────────────────────────────
# Central configuration for Niyamsetu backend.
# All settings are loaded from the .env file.
# Every other module imports from here — never hardcode values elsewhere.
# ─────────────────────────────────────────────────────────

from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    # ── MongoDB ───────────────────────────────────────────
    # Connection string from Atlas (stored in .env, never hardcoded)
    MONGODB_URL: str
    # Name of the database inside your Atlas cluster
    DATABASE_NAME: str = "niyamsetu"

    # ── JWT Authentication ────────────────────────────────
    # Secret key used to sign tokens — anyone with this can forge tokens
    # So it stays in .env and never touches GitHub
    JWT_SECRET: str
    # How many hours before a login token expires (user gets logged out)
    JWT_EXPIRE_HOURS: int = 24

    # ── Ollama (local LLM) ────────────────────────────────
    # Where Ollama is running — always localhost for on-premise
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    # Model used to generate embeddings (converts text to vectors)
    EMBEDDING_MODEL: str = "bge-m3"
    # Model used to generate answers (the actual LLM)
    LLM_MODEL: str = "qwen2.5:7b"
    # Hard ceiling on a single LLM call, in seconds.
    # Without this the Ollama client defaults to timeout=None — an unbounded
    # wait. A client that gave up long ago leaves the generation running, and
    # because Ollama serialises on the model, that orphan blocks every later
    # request until it finishes on its own.
    # Set just above the harness's 600s per-question timeout so the caller
    # still governs normal runs and this only fires as a backstop.
    LLM_TIMEOUT: float = 620.0
    # Context window pinned explicitly, in tokens.
    # Left unset, Ollama sizes the context PER REQUEST from the prompt length
    # and restarts llama-server whenever the tier changes (observed: -c 512,
    # -c 2048, -c 4096 for the same model). That restart is a fresh CPU
    # allocation of the weights (~2.3 GiB, mmap is off on the CPU path) plus
    # the KV cache, and on a loaded machine it fails outright:
    #   ggml_backend_cpu_buffer_type_alloc_buffer: failed to allocate buffer
    #   graph_reserve: failed to allocate compute buffers      -> HTTP 500
    # Pinning removes the resize, so the model loads once and stays put.
    # 8192 (not 4096) because the chat path never truncates: TOP_K=15 chunks
    # of CHUNK_SIZE=800 plus source headers and CONTEXT_WINDOW=6 history turns
    # reach ~16k characters, which is denser than 4096 tokens in Devanagari.
    # Ollama runs with --context-shift, so a prompt that overflows a pinned
    # context is NOT an error — it silently drops the front of the prompt,
    # i.e. retrieved chunks, while citations still reference them.
    LLM_NUM_CTX: int = 8192
    # How long Ollama keeps the model resident after a request.
    # Ollama's default is 5m, so any normal pause between questions unloads
    # the model and the next question pays a full reload — another chance to
    # hit the allocation failure above. 30m spans idle gaps in a chat session.
    LLM_KEEP_ALIVE: str = "30m"

    # ── Remote LLM Provider (e.g., Google Colab) ──────────
    # URL for the remote inference API (e.g., https://your-colab-url.ngrok.io)
    # Leave empty to use local Ollama (default behavior)
    COLAB_API_URL: str = ""
    # API key for the remote inference API
    # Format: X-API-Key: <secret> header
    COLAB_API_KEY: str = ""
    # Hard ceiling on a single remote LLM call, in seconds.
    # Without this, requests could hang indefinitely.
    COLAB_LLM_TIMEOUT: float = 60.0

    # ── RAG settings ──────────────────────────────────────
    # How many characters per chunk when splitting PDFs
    CHUNK_SIZE: int = 800
    # How many characters overlap between chunks (prevents cutting mid-sentence)
    CHUNK_OVERLAP: int = 150
    # How many chunks to retrieve per query
    TOP_K: int = 12
    # How many previous chat turns to include for context
    CONTEXT_WINDOW: int = 6

    # ── Retrieval and answer experiments ──────────────────
    # Every flag in this section defaults to the behaviour that existed
    # before it, so a deployment that sets none of them runs identically
    # to one that has never heard of them.

    # Narrow the context to the few source files that dominate the
    # ranking, instead of sending every retrieved chunk to the LLM.
    ROUTE_ENABLED: bool = False
    # How many source files survive routing.
    ROUTE_TOP_DOCS: int = 3
    # How deep to retrieve before routing scores the files. Larger than
    # TOP_K on purpose — routing needs a pool to measure, not a shortlist.
    ROUTE_POOL_K: int = 30
    # Ceiling on how many chunks routing passes through.
    ROUTE_MAX_CHUNKS: int = 8

    # Prepend each kept file's header chunk, so the LLM sees the GR
    # number, date and department that body chunks leave out.
    HEADER_INJECT: bool = False

    # Ask the LLM for synonym phrasings of the question and retrieve on
    # all of them. Costs one extra LLM call per question.
    QUERY_REWRITE: bool = False
    # Hard ceiling on that rewrite call, in seconds. On timeout the
    # original question is used and nothing else changes.
    QUERY_REWRITE_TIMEOUT: float = 25.0

    # Second LLM pass that checks the generated answer against the same
    # context block, and refuses rather than answer unsupported claims.
    ANSWER_VERIFY: bool = False
    # Hard ceiling on that verification call, in seconds. On timeout the
    # original answer is kept.
    VERIFY_TIMEOUT: float = 40.0

    # Pull in chunks from documents the retrieved ones are linked to in the
    # GR relationship graph — the superseding order, the amendment — which
    # keyword and vector search both miss because they share no wording
    # with the question.
    GRAPH_EXPAND: bool = False
    # Number of top-ranked distinct files whose graph neighbours may be added.
    GRAPH_SEED_DOCS: int = 2
    # How many linked documents may contribute chunks to one answer.
    GRAPH_MAX_NEIGHBOR_DOCS: int = 2
    # How many chunks are taken from each of them.
    GRAPH_CHUNKS_PER_NEIGHBOR: int = 3

    # Answer header questions — the date, the department, the GR number —
    # straight from the extracted metadata table, with no retrieval and no
    # LLM call. Only fires on questions that name one document and ask for
    # one of those three fields; everything else takes the normal path.
    META_FASTPATH: bool = False

    # For a query written in Latin script, skip BM25 and return the dense
    # ranking alone. The corpus is almost entirely Devanagari OCR, so an
    # English query's terms rarely occur in it and the lexical side adds
    # rank noise rather than evidence. Devanagari and mixed queries are
    # fused exactly as before.
    SCRIPT_AWARE_FUSION: bool = False

    # Document summaries via map-reduce over the indexed chunks, with a
    # numeric faithfulness check and one repair pass (core/summary_v2.py).
    # Off keeps the original single-prompt summarizer.
    SUMMARY_V2_ENABLED: bool = False
    # Characters of chunk text per MAP call.
    SUMMARY_BATCH_CHARS: int = 5000
    # Ceiling on MAP calls per document; excess text joins the last batch.
    SUMMARY_MAX_BATCHES: int = 12

    # Condense a follow-up and the recent chat history into one standalone
    # question, used ONLY for retrieval (core/condense.py). The prompt and
    # the logged question are unchanged. One extra LLM call per follow-up.
    HISTORY_REWRITE: bool = False
    # How many recent exchanges (user + assistant) the condenser reads.
    HISTORY_REWRITE_TURNS: int = 2
    # Ceiling on the condense call, in seconds; on timeout the original
    # question is used.
    HISTORY_REWRITE_TIMEOUT: int = 60

    # Output token ceiling for chat answer generation only (a copy of
    # core.rag.get_llm()'s client with num_predict set; summary, condense
    # and rewrite calls are unaffected). 0 or less means no ceiling, the
    # behaviour before this setting existed.
    LLM_NUM_PREDICT: int = 800

    # Cross-encoder reranking of retrieved chunks (core/reranker.py): the
    # hybrid search returns RERANK_CANDIDATES chunks, the cross-encoder
    # scores them against the user's question, and the best RERANK_TOP_N
    # become the prompt context and the citations.
    RERANK_ENABLED: bool = False
    RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANK_CANDIDATES: int = 20
    RERANK_TOP_N: int = 5

    # ── CORS ──────────────────────────────────────────────
    # Browser origins allowed to call this API, comma-separated.
    #
    # This replaces a hardcoded ["*"]. A wildcard here is dangerous
    # specifically BECAUSE allow_credentials stays True: Starlette then
    # reflects whatever Origin the caller sent back in the preflight
    # response, so every cross-site preflight succeeds no matter who
    # sent it. Today this app carries its JWT in an Authorization header
    # from localStorage, which a browser will not attach cross-origin on
    # its own — so the wildcard is not directly exploitable yet. It
    # becomes exploitable the moment anything moves to a cookie, and it
    # already lets any site read every unauthenticated endpoint.
    #
    # Default is the Vite dev server, named explicitly rather than "*".
    # Note the dev frontend normally reaches the API through Vite's /api
    # proxy, which is same-origin and never triggers CORS at all — so
    # this default exists for direct calls (Swagger at /docs, a curl from
    # a browser tab, a build served without the proxy).
    #
    # Deployments set their real front-end origin, e.g.
    #     CORS_ALLOWED_ORIGINS=https://niyamsetu.example.gov.in
    # Multiple origins are comma-separated. Do not put "*" here.
    CORS_ALLOWED_ORIGINS: str = "http://localhost:5173"

    # ── Default user seeding ──────────────────────────────
    # Whether seed_default_users() is allowed to run at startup.
    #
    # That function creates admin/admin123 and user1/user123 when the users
    # collection is empty. Both are written in the source, so anyone who
    # reads the repository knows the credentials of any deployment that
    # started from an empty database — a public demo, a fresh staging box,
    # or a reviewer's clone.
    #
    # Defaults to True so a developer who has never heard of this setting
    # gets exactly the behaviour that existed before it: clone, run, log in.
    # Any deployment that is not someone's laptop should set
    #     SEED_DEFAULT_USERS=false
    # in its .env, create a real admin account, and never rely on the seeds.
    #
    # Turning it off does NOT remove accounts that were already seeded — it
    # only stops new ones being created.
    SEED_DEFAULT_USERS: bool = True

    # ── Local file storage ────────────────────────────────
    # Base data directory — relative to backend/ folder
    DATA_DIR: str = "./data"

    # ── Computed paths (derived from DATA_DIR) ────────────
    # These are properties, not .env variables
    # They give you Path objects you can use directly in code
    @property
    def CORS_ORIGINS(self) -> list[str]:
        """
        CORS_ALLOWED_ORIGINS split into the list CORSMiddleware wants.

        Kept as a comma-separated string in .env rather than a typed
        list: pydantic parses a list-typed setting as JSON, which would
        make the .env line ["https://a","https://b"] instead of plain
        text, and get quietly rejected when someone writes it the
        obvious way. Blank entries are dropped so a trailing comma is
        harmless.
        """
        return [o.strip() for o in self.CORS_ALLOWED_ORIGINS.split(",") if o.strip()]

    @property
    def GRDOCS_PATH(self) -> Path:
        # Where uploaded GR PDFs are stored
        return Path(self.DATA_DIR) / "grdocs"

    @property
    def VECTORSTORE_PATH(self) -> Path:
        # Where FAISS index files are saved
        return Path(self.DATA_DIR) / "vectorstore"

    @property
    def SUMMARIES_PATH(self) -> Path:
        # Where generated summary files are saved
        return Path(self.DATA_DIR) / "summaries"

    # pydantic-settings v2 configuration. Replaces the class-based
    # `class Config`, which pydantic v2 deprecated.
    model_config = SettingsConfigDict(
        # Tells pydantic to read values from the .env file automatically
        env_file=".env",
        # If a variable is in .env but not defined above, ignore it
        extra="ignore",
    )


# ── Single instance used across the entire app ────────────
# Every file does: from config import settings
# Then uses: settings.LLM_MODEL, settings.MONGODB_URL etc.
settings = Settings()

# ── Ensure data directories exist on startup ──────────────
# Creates the folders if they don't exist yet
# exist_ok=True means no error if folder already exists
settings.GRDOCS_PATH.mkdir(parents=True, exist_ok=True)
settings.VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)
settings.SUMMARIES_PATH.mkdir(parents=True, exist_ok=True)