# backend/core/condense.py
# ─────────────────────────────────────────────────────────
# Follow-up condensation for retrieval.
#
# Retrieval searches the question it is given and nothing else, so a
# follow-up like "त्यासाठी किती खर्च मंजूर झाला?" or "And who signed it?"
# retrieves on words that name no document. The chat history reaches only
# the prompt, by which point the wrong chunks have already been chosen.
#
# condense_query() asks the LLM to rewrite the follow-up, using the last
# few exchanges, as one standalone question. The result is used ONLY as
# the retrieval query (settings.HISTORY_REWRITE, see core/rag.py query()).
# The prompt, the history shown to the model and the logged question all
# stay the user's own words, so a bad rewrite can cost recall but can
# never change what the user is answered about or what is recorded.
#
# Every failure — an exception, a timeout, an empty or runaway output —
# falls back to the original question.
# ─────────────────────────────────────────────────────────

import sys
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

CONDENSE_PROMPT = (
    "Given the conversation below and the follow-up question, rewrite the follow-up "
    "as one standalone question that can be understood without the conversation. "
    "Keep the same language as the follow-up question. Include the specific document "
    "subject, names and numbers from the conversation that the follow-up refers to. "
    "If the follow-up is already standalone, return it unchanged. Return only the "
    "question.\n\nCONVERSATION:\n{history}\n\nFOLLOW-UP: {question}\n\nSTANDALONE QUESTION:"
)

# Assistant turns are cut to this many characters in the condense prompt.
ASSISTANT_CHARS = 300
# A rewrite longer than this is not a question; the original is used.
MAX_OUTPUT_CHARS = 400
# Output token ceiling for the condense call.
NUM_PREDICT = 100

# Quote characters stripped from both ends of the rewrite.
_QUOTES = "\"'`“”‘’«»"


def format_history(history: list, turns: int) -> str:
    """
    The last `turns` exchanges as "User: ..." / "Assistant: ..." lines.

    An exchange is a user turn plus an assistant turn, so this keeps the
    last 2 * turns messages. Assistant turns are cut to ASSISTANT_CHARS.
    """
    recent = (history or [])[-2 * turns:] if turns > 0 else []
    lines = []
    for turn in recent:
        content = (turn.get("content") or "").strip()
        if turn.get("role") == "user":
            lines.append(f"User: {content}")
        else:
            lines.append(f"Assistant: {content[:ASSISTANT_CHARS]}")
    return "\n".join(lines)


def build_prompt(question: str, history: list, turns: int) -> str:
    return CONDENSE_PROMPT.format(history=format_history(history, turns), question=question)


def extract_question(output: str) -> str:
    """First non-empty line of the output, whitespace and quotes stripped."""
    for line in (output or "").splitlines():
        line = line.strip().strip(_QUOTES).strip()
        if line:
            return line
    return ""


def _generate(prompt: str) -> str:
    """
    One call through the chat path's own helper, core.rag.get_llm()
    (temperature 0, same model, num_ctx, keep_alive and client timeout),
    called as the chat path calls it.

    The output ceiling is set on a copy of that helper's client
    (model_copy with num_predict) rather than per call: langchain-ollama's
    invoke() has no per-call token limit, and passing an options dict
    would replace every other option the helper sets.
    """
    from langchain_core.messages import HumanMessage
    from core.rag import get_llm

    llm = get_llm().model_copy(update={"num_predict": NUM_PREDICT})
    response = llm.invoke([HumanMessage(content=prompt)], stream=False)
    return response.content or ""


def condense_query(question: str, history: list, language: str | None = None) -> str:
    """
    A standalone version of `question` for retrieval.

    Args:
        question : the follow-up as the user typed it
        history  : previous {"role", "content"} turns, oldest first
        language : the question's language. Accepted for the caller's
                   convenience; the prompt itself tells the model to keep
                   the follow-up's language, so it is not used here.

    Returns:
        the condensed question, or `question` unchanged when history is
        empty, or when the call raises, times out after
        HISTORY_REWRITE_TIMEOUT seconds, returns nothing, or returns more
        than MAX_OUTPUT_CHARS characters.

    Synchronous; the caller runs it on a worker thread. The timeout bounds
    how long the caller waits. A hung call cannot be killed from here, so
    it is abandoned, and the client's own LLM_TIMEOUT still bounds it.
    """
    if not history:
        return question

    prompt = build_prompt(question, history, settings.HISTORY_REWRITE_TURNS)
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        output = pool.submit(_generate, prompt).result(timeout=settings.HISTORY_REWRITE_TIMEOUT)
    except (FutureTimeout, Exception):   # timed out, or the call raised
        return question
    finally:
        pool.shutdown(wait=False)

    condensed = extract_question(output)
    if not condensed or len(condensed) > MAX_OUTPUT_CHARS:
        return question
    return condensed
