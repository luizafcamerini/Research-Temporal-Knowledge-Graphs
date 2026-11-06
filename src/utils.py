"""Utilitários compartilhados: caminhos, I/O, formato LongMemEval e chamadas ao Gemini."""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

# --------------------------------------------------------------------------- caminhos

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = SRC_DIR / "results"

ORACLE_PATH = DATA_DIR / "longmemeval_oracle.json"
SUBSET_PATH = DATA_DIR / "subset.json"
INGEST_LOG_PATH = RESULTS_DIR / "ingest_log.jsonl"
ANSWERS_PATH = RESULTS_DIR / "respostas.jsonl"
JUDGMENTS_PATH = RESULTS_DIR / "julgamentos.jsonl"

QUESTION_TYPES = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "knowledge-update",
    "temporal-reasoning",
)
# Perguntas sem resposta (question_id terminando em "_abs") viram uma categoria à parte.
CATEGORIES = QUESTION_TYPES + ("abstention",)

T = TypeVar("T")
R = TypeVar("R")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- I/O

def read_json(path: Path) -> Any:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{path} não existe ou está vazio")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    """Lê um JSONL tolerando linhas vazias e uma última linha truncada (queda no meio da escrita)."""
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[aviso] {path.name}:{lineno} ignorada (JSON inválido)", file=sys.stderr)
    return records


def append_jsonl(path: Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def latest_by(records: Iterable[dict], key: Callable[[dict], Any]) -> dict[Any, dict]:
    """Indexa registros por chave; em duplicatas, vence o mais recente (último no arquivo)."""
    out: dict[Any, dict] = {}
    for r in records:
        out[key(r)] = r
    return out


# --------------------------------------------------------------------------- LongMemEval

_LME_DATE_RE = re.compile(r"(\d{4})/(\d{1,2})/(\d{1,2})\s*(?:\(\w+\))?\s*(\d{1,2}):(\d{2})")


def parse_lme_date(value: str) -> datetime:
    """Converte '2023/05/30 (Tue) 23:40' em datetime UTC (o Graphiti exige datetimes com fuso)."""
    m = _LME_DATE_RE.search(value)
    if not m:
        raise ValueError(f"Data em formato inesperado: {value!r}")
    y, mo, d, h, mi = map(int, m.groups())
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc)


def fmt_dt(dt: datetime | None) -> str | None:
    return dt.strftime("%Y-%m-%d %H:%M") if dt else None


def is_abstention(question_id: str) -> bool:
    return question_id.endswith("_abs")


def category_of(item: dict) -> str:
    return "abstention" if is_abstention(item["question_id"]) else item["question_type"]


def group_id_for(question_id: str) -> str:
    """Partição do Graphiti para uma pergunta (só aceita [A-Za-z0-9_-])."""
    return "lme_" + re.sub(r"[^A-Za-z0-9_-]", "_", question_id)


@dataclass
class Session:
    session_id: str
    date_str: str  # formato original, com dia da semana
    date: datetime
    turns: list[dict]


def sessions_of(item: dict) -> list[Session]:
    """Sessões do histórico da pergunta em ordem cronológica."""
    sessions = [
        Session(session_id=str(sid), date_str=date_str, date=parse_lme_date(date_str), turns=turns)
        for sid, date_str, turns in zip(
            item["haystack_session_ids"], item["haystack_dates"], item["haystack_sessions"], strict=True
        )
    ]
    return sorted(sessions, key=lambda s: s.date)  # sort estável: empates mantêm a ordem original


def format_turn(turn: dict) -> str:
    return f"{turn['role']}: {str(turn['content']).strip()}"


def split_rounds(turns: Sequence[dict]) -> list[list[dict]]:
    """Agrupa turnos em rodadas: uma fala do usuário + as respostas do assistente que a seguem."""
    rounds: list[list[dict]] = []
    for turn in turns:
        if turn["role"] == "user" or not rounds:
            rounds.append([turn])
        else:
            rounds[-1].append(turn)
    return rounds


GRANULARITIES = ("session", "round", "turn")


def chunk_session(session: Session, granularity: str) -> list[str]:
    """Quebra uma sessão em blocos de texto no formato 'role: conteúdo'."""
    if granularity == "session":
        return ["\n".join(format_turn(t) for t in session.turns)]
    if granularity == "round":
        return ["\n".join(format_turn(t) for t in r) for r in split_rounds(session.turns)]
    if granularity == "turn":
        return [format_turn(t) for t in session.turns]
    raise ValueError(f"Granularidade desconhecida: {granularity}")


# --------------------------------------------------------------------------- seleção de perguntas

def add_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ids", nargs="+", help="Processa só estes question_id")
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES, help="Filtra por categoria")
    parser.add_argument("--limit", type=int, help="Processa no máximo N perguntas")


def load_subset(args: argparse.Namespace | None = None) -> list[dict]:
    try:
        items = read_json(SUBSET_PATH)
    except FileNotFoundError:
        sys.exit(f"{SUBSET_PATH} vazio. Rode antes: python src/prepare_data.py")
    if args is not None:
        if getattr(args, "ids", None):
            wanted = set(args.ids)
            items = [it for it in items if it["question_id"] in wanted]
        if getattr(args, "categories", None):
            cats = set(args.categories)
            items = [it for it in items if category_of(it) in cats]
        if getattr(args, "limit", None):
            items = items[: args.limit]
    return items


# --------------------------------------------------------------------------- concorrência

async def gather_limited(
    items: Sequence[T], worker: Callable[[int, T], Awaitable[R]], limit: int
) -> list[R]:
    """Executa worker(i, item) para cada item com no máximo `limit` tarefas simultâneas."""
    sem = asyncio.Semaphore(max(1, limit))

    async def run(i: int, item: T) -> R:
        async with sem:
            return await worker(i, item)

    return await asyncio.gather(*(run(i, it) for i, it in enumerate(items)))


# --------------------------------------------------------------------------- Gemini

@dataclass
class LLMResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0


def _should_retry(exc: BaseException) -> bool:
    try:
        from google.genai import errors as genai_errors
    except ImportError:  # pragma: no cover
        return isinstance(exc, Exception)
    if isinstance(exc, genai_errors.ClientError):  # 4xx: só vale repetir timeout/rate limit
        return getattr(exc, "code", None) in (408, 429)
    if isinstance(exc, ValueError):  # erro determinístico (ex.: nº de embeddings errado)
        return False
    return isinstance(exc, Exception)  # 5xx, rede...


def _log_retry(state: RetryCallState) -> None:
    exc = state.outcome.exception() if state.outcome else None
    wait = state.next_action.sleep if state.next_action else 0
    msg = str(exc).splitlines()[0][:160] if exc else "?"
    print(f"  [retry] tentativa {state.attempt_number} falhou ({type(exc).__name__}: {msg}); "
          f"aguardando {wait:.0f}s", file=sys.stderr)


gemini_retry = retry(
    retry=retry_if_exception(_should_retry),
    wait=wait_random_exponential(multiplier=2, max=90),
    stop=stop_after_attempt(8),
    before_sleep=_log_retry,
    reraise=True,
)


@gemini_retry
async def generate_text(
    client, model: str, prompt: str, *, temperature: float = 0.0, system_instruction: str | None = None
) -> LLMResult:
    from google.genai import types

    t0 = time.perf_counter()
    resp = await client.aio.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=temperature, system_instruction=system_instruction),
    )
    usage = resp.usage_metadata
    input_tokens = (getattr(usage, "prompt_token_count", None) or 0) if usage else 0
    output_tokens = 0
    if usage:  # modelos "thinking" cobram os tokens de raciocínio como saída
        output_tokens = (getattr(usage, "candidates_token_count", None) or 0) + (
            getattr(usage, "thoughts_token_count", None) or 0
        )
    return LLMResult(
        text=(resp.text or "").strip(),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        seconds=time.perf_counter() - t0,
    )


# ~2k tokens: limite de entrada dos modelos de embedding do Gemini.
MAX_EMBED_CHARS = 8000


@gemini_retry
async def _embed_batch(client, model: str, texts: list[str], dim: int, task_type: str | None) -> list[list[float]]:
    from google.genai import types

    # Um Content por texto: no gemini-embedding-2 (multimodal), uma lista de strings
    # soltas vira um único embedding agregado.
    resp = await client.aio.models.embed_content(
        model=model,
        contents=[types.Content(parts=[types.Part(text=t[:MAX_EMBED_CHARS])]) for t in texts],
        config=types.EmbedContentConfig(output_dimensionality=dim, task_type=task_type),
    )
    vectors = [list(e.values or []) for e in (resp.embeddings or [])]
    if len(vectors) != len(texts) or any(len(v) != dim for v in vectors):
        raise ValueError(f"Embeddings inesperados: {len(vectors)} vetores para {len(texts)} textos")
    return vectors


async def embed_texts(
    client, model: str, texts: Sequence[str], *, dim: int, task_type: str | None = None, batch_size: int = 16
) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        vectors.extend(await _embed_batch(client, model, list(texts[start:start + batch_size]), dim, task_type))
    return vectors
