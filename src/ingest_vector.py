"""Baseline de RAG vetorial: indexa o histórico de cada pergunta no Postgres (pgvector).

Usa o MESMO modelo e dimensão de embedding do Graphiti, para que a comparação isole o
efeito da representação (grafo temporal x trechos de conversa).

Exemplos:
    python src/ingest_vector.py
    python src/ingest_vector.py --granularity session
    python src/ingest_vector.py --reset            # apaga a tabela inteira antes
"""
import argparse
import asyncio
import sys
import time

from config import EMBEDDING_DIM, GEMINI_EMBED_MODEL, make_genai_client, pg_connect
from utils import (
    INGEST_LOG_PATH,
    add_selection_args,
    append_jsonl,
    category_of,
    chunk_session,
    embed_texts,
    load_subset,
    now_iso,
    sessions_of,
)

BACKEND = "vector"
TABLE = "lme_chunks"
VECTOR_GRANULARITIES = ("round", "session", "turn")

# Sem índice HNSW de propósito: toda busca filtra por question_id (dezenas de trechos),
# então a varredura exata via índice B-tree é mais rápida e não perde resultados.
SCHEMA_SQL = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS {TABLE} (
    id               bigserial PRIMARY KEY,
    question_id      text        NOT NULL,
    session_id       text        NOT NULL,
    session_date     timestamptz NOT NULL,
    session_date_str text        NOT NULL,
    chunk_idx        int         NOT NULL,
    granularity      text        NOT NULL,
    content          text        NOT NULL,
    embedding        vector({{dim}}) NOT NULL,
    UNIQUE (question_id, session_id, chunk_idx)
);
CREATE INDEX IF NOT EXISTS {TABLE}_question_idx ON {TABLE} (question_id);
"""


def ensure_schema(conn, dim: int = EMBEDDING_DIM) -> None:
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL.format(dim=dim))
        cur.execute(
            """
            SELECT format_type(a.atttypid, a.atttypmod)
            FROM pg_attribute a
            WHERE a.attrelid = %s::regclass AND a.attname = 'embedding'
            """,
            (TABLE,),
        )
        row = cur.fetchone()
    if row and row[0] != f"vector({dim})":
        sys.exit(f"{TABLE}.embedding é {row[0]}, mas EMBEDDING_DIM={dim}. Rode com --reset para recriar a tabela.")


def to_pgvector(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in vec) + "]"


def search_chunks(conn, question_id: str, query_vec: list[float], k: int) -> list[dict]:
    """Top-k trechos da pergunta por similaridade de cosseno."""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT session_id, session_date, session_date_str, chunk_idx, content,
                   1 - (embedding <=> %s::vector) AS score
            FROM {TABLE}
            WHERE question_id = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (to_pgvector(query_vec), question_id, to_pgvector(query_vec), k),
        )
        cols = [c.name for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def build_chunks(item: dict, granularity: str) -> list[dict]:
    chunks = []
    for session in sessions_of(item):
        for idx, content in enumerate(chunk_session(session, granularity)):
            if content.strip():
                chunks.append({
                    "session_id": session.session_id,
                    "session_date": session.date,
                    "session_date_str": session.date_str,
                    "chunk_idx": idx,
                    "content": content,
                })
    return chunks


async def ingest_question(conn, client, item: dict, args: argparse.Namespace) -> dict:
    qid = item["question_id"]
    chunks = build_chunks(item, args.granularity)
    record = {
        "backend": BACKEND,
        "question_id": qid,
        "question_type": item["question_type"],
        "category": category_of(item),
        "granularity": args.granularity,
        "model": GEMINI_EMBED_MODEL,
        "n_sessions": len(item["haystack_sessions"]),
        "n_chunks": len(chunks),
    }
    t0 = time.perf_counter()
    try:
        vectors = await embed_texts(
            client, GEMINI_EMBED_MODEL, [c["content"] for c in chunks],
            dim=EMBEDDING_DIM, task_type=args.task_type, batch_size=args.batch_size,
        )
        rows = [
            (qid, c["session_id"], c["session_date"], c["session_date_str"], c["chunk_idx"],
             args.granularity, c["content"], to_pgvector(v))
            for c, v in zip(chunks, vectors, strict=True)
        ]
        with conn.transaction(), conn.cursor() as cur:
            cur.execute(f"DELETE FROM {TABLE} WHERE question_id = %s", (qid,))
            cur.executemany(
                f"""
                INSERT INTO {TABLE} (question_id, session_id, session_date, session_date_str,
                                     chunk_idx, granularity, content, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
                """,
                rows,
            )
        record["status"] = "ok"
    except Exception as e:  # noqa: BLE001
        record["status"] = "error"
        record["error"] = f"{type(e).__name__}: {e}"[:2000]
    record["seconds"] = round(time.perf_counter() - t0, 2)
    record["timestamp"] = now_iso()
    append_jsonl(INGEST_LOG_PATH, record)
    return record


def already_indexed(conn, granularity: str) -> dict[str, int]:
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT question_id, count(*) FROM {TABLE} WHERE granularity = %s GROUP BY question_id",
            (granularity,),
        )
        return dict(cur.fetchall())


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_selection_args(p)
    p.add_argument("--granularity", choices=VECTOR_GRANULARITIES, default="round",
                   help="Unidade de cada trecho indexado (padrão: round = pergunta do usuário + resposta)")
    p.add_argument("--batch-size", type=int, default=16, help="Textos por chamada de embedding")
    p.add_argument("--task-type", default=None,
                   help="task_type do Gemini (ex.: RETRIEVAL_DOCUMENT). Padrão: nenhum, igual ao Graphiti")
    p.add_argument("--force", action="store_true", help="Reindexa mesmo as perguntas já completas")
    p.add_argument("--reset", action="store_true", help=f"Apaga a tabela {TABLE} inteira antes de começar")
    args = p.parse_args()

    items = load_subset(args)
    client = make_genai_client()
    with pg_connect() as conn:
        if args.reset:
            conn.execute(f"DROP TABLE IF EXISTS {TABLE}")
            print(f"Tabela {TABLE} apagada")
        ensure_schema(conn)

        existing = already_indexed(conn, args.granularity)
        pending = [
            it for it in items
            if args.force or existing.get(it["question_id"]) != len(build_chunks(it, args.granularity))
        ]
        print(f"{len(items)} perguntas selecionadas | {len(items) - len(pending)} já indexadas | {len(pending)} a indexar")

        errors = 0
        for i, item in enumerate(pending, 1):
            rec = await ingest_question(conn, client, item, args)
            label = f"[{i}/{len(pending)}] {item['question_id']}"
            if rec["status"] == "ok":
                print(f"{label} ok: {rec['n_chunks']} trechos em {rec['seconds']:.1f}s")
            else:
                errors += 1
                print(f"{label} ERRO: {rec['error'][:300]}", file=sys.stderr)

    if errors:
        print(f"{errors} perguntas com erro; rode de novo para reprocessar.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
