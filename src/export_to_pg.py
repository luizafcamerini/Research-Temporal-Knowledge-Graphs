"""Exporta os resultados da POC para o Postgres (Supabase), para análise em SQL.

Tabelas (upsert, pode rodar quantas vezes quiser):
    lme_respostas    respostas.jsonl   (última versão por question_id + system)
    lme_julgamentos  julgamentos.jsonl (idem)
    lme_ingest_log   ingest_log.jsonl  (última execução por backend + question_id)
    lme_graph_facts  arestas RELATES_TO do Neo4j, com validade temporal (só com --graph)
    lme_accuracy     view com a acurácia por sistema e categoria

Exemplos:
    python src/export_to_pg.py
    python src/export_to_pg.py --graph
"""
import argparse
import asyncio

from psycopg.types.json import Jsonb

from config import pg_connect
from utils import ANSWERS_PATH, INGEST_LOG_PATH, JUDGMENTS_PATH, latest_by, read_jsonl

RESULTS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS lme_respostas (
    question_id   text NOT NULL,
    system        text NOT NULL,
    question_type text,
    category      text,
    abstention    boolean,
    question      text,
    question_date text,
    gold_answer   text,
    hypothesis    text,
    model         text,
    retrieval_s   double precision,
    generation_s  double precision,
    input_tokens  int,
    output_tokens int,
    context_chars int,
    created_at    timestamptz,
    data          jsonb,
    PRIMARY KEY (question_id, system)
);
CREATE TABLE IF NOT EXISTS lme_julgamentos (
    question_id   text NOT NULL,
    system        text NOT NULL,
    question_type text,
    category      text,
    abstention    boolean,
    label         boolean,
    judge_model   text,
    judge_raw     text,
    created_at    timestamptz,
    PRIMARY KEY (question_id, system)
);
CREATE TABLE IF NOT EXISTS lme_ingest_log (
    backend       text NOT NULL,
    question_id   text NOT NULL,
    question_type text,
    category      text,
    granularity   text,
    status        text,
    seconds       double precision,
    input_tokens  int,
    output_tokens int,
    created_at    timestamptz,
    data          jsonb,
    PRIMARY KEY (backend, question_id)
);
CREATE TABLE IF NOT EXISTS lme_graph_facts (
    uuid        text PRIMARY KEY,
    group_id    text NOT NULL,
    source      text,
    relation    text,
    target      text,
    fact        text,
    valid_at    timestamptz,
    invalid_at  timestamptz,
    created_at  timestamptz,
    expired_at  timestamptz,
    episodes    text[]
);
CREATE INDEX IF NOT EXISTS lme_graph_facts_group_idx ON lme_graph_facts (group_id);
CREATE OR REPLACE VIEW lme_accuracy AS
SELECT system, category, count(*) AS n, sum(label::int) AS correct, round(avg(label::int), 4) AS accuracy
FROM lme_julgamentos
GROUP BY system, category;
"""


def ensure_results_schema(conn) -> None:
    conn.execute(RESULTS_SCHEMA_SQL)


def export_answers(conn) -> int:
    rows = [
        (r["question_id"], r["system"], r.get("question_type"), r.get("category"), r.get("abstention"),
         r.get("question"), r.get("question_date"), r.get("gold_answer"), r.get("hypothesis"), r.get("model"),
         r.get("retrieval_s"), r.get("generation_s"), r.get("input_tokens"), r.get("output_tokens"),
         r.get("context_chars"), r.get("timestamp"), Jsonb(r))
        for r in latest_by(read_jsonl(ANSWERS_PATH), key=lambda r: (r["question_id"], r["system"])).values()
    ]
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO lme_respostas VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (question_id, system) DO UPDATE SET
                question_type = EXCLUDED.question_type, category = EXCLUDED.category,
                abstention = EXCLUDED.abstention, question = EXCLUDED.question,
                question_date = EXCLUDED.question_date, gold_answer = EXCLUDED.gold_answer,
                hypothesis = EXCLUDED.hypothesis, model = EXCLUDED.model,
                retrieval_s = EXCLUDED.retrieval_s, generation_s = EXCLUDED.generation_s,
                input_tokens = EXCLUDED.input_tokens, output_tokens = EXCLUDED.output_tokens,
                context_chars = EXCLUDED.context_chars, created_at = EXCLUDED.created_at, data = EXCLUDED.data
            """,
            rows,
        )
    return len(rows)


def export_judgments(conn) -> int:
    rows = [
        (r["question_id"], r["system"], r.get("question_type"), r.get("category"), r.get("abstention"),
         r.get("label"), r.get("judge_model"), r.get("judge_raw"), r.get("timestamp"))
        for r in latest_by(read_jsonl(JUDGMENTS_PATH), key=lambda r: (r["question_id"], r["system"])).values()
    ]
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO lme_julgamentos VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (question_id, system) DO UPDATE SET
                question_type = EXCLUDED.question_type, category = EXCLUDED.category,
                abstention = EXCLUDED.abstention, label = EXCLUDED.label, judge_model = EXCLUDED.judge_model,
                judge_raw = EXCLUDED.judge_raw, created_at = EXCLUDED.created_at
            """,
            rows,
        )
    return len(rows)


def export_ingest_log(conn) -> int:
    rows = [
        (r["backend"], r["question_id"], r.get("question_type"), r.get("category"), r.get("granularity"),
         r.get("status"), r.get("seconds"), r.get("input_tokens"), r.get("output_tokens"), r.get("timestamp"),
         Jsonb(r))
        for r in latest_by(read_jsonl(INGEST_LOG_PATH), key=lambda r: (r["backend"], r["question_id"])).values()
    ]
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO lme_ingest_log VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (backend, question_id) DO UPDATE SET
                question_type = EXCLUDED.question_type, category = EXCLUDED.category,
                granularity = EXCLUDED.granularity, status = EXCLUDED.status, seconds = EXCLUDED.seconds,
                input_tokens = EXCLUDED.input_tokens, output_tokens = EXCLUDED.output_tokens,
                created_at = EXCLUDED.created_at, data = EXCLUDED.data
            """,
            rows,
        )
    return len(rows)


async def fetch_graph_facts() -> list[tuple]:
    from config import open_graphiti

    g = await open_graphiti()
    try:
        records, _, _ = await g.driver.execute_query(
            """
            MATCH (s:Entity)-[e:RELATES_TO]->(t:Entity)
            WHERE e.group_id STARTS WITH 'lme_'
            RETURN e.uuid AS uuid, e.group_id AS group_id, s.name AS source, e.name AS relation,
                   t.name AS target, e.fact AS fact, e.valid_at AS valid_at, e.invalid_at AS invalid_at,
                   e.created_at AS created_at, e.expired_at AS expired_at, e.episodes AS episodes
            """
        )
    finally:
        await g.close()

    def native(v):
        return v.to_native() if hasattr(v, "to_native") else v

    return [
        (r["uuid"], r["group_id"], r["source"], r["relation"], r["target"], r["fact"],
         native(r["valid_at"]), native(r["invalid_at"]), native(r["created_at"]), native(r["expired_at"]),
         list(r["episodes"] or []))
        for r in records
    ]


def export_graph_facts(conn, rows: list[tuple]) -> int:
    with conn.transaction(), conn.cursor() as cur:
        # Substitui por grupo: fatos removidos numa reingestão também somem daqui.
        cur.execute("DELETE FROM lme_graph_facts WHERE group_id = ANY(%s)", (sorted({r[1] for r in rows}),))
        cur.executemany(
            "INSERT INTO lme_graph_facts VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            rows,
        )
    return len(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--graph", action="store_true", help="Também exporta os fatos do grafo (Neo4j)")
    args = p.parse_args()

    graph_rows = asyncio.run(fetch_graph_facts()) if args.graph else None

    with pg_connect() as conn:
        ensure_results_schema(conn)
        print(f"lme_respostas:   {export_answers(conn)} linhas")
        print(f"lme_julgamentos: {export_judgments(conn)} linhas")
        print(f"lme_ingest_log:  {export_ingest_log(conn)} linhas")
        if graph_rows is not None:
            print(f"lme_graph_facts: {export_graph_facts(conn, graph_rows)} linhas")


if __name__ == "__main__":
    main()
