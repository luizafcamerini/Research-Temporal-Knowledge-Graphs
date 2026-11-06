"""Ingere o histórico de cada pergunta do subset no Graphiti (Neo4j).

Cada pergunta vira uma partição própria (group_id = lme_<question_id>), e as sessões
entram em ordem cronológica com reference_time = data da sessão, para que o Graphiti
consiga datar e invalidar fatos (valid_at / invalid_at).

A ingestão é retomável: perguntas com status "ok" no ingest_log.jsonl e com todos os
episódios presentes no Neo4j são puladas. Uma pergunta que falhou no meio tem sua
partição apagada e é reingerida do zero. Funciona a partir de um banco vazio: os
índices do Graphiti são criados antes da primeira ingestão.

Exemplos:
    python src/ingest_graph.py
    python src/ingest_graph.py --granularity round --concurrency 2
    python src/ingest_graph.py --ids gpt4_2655b836 --force
"""
import argparse
import asyncio
import sys
import time
from datetime import timedelta

from graphiti_core import Graphiti
from graphiti_core.llm_client.errors import RateLimitError
from graphiti_core.nodes import EpisodeType
from graphiti_core.utils.maintenance.graph_data_operations import clear_data
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from config import GEMINI_MODEL, open_graphiti
from utils import (
    GRANULARITIES,
    INGEST_LOG_PATH,
    add_selection_args,
    append_jsonl,
    category_of,
    chunk_session,
    group_id_for,
    latest_by,
    load_subset,
    now_iso,
    read_jsonl,
    sessions_of,
)

BACKEND = "graph"


def build_episodes(item: dict, granularity: str) -> list[dict]:
    """Episódios da pergunta em ordem cronológica.

    Rodadas/turnos da mesma sessão recebem +N segundos no reference_time para que o
    Graphiti os ordene corretamente ao buscar episódios anteriores como contexto.
    """
    episodes = []
    for session in sessions_of(item):
        for i, body in enumerate(chunk_session(session, granularity)):
            if not body.strip():
                continue
            name = session.session_id if granularity == "session" else f"{session.session_id}#{i}"
            episodes.append({
                "name": name,
                "body": body,
                "reference_time": session.date + timedelta(seconds=i),
                "session_date": session.date_str,
            })
    return episodes


@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_random_exponential(multiplier=5, max=120),
    stop=stop_after_attempt(6),
    reraise=True,
)
async def add_episode(g: Graphiti, group_id: str, ep: dict):
    return await g.add_episode(
        name=ep["name"],
        episode_body=ep["body"],
        source_description=f"LongMemEval chat session ({ep['session_date']})",
        reference_time=ep["reference_time"],
        source=EpisodeType.message,
        group_id=group_id,
    )


async def graph_stats(g: Graphiti, group_id: str) -> dict:
    nodes, _, _ = await g.driver.execute_query(
        "MATCH (n:Entity {group_id: $gid}) RETURN count(n) AS n", params={"gid": group_id}
    )
    edges, _, _ = await g.driver.execute_query(
        """
        MATCH (:Entity)-[e:RELATES_TO {group_id: $gid}]->(:Entity)
        RETURN count(e) AS n, count(e.invalid_at) AS invalidated, count(e.valid_at) AS dated
        """,
        params={"gid": group_id},
    )
    return {
        "n_nodes": nodes[0]["n"],
        "n_edges": edges[0]["n"],
        "n_edges_invalidated": edges[0]["invalidated"],
        "n_edges_with_valid_at": edges[0]["dated"],
    }


async def ingest_question(g: Graphiti, item: dict, granularity: str, label: str) -> dict:
    qid = item["question_id"]
    gid = group_id_for(qid)
    episodes = build_episodes(item, granularity)
    record = {
        "backend": BACKEND,
        "question_id": qid,
        "question_type": item["question_type"],
        "category": category_of(item),
        "group_id": gid,
        "granularity": granularity,
        "model": GEMINI_MODEL,
        "n_sessions": len(item["haystack_sessions"]),
        "n_episodes": len(episodes),
    }

    tracker = g.llm_client.token_tracker
    tracker.reset()  # cada worker tem seu próprio Graphiti, então a contagem é só desta pergunta
    t0 = time.perf_counter()
    episode_seconds = []
    try:
        await clear_data(g.driver, group_ids=[gid])  # idempotência: descarta restos de execuções anteriores
        for i, ep in enumerate(episodes, 1):
            t_ep = time.perf_counter()
            await add_episode(g, gid, ep)
            episode_seconds.append(round(time.perf_counter() - t_ep, 2))
            print(f"  {label} episódio {i}/{len(episodes)} ({episode_seconds[-1]:.1f}s)")
        record.update(await graph_stats(g, gid))
        record["status"] = "ok"
    except Exception as e:  # noqa: BLE001 - registra e segue para a próxima pergunta
        record["status"] = "error"
        record["error"] = f"{type(e).__name__}: {e}"[:2000]
        record["episodes_done"] = len(episode_seconds)

    usage = tracker.get_total_usage()
    record.update({
        "seconds": round(time.perf_counter() - t0, 2),
        "episode_seconds": episode_seconds,
        "llm_calls": sum(u.call_count for u in tracker.get_usage().values()),
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "timestamp": now_iso(),
    })
    append_jsonl(INGEST_LOG_PATH, record)
    return record


async def episodes_in_db(g: Graphiti, group_ids: list[str]) -> dict[str, int]:
    records, _, _ = await g.driver.execute_query(
        "MATCH (e:Episodic) WHERE e.group_id IN $gids RETURN e.group_id AS gid, count(e) AS n",
        params={"gids": group_ids},
    )
    return {r["gid"]: r["n"] for r in records}


async def worker(queue: asyncio.Queue, total: int, granularity: str, results: list) -> None:
    g = await open_graphiti()
    try:
        while True:
            try:
                idx, item = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            label = f"[{idx}/{total}] {item['question_id']}"
            print(f"{label} iniciando ({len(item['haystack_sessions'])} sessões)")
            rec = await ingest_question(g, item, granularity, label)
            if rec["status"] == "ok":
                print(f"{label} ok em {rec['seconds']:.0f}s | nós={rec['n_nodes']} arestas={rec['n_edges']} "
                    f"(invalidadas={rec['n_edges_invalidated']}) | tokens={rec['input_tokens'] + rec['output_tokens']}")
            else:
                print(f"{label} ERRO: {rec['error'][:300]}", file=sys.stderr)
            results.append(rec)
    finally:
        await g.close()


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_selection_args(p)
    p.add_argument("--granularity", choices=GRANULARITIES, default="session",
                help="Unidade de cada episódio (padrão: session = 1 episódio por sessão)")
    p.add_argument("--concurrency", type=int, default=1,
                help="Perguntas ingeridas em paralelo (cada uma é sequencial internamente)")
    p.add_argument("--force", action="store_true", help="Reingere mesmo as perguntas já concluídas")
    args = p.parse_args()

    items = load_subset(args)

    # Cria os índices (banco vazio) uma única vez, antes dos workers, e confere o que já está no banco.
    g = await open_graphiti()
    try:
        in_db = await episodes_in_db(g, [group_id_for(it["question_id"]) for it in items])
    finally:
        await g.close()

    # Concluída = log diz "ok" com a mesma granularidade E o banco tem todos os episódios
    # (se o grafo foi apagado, o log sozinho não basta).
    logged_ok = {
        qid for (backend, qid), rec in latest_by(
            read_jsonl(INGEST_LOG_PATH), key=lambda r: (r.get("backend"), r.get("question_id"))
        ).items()
        if backend == BACKEND and rec.get("status") == "ok" and rec.get("granularity") == args.granularity
    }
    done = {
        it["question_id"] for it in items
        if it["question_id"] in logged_ok
        and in_db.get(group_id_for(it["question_id"])) == len(build_episodes(it, args.granularity))
    }
    pending = items if args.force else [it for it in items if it["question_id"] not in done]
    print(f"{len(items)} perguntas selecionadas | {len(items) - len(pending)} já ingeridas | {len(pending)} a ingerir")
    if not pending:
        return

    queue: asyncio.Queue = asyncio.Queue()
    for idx, item in enumerate(pending, 1):
        queue.put_nowait((idx, item))
    results: list[dict] = []
    t0 = time.perf_counter()
    await asyncio.gather(*(
        worker(queue, len(pending), args.granularity, results)
        for _ in range(max(1, min(args.concurrency, len(pending))))
    ))

    ok = [r for r in results if r["status"] == "ok"]
    print(f"\nConcluído em {time.perf_counter() - t0:.0f}s: {len(ok)} ok, {len(results) - len(ok)} com erro")
    if ok:
        print(f"Tokens LLM: entrada={sum(r['input_tokens'] for r in ok):,} saída={sum(r['output_tokens'] for r in ok):,}")
    if len(ok) < len(results):
        print("Rode o mesmo comando de novo para reprocessar as que falharam.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
