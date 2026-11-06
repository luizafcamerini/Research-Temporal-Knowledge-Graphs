"""Responde as perguntas do subset com cada sistema de memória e salva em respostas.jsonl.

Sistemas:
    graph   busca híbrida no Graphiti (fatos com validade temporal + entidades)
    vector  top-k trechos de conversa no pgvector (baseline de RAG)
    full    todas as sessões do oracle no prompt (teto de contexto longo, sem recuperação)

Todos usam o mesmo modelo e o mesmo prompt; só muda o bloco de memória.
Retomável: pares (question_id, system) já respondidos são pulados (use --overwrite).

Exemplos:
    python src/answer.py --systems graph vector full
    python src/answer.py --systems graph --graph-search cross_encoder --tag ce
    python src/answer.py --systems vector --vector-k 5 --tag k5
"""
import argparse
import asyncio
import sys
import time

from config import ANSWER_MODEL, EMBEDDING_DIM, GEMINI_EMBED_MODEL, make_genai_client
from utils import (
    ANSWERS_PATH,
    INGEST_LOG_PATH,
    add_selection_args,
    append_jsonl,
    category_of,
    embed_texts,
    fmt_dt,
    gather_limited,
    generate_text,
    group_id_for,
    is_abstention,
    latest_by,
    load_subset,
    now_iso,
    read_jsonl,
    sessions_of,
)

SYSTEMS = ("graph", "vector", "full")

MEMORY_DESCRIPTIONS = {
    "graph": (
        "The memory below was retrieved from a temporal knowledge graph built from past conversations "
        "with the user. FACTS carry a validity interval: when the fact became true and, if it is no longer "
        "true, when it stopped being true. ENTITIES carry a short summary."
    ),
    "vector": "The memory below contains excerpts of past conversations with the user, each with its date.",
    "full": "The memory below contains the past conversations with the user, each with its date.",
}

ANSWER_PROMPT = """You are a helpful assistant with long-term memory of your past conversations with the user.
{memory_description}

<MEMORY>
{context}
</MEMORY>

Current date: {question_date}
Question: {question}

Instructions:
- Answer using only the memory above.
- Resolve relative time expressions ("last week", "two months ago") against the dates in the memory and the current date.
- If the memory contains conflicting information, the most recent information is the valid one.
- If the memory does not contain the information needed to answer, say that you don't know.
- Be concise.

Answer:"""


# --------------------------------------------------------------------------- recuperação: grafo

def graph_search_config(name: str, k_facts: int, k_entities: int, k_episodes: int):
    from graphiti_core.search.search_config_recipes import (
        COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
        COMBINED_HYBRID_SEARCH_MMR,
        COMBINED_HYBRID_SEARCH_RRF,
    )

    recipes = {
        "rrf": COMBINED_HYBRID_SEARCH_RRF,
        "mmr": COMBINED_HYBRID_SEARCH_MMR,
        "cross_encoder": COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
    }
    # Cópia profunda: as receitas são globais e o próprio Graphiti as altera em search().
    cfg = recipes[name].model_copy(deep=True)
    cfg.limit = max(k_facts, k_entities, k_episodes, 1)
    cfg.community_config = None  # não construímos comunidades na ingestão
    if k_entities == 0:
        cfg.node_config = None
    if k_episodes == 0:
        cfg.episode_config = None
    return cfg


def format_graph_context(results, k_facts: int, k_entities: int, k_episodes: int) -> tuple[str, dict]:
    facts = results.edges[:k_facts]
    entities = results.nodes[:k_entities]
    episodes = results.episodes[:k_episodes]

    parts = ["<FACTS>", "# format: fact (valid from <start> to <end>)"]
    for e in facts:
        start = fmt_dt(e.valid_at) or "unknown"
        end = fmt_dt(e.invalid_at) or "present"
        parts.append(f"- {e.fact} (valid from {start} to {end})")
    parts.append("</FACTS>")
    if k_entities:
        parts += ["<ENTITIES>", "# format: entity name: summary"]
        parts += [f"- {n.name}: {n.summary}" for n in entities]
        parts.append("</ENTITIES>")
    if k_episodes:
        parts.append("<EPISODES>")
        parts += [f"[{fmt_dt(ep.valid_at)}]\n{ep.content}" for ep in episodes]
        parts.append("</EPISODES>")

    meta = {
        "n_facts": len(facts),
        "n_facts_invalidated": sum(1 for e in facts if e.invalid_at),
        "n_entities": len(entities),
        "n_episodes": len(episodes),
    }
    return "\n".join(parts), meta


# --------------------------------------------------------------------------- recuperação: texto

def format_conversation_blocks(blocks: list[tuple[str, str, str]]) -> str:
    """blocks = [(session_id, date_str, texto)] em ordem cronológica; agrupa por sessão."""
    out, current = [], None
    for session_id, date_str, text in blocks:
        if session_id != current:
            out.append(f"\n### Conversation on {date_str}")
            current = session_id
        out.append(text)
    return "\n".join(out).strip()


def full_context(item: dict) -> tuple[str, dict]:
    sessions = sessions_of(item)
    blocks = [
        (s.session_id, s.date_str, "\n".join(f"{t['role']}: {str(t['content']).strip()}" for t in s.turns))
        for s in sessions
    ]
    return format_conversation_blocks(blocks), {"n_sessions": len(sessions)}


# --------------------------------------------------------------------------- pipeline

class Answerer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.client = make_genai_client()
        self.graphiti = None
        self.pg = None

    async def __aenter__(self):
        if "graph" in self.args.systems:
            from config import open_graphiti

            self.graphiti = await open_graphiti()
        if "vector" in self.args.systems:
            from config import pg_connect

            self.pg = pg_connect()
        return self

    async def __aexit__(self, *exc):
        if self.graphiti is not None:
            await self.graphiti.close()
        if self.pg is not None:
            self.pg.close()

    def system_id(self, system: str) -> str:
        return f"{system}@{self.args.tag}" if self.args.tag else system

    def system_config(self, system: str) -> dict:
        a = self.args
        if system == "graph":
            return {"search": a.graph_search, "k_facts": a.graph_facts,
                    "k_entities": a.graph_entities, "k_episodes": a.graph_episodes}
        if system == "vector":
            return {"k": a.vector_k, "query_task_type": a.query_task_type}
        return {}

    async def retrieve(self, system: str, item: dict) -> tuple[str, dict]:
        a = self.args
        if system == "graph":
            cfg = graph_search_config(a.graph_search, a.graph_facts, a.graph_entities, a.graph_episodes)
            results = await self.graphiti.search_(
                item["question"], config=cfg, group_ids=[group_id_for(item["question_id"])]
            )
            return format_graph_context(results, a.graph_facts, a.graph_entities, a.graph_episodes)

        if system == "vector":
            from ingest_vector import search_chunks

            qvec = (await embed_texts(self.client, GEMINI_EMBED_MODEL, [item["question"]],
                                      dim=EMBEDDING_DIM, task_type=a.query_task_type))[0]
            rows = await asyncio.to_thread(search_chunks, self.pg, item["question_id"], qvec, a.vector_k)
            scores = [round(float(r["score"]), 4) for r in rows]
            rows.sort(key=lambda r: (r["session_date"], r["chunk_idx"]))  # ordem cronológica no prompt
            blocks = [(r["session_id"], r["session_date_str"], r["content"]) for r in rows]
            return format_conversation_blocks(blocks), {"n_chunks": len(rows), "scores": scores}

        return full_context(item)

    async def answer(self, system: str, item: dict) -> dict:
        t0 = time.perf_counter()
        context, meta = await self.retrieve(system, item)
        retrieval_s = time.perf_counter() - t0

        prompt = ANSWER_PROMPT.format(
            memory_description=MEMORY_DESCRIPTIONS[system],
            context=context or "(no memory retrieved)",
            question_date=item["question_date"],
            question=item["question"],
        )
        llm = await generate_text(self.client, ANSWER_MODEL, prompt)
        record = {
            "question_id": item["question_id"],
            "system": self.system_id(system),
            "base_system": system,
            "question_type": item["question_type"],
            "category": category_of(item),
            "abstention": is_abstention(item["question_id"]),
            "question": item["question"],
            "question_date": item["question_date"],
            "gold_answer": str(item["answer"]),
            "hypothesis": llm.text,
            "model": ANSWER_MODEL,
            "config": self.system_config(system),
            "retrieval": meta,
            "context_chars": len(context),
            "retrieval_s": round(retrieval_s, 3),
            "generation_s": round(llm.seconds, 3),
            "input_tokens": llm.input_tokens,
            "output_tokens": llm.output_tokens,
            "timestamp": now_iso(),
        }
        if not self.args.no_context:
            record["context"] = context
        return record

    async def ready_questions(self, system: str, items: list[dict]) -> set[str] | None:
        """Perguntas que podem ser respondidas: ingestão "ok" no log E dados presentes no banco.

        Checar o banco evita responder com memória vazia se o Neo4j/Supabase foi recriado.
        None = o sistema não depende de ingestão (full).
        """
        if system not in ("graph", "vector"):
            return None
        latest = latest_by(read_jsonl(INGEST_LOG_PATH), key=lambda r: (r.get("backend"), r.get("question_id")))
        logged = {qid for (b, qid), r in latest.items() if b == system and r.get("status") == "ok"}

        if system == "graph":
            from ingest_graph import episodes_in_db

            counts = await episodes_in_db(self.graphiti, [group_id_for(it["question_id"]) for it in items])
            in_db = {it["question_id"] for it in items if counts.get(group_id_for(it["question_id"]))}
        else:
            from ingest_vector import TABLE

            def query() -> set[str]:
                if self.pg.execute("SELECT to_regclass(%s)", (TABLE,)).fetchone()[0] is None:
                    return set()
                return {r[0] for r in self.pg.execute(f"SELECT DISTINCT question_id FROM {TABLE}").fetchall()}

            in_db = await asyncio.to_thread(query)
        return logged & in_db


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_selection_args(p)
    p.add_argument("--systems", nargs="+", choices=SYSTEMS, default=list(SYSTEMS))
    p.add_argument("--tag", help="Sufixo para distinguir variações (system vira 'graph@<tag>')")
    p.add_argument("--concurrency", type=int, default=2, help="Perguntas respondidas em paralelo")
    p.add_argument("--overwrite", action="store_true", help="Responde de novo mesmo o que já existe")
    p.add_argument("--no-context", action="store_true", help="Não salva o contexto recuperado no JSONL")
    g = p.add_argument_group("grafo")
    g.add_argument("--graph-search", choices=("rrf", "mmr", "cross_encoder"), default="rrf",
                   help="Reranker do Graphiti (cross_encoder faz 1 chamada de LLM por candidato)")
    g.add_argument("--graph-facts", type=int, default=20, help="Fatos (arestas) no contexto")
    g.add_argument("--graph-entities", type=int, default=10, help="Entidades (nós) no contexto")
    g.add_argument("--graph-episodes", type=int, default=0, help="Episódios brutos no contexto")
    v = p.add_argument_group("vetorial")
    v.add_argument("--vector-k", type=int, default=10, help="Trechos recuperados")
    v.add_argument("--query-task-type", default=None, help="task_type do embedding da pergunta (ex.: RETRIEVAL_QUERY)")
    args = p.parse_args()

    items = load_subset(args)
    done = set(latest_by(read_jsonl(ANSWERS_PATH), key=lambda r: (r["question_id"], r["system"])))

    errors = 0
    async with Answerer(args) as answerer:
        jobs: list[tuple[str, dict]] = []
        for system in args.systems:
            ready = await answerer.ready_questions(system, items)
            sid = answerer.system_id(system)
            skipped_missing = 0
            for item in items:
                if not args.overwrite and (item["question_id"], sid) in done:
                    continue
                if ready is not None and item["question_id"] not in ready:
                    skipped_missing += 1
                    continue
                jobs.append((system, item))
            if skipped_missing:
                print(f"[aviso] {sid}: {skipped_missing} perguntas puladas por não estarem no banco "
                      f"(rode ingest_{system}.py)", file=sys.stderr)

        print(f"{len(items)} perguntas x {len(args.systems)} sistemas | {len(jobs)} respostas a gerar")
        if not jobs:
            return

        async def run(i: int, job: tuple[str, dict]) -> None:
            nonlocal errors
            system, item = job
            label = f"[{i + 1}/{len(jobs)}] {answerer.system_id(system)} {item['question_id']}"
            try:
                rec = await answerer.answer(system, item)
            except Exception as e:  # noqa: BLE001 - segue com as demais; rodar de novo refaz só as que falharam
                errors += 1
                print(f"{label} ERRO: {type(e).__name__}: {str(e)[:300]}", file=sys.stderr)
                return
            append_jsonl(ANSWERS_PATH, rec)
            print(f"{label} ok ({rec['retrieval_s']:.1f}s busca, {rec['generation_s']:.1f}s geração)")

        await gather_limited(jobs, run, args.concurrency)

    print(f"\nRespostas salvas em {ANSWERS_PATH} ({len(jobs) - errors} novas, {errors} erros)")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
