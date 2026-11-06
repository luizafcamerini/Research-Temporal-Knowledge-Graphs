"""Verifica se tudo está configurado: .env, Gemini (LLM e embedding), Neo4j e Postgres.

Exemplos:
    python src/smoke_test.py
    python src/smoke_test.py --e2e     # também ingere 1 episódio num grupo de teste e busca nele
    python src/smoke_test.py --skip-pg
"""
import argparse
import asyncio
import os
import sys
import time
import traceback
from datetime import datetime, timezone

from dotenv import load_dotenv

REQUIRED_ENV = ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "GOOGLE_API_KEY",
                "GEMINI_MODEL", "GEMINI_EMBED_MODEL"]
SMOKE_GROUP = "smoke_test"

failures: list[str] = []


async def check(name: str, coro_fn, verbose: bool) -> None:
    t0 = time.perf_counter()
    try:
        detail = await coro_fn()
        print(f"[OK]    {name} ({time.perf_counter() - t0:.1f}s){' - ' + detail if detail else ''}")
    except Exception as e:  # noqa: BLE001
        failures.append(name)
        print(f"[FALHA] {name}: {type(e).__name__}: {str(e)[:300]}")
        if verbose:
            traceback.print_exc()


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--e2e", action="store_true", help="Teste ponta a ponta no Graphiti (gasta algumas chamadas de LLM)")
    p.add_argument("--skip-pg", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true", help="Mostra o traceback das falhas")
    args = p.parse_args()

    load_dotenv()
    missing = [k for k in REQUIRED_ENV + ([] if args.skip_pg else ["PG_DSN"]) if not (os.getenv(k) or "").strip()]
    if missing:
        sys.exit(f"[FALHA] variáveis ausentes no .env: {', '.join(missing)}")
    print("[OK]    variáveis de ambiente")

    import config
    from utils import embed_texts, generate_text

    client = config.make_genai_client()

    async def llm():
        r = await generate_text(client, config.GEMINI_MODEL, "Reply with the single word: pong")
        return f"{config.GEMINI_MODEL} respondeu {r.text!r}"

    async def embedding():
        [vec] = await embed_texts(client, config.GEMINI_EMBED_MODEL, ["hello world"], dim=config.EMBEDDING_DIM)
        return f"{config.GEMINI_EMBED_MODEL}, dim={len(vec)}"

    await check("Gemini LLM", llm, args.verbose)
    await check("Gemini embedding", embedding, args.verbose)

    g = config.make_graphiti()
    try:
        async def neo4j():
            # Espera a criação dos índices disparada pelo driver (também valida a autenticação).
            init_task = getattr(g.driver, "_init_task", None)
            if init_task is not None:
                await init_task
            records, _, _ = await g.driver.execute_query("MATCH (n) RETURN count(n) AS n")
            return (f"banco '{config.NEO4J_DATABASE}' com {records[0]['n']} nós | "
                    f"SEMAPHORE_LIMIT={config.SEMAPHORE_LIMIT}")

        await check("Neo4j", neo4j, args.verbose)

        if args.e2e:
            from graphiti_core.nodes import EpisodeType
            from graphiti_core.utils.maintenance.graph_data_operations import clear_data

            async def e2e():
                await clear_data(g.driver, group_ids=[SMOKE_GROUP])
                await g.add_episode(
                    name="smoke-1",
                    episode_body="user: I just adopted a golden retriever named Max.\n"
                                 "assistant: Congratulations! Max is a great name.",
                    source_description="smoke test",
                    reference_time=datetime(2024, 3, 10, 12, 0, tzinfo=timezone.utc),
                    source=EpisodeType.message,
                    group_id=SMOKE_GROUP,
                )
                edges = await g.search("What is the name of the user's dog?", group_ids=[SMOKE_GROUP])
                await clear_data(g.driver, group_ids=[SMOKE_GROUP])
                if not edges:
                    raise RuntimeError("busca não retornou nenhum fato")
                return "fatos: " + " | ".join(e.fact for e in edges[:3])

            await check("Graphiti ponta a ponta", e2e, args.verbose)
    finally:
        await g.close()

    if not args.skip_pg:
        async def postgres():
            def run():
                with config.pg_connect() as conn:
                    version = conn.execute("SHOW server_version").fetchone()[0]
                    row = conn.execute(
                        "SELECT installed_version, default_version FROM pg_available_extensions WHERE name = 'vector'"
                    ).fetchone()
                if row is None:
                    raise RuntimeError("extensão pgvector não disponível neste servidor")
                status = f"pgvector {row[0]} instalada" if row[0] else "pgvector disponível (rode setup_db.py)"
                return f"Postgres {version} | {status}"

            return await asyncio.to_thread(run)

        await check("Postgres", postgres, args.verbose)

    if failures:
        sys.exit(f"\n{len(failures)} verificação(ões) falharam: {', '.join(failures)}")
    print("\nTudo certo.")


if __name__ == "__main__":
    asyncio.run(main())
