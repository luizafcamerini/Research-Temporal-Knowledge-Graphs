"""Cria índices/constraints do Graphiti no Neo4j e as tabelas da POC no Postgres.

Exemplos:
    python src/setup_db.py
    python src/setup_db.py --skip-pg
    python src/setup_db.py --wipe-neo4j     # APAGA todo o grafo antes (pede confirmação)
"""
import argparse
import asyncio

from config import NEO4J_DATABASE, open_graphiti


async def setup_neo4j(wipe: bool) -> None:
    from graphiti_core.utils.maintenance.graph_data_operations import clear_data

    g = await open_graphiti()  # cria os índices/constraints (idempotente)
    try:
        if wipe:
            await clear_data(g.driver)
            print("Neo4j: grafo apagado")
        records, _, _ = await g.driver.execute_query("SHOW INDEXES YIELD name RETURN count(*) AS n")
        print(f"Neo4j: banco '{NEO4J_DATABASE}' pronto ({records[0]['n']} índices)")
    finally:
        await g.close()


def setup_pg() -> None:
    from config import pg_connect
    from export_to_pg import ensure_results_schema
    from ingest_vector import ensure_schema

    with pg_connect() as conn:
        ensure_schema(conn)
        ensure_results_schema(conn)
    print("Postgres: extensão pgvector e tabelas lme_* prontas")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-neo4j", action="store_true")
    p.add_argument("--skip-pg", action="store_true")
    p.add_argument("--wipe-neo4j", action="store_true", help="Apaga TODOS os nós e arestas do Neo4j antes")
    args = p.parse_args()

    if args.wipe_neo4j and input("Apagar todo o grafo do Neo4j? Digite 'sim': ").strip().lower() != "sim":
        print("Cancelado")
        return
    if not args.skip_neo4j:
        asyncio.run(setup_neo4j(args.wipe_neo4j))
    if not args.skip_pg:
        setup_pg()


if __name__ == "__main__":
    main()
