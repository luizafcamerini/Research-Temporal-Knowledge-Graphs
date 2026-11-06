"""Configuração central da POC: variáveis de ambiente e fábricas de clientes."""
import os

from dotenv import load_dotenv

# IMPORTANTE: precisa rodar ANTES de importar graphiti_core. A biblioteca lê
# SEMAPHORE_LIMIT, EMBEDDING_DIM etc. do ambiente no momento do import; se o
# .env for carregado depois, esses valores são ignorados silenciosamente.
load_dotenv()

from graphiti_core import Graphiti  # noqa: E402
from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient  # noqa: E402
from graphiti_core.driver.neo4j_driver import Neo4jDriver  # noqa: E402
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig  # noqa: E402
from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig  # noqa: E402


def _require(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise RuntimeError(f"Variável de ambiente obrigatória ausente: {name} (confira o .env)")
    return value


GOOGLE_API_KEY = _require("GOOGLE_API_KEY")
KEY = GOOGLE_API_KEY  # nome antigo, mantido por compatibilidade

GEMINI_MODEL = _require("GEMINI_MODEL")
GEMINI_SMALL_MODEL = (os.getenv("GEMINI_SMALL_MODEL") or "").strip() or GEMINI_MODEL
GEMINI_EMBED_MODEL = _require("GEMINI_EMBED_MODEL")

# Modelos usados fora do Graphiti (geração da resposta e juiz). Por padrão, o mesmo
# modelo principal, para que grafo e baseline vetorial sejam comparados em pé de igualdade.
ANSWER_MODEL = (os.getenv("ANSWER_MODEL") or "").strip() or GEMINI_MODEL
JUDGE_MODEL = (os.getenv("JUDGE_MODEL") or "").strip() or GEMINI_MODEL

# Mesma dimensão que o Graphiti usa (graphiti_core.embedder.client.EMBEDDING_DIM).
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
SEMAPHORE_LIMIT = int(os.getenv("SEMAPHORE_LIMIT", "20"))


class PerTextGeminiEmbedder(GeminiEmbedder):
    """GeminiEmbedder que devolve exatamente um vetor por texto.

    O gemini-embedding-2 é multimodal: `contents=[str, str, ...]` vira UM embedding agregado,
    e o create_batch original do graphiti_core 0.31.2 quebra com
    "embedder returned 1 vectors for N queries". Embrulhar cada texto num `Content`
    separado resolve (e continua funcionando no gemini-embedding-001).
    """

    async def _embed_contents(self, texts: list[str]) -> list[list[float]]:
        from google.genai import types

        result = await self.client.aio.models.embed_content(
            model=self.config.embedding_model,
            contents=[types.Content(parts=[types.Part(text=t)]) for t in texts],
            config=types.EmbedContentConfig(output_dimensionality=self.config.embedding_dim),
        )
        vectors = [list(e.values or []) for e in (result.embeddings or [])]
        if len(vectors) != len(texts) or not all(vectors):
            raise ValueError(f"Gemini devolveu {len(vectors)} embeddings para {len(texts)} textos")
        return vectors

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(input_data_list), self.batch_size):
            vectors.extend(await self._embed_contents(input_data_list[start:start + self.batch_size]))
        return vectors


# Instâncias novas do Aura usam o ID da instância como usuário E como nome do banco;
# as antigas usam "neo4j" para os dois.
NEO4J_DATABASE = (os.getenv("NEO4J_DATABASE") or "").strip() or "neo4j"


def make_graphiti() -> Graphiti:
    return Graphiti(
        graph_driver=Neo4jDriver(
            _require("NEO4J_URI"),
            _require("NEO4J_USER"),
            _require("NEO4J_PASSWORD"),
            database=NEO4J_DATABASE,
        ),
        llm_client=GeminiClient(config=LLMConfig(
            api_key=GOOGLE_API_KEY,
            model=GEMINI_MODEL,
            small_model=GEMINI_SMALL_MODEL,
            temperature=0,
        )),
        embedder=PerTextGeminiEmbedder(config=GeminiEmbedderConfig(
            api_key=GOOGLE_API_KEY,
            embedding_model=GEMINI_EMBED_MODEL,
            embedding_dim=EMBEDDING_DIM,
        )),
        cross_encoder=GeminiRerankerClient(config=LLMConfig(
            api_key=GOOGLE_API_KEY,
            model=GEMINI_SMALL_MODEL,
        )),
        max_coroutines=SEMAPHORE_LIMIT,
    )


async def open_graphiti() -> Graphiti:
    """Cria o Graphiti e só retorna depois que os índices do Neo4j estão prontos.

    O Neo4jDriver dispara a criação dos índices em background ao ser instanciado dentro
    de um event loop. Num banco vazio, a primeira ingestão poderia rodar antes dos índices
    fulltext existirem; esperar essa tarefa evita a corrida (e erros de autenticação
    aparecem aqui, logo no início).
    """
    g = make_graphiti()
    init_task = getattr(g.driver, "_init_task", None)
    try:
        if init_task is not None:
            await init_task
        else:
            await g.build_indices_and_constraints()
    except BaseException:
        await g.close()
        raise
    return g


def make_genai_client():
    """Cliente google-genai usado para responder, julgar e gerar embeddings do baseline."""
    from google import genai

    return genai.Client(api_key=GOOGLE_API_KEY)


def pg_connect(autocommit: bool = True):
    """Conexão síncrona com o Postgres (Supabase + pgvector)."""
    import psycopg
    from psycopg.conninfo import conninfo_to_dict

    dsn = _require("PG_DSN")
    kwargs = {}
    supabase_password = (os.getenv("SUPABASE_PASSWORD") or "").strip()
    if supabase_password and not conninfo_to_dict(dsn).get("password"):
        kwargs["password"] = supabase_password
    # prepare_threshold=None: o pooler do Supabase (pgbouncer em modo transação)
    # não suporta prepared statements.
    return psycopg.connect(dsn, autocommit=autocommit, prepare_threshold=None, **kwargs)
