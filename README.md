# Research-Temporal-Knowledge-Graphs

POC que compara **memória em grafo de conhecimento temporal** (Graphiti + Neo4j) com um
**baseline de RAG vetorial** (pgvector no Supabase) no benchmark
[LongMemEval](https://github.com/xiaowu0162/LongMemEval) (versão *oracle*), usando Gemini
como LLM e LLM-as-judge com os prompts oficiais do benchmark.

Sistemas avaliados (mesmo modelo e mesmo prompt; só muda a memória):

| sistema  | memória entregue ao LLM |
|----------|-------------------------|
| `graph`  | fatos do Graphiti com intervalo de validade (`valid_at`/`invalid_at`) + resumos de entidades |
| `vector` | top-k trechos de conversa por similaridade de cosseno (mesmo embedding do Graphiti) |
| `full`   | todas as sessões do oracle no prompt (teto de contexto longo, sem recuperação) |

## Estrutura

```
data/
  longmemeval_oracle.json   dataset completo (baixado)
  subset.json               amostra estratificada usada na POC
src/
  config.py         .env, Graphiti, cliente Gemini, conexão Postgres
  utils.py          caminhos, JSONL, formato LongMemEval, chamadas ao Gemini com retry
  prepare_data.py   baixa o dataset e gera o subset
  setup_db.py       índices do Neo4j + tabelas no Postgres
  smoke_test.py     checa conexões (Gemini, Neo4j, Postgres)
  ingest_graph.py   sessões -> episódios no Graphiti (1 group_id por pergunta)
  ingest_vector.py  sessões -> trechos com embedding no pgvector
  answer.py         recupera memória e responde -> results/respostas.jsonl
  judge.py          LLM-as-judge + relatório -> results/julgamentos.jsonl
  export_to_pg.py   resultados (e fatos do grafo) -> tabelas lme_* no Postgres
  results/          ingest_log.jsonl, respostas.jsonl, julgamentos.jsonl
```

## Configuração

```bash
pip install -r requirements.txt
```

`.env` na raiz:

```
NEO4J_URI=neo4j+s://<id>.databases.neo4j.io
NEO4J_USER=<id>            # instâncias novas do Aura: o ID da instância (antigas: neo4j)
NEO4J_PASSWORD=...
NEO4J_DATABASE=<id>        # idem; padrão: neo4j
GOOGLE_API_KEY=...
GEMINI_MODEL=gemini-2.5-flash
GEMINI_SMALL_MODEL=gemini-flash-lite-latest
GEMINI_EMBED_MODEL=...
PG_DSN=postgresql://...supabase.com:5432/postgres
SEMAPHORE_LIMIT=2          # chamadas simultâneas ao LLM dentro do Graphiti

# opcionais
ANSWER_MODEL=...           # padrão: GEMINI_MODEL
JUDGE_MODEL=...            # padrão: GEMINI_MODEL
EMBEDDING_DIM=1024         # padrão do Graphiti; mudar exige recriar índices e a tabela lme_chunks
SUPABASE_PASSWORD=...      # usado só se o PG_DSN não tiver senha
```

## Execução

Parte de bancos vazios. O que cada etapa escreve:

| etapa | Aura (Neo4j) | Supabase (Postgres) |
|-------|--------------|---------------------|
| `setup_db.py` | índices/constraints do Graphiti | extensão `vector` + tabelas `lme_*` vazias |
| `ingest_graph.py` | episódios, entidades e fatos (1 `group_id` por pergunta) | - |
| `ingest_vector.py` | - | trechos + embeddings em `lme_chunks` |
| `export_to_pg.py` | (só leitura) | respostas, julgamentos, log e fatos do grafo |

`answer.py` e `judge.py` só leem dos bancos. Todos os scripts são retomáveis: se caírem
(rate limit, rede), basta rodar de novo. O que já foi ingerido é conferido no banco, não
só no log local.

```bash
python src/prepare_data.py --download
python src/prepare_data.py --per-type 5 --seed 42
python src/setup_db.py
python src/smoke_test.py --e2e
python src/ingest_graph.py
python src/ingest_vector.py
python src/answer.py --systems graph vector full
python src/judge.py --csv
python src/export_to_pg.py --graph
```

Variações úteis:

```bash
python src/ingest_graph.py --granularity round          # 1 episódio por rodada (mais caro, mais fino)
python src/answer.py --systems graph --graph-search cross_encoder --tag ce
python src/answer.py --systems vector --vector-k 5 --tag k5
python src/judge.py --report-only --systems graph graph@ce
```

## Observações

- Dataset: `longmemeval_oracle.json` do repositório
  [xiaowu0162/longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)
  (500 perguntas, sha256 `821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c`,
  idêntico ao oracle do repositório original).
- No oracle, 34 perguntas têm sessões fora de ordem no arquivo; o código as ordena por data
  antes de ingerir. Em 43 perguntas (36 de temporal-reasoning), sessões de evidência têm
  horário posterior ao da pergunta, sempre no mesmo dia. É um artefato do benchmark, mantido
  como está para não divergir do protocolo oficial, e afeta os três sistemas igualmente.

- A ingestão no grafo é a etapa cara: o Graphiti faz várias chamadas de LLM por episódio
  (extração de entidades, deduplicação, extração e invalidação de arestas). O custo por
  pergunta fica em `results/ingest_log.jsonl` (`input_tokens`, `output_tokens`, `seconds`).
- O host direto do Supabase (`db.<projeto>.supabase.co`) só tem IPv6. Se a conexão falhar,
  use a string do *Session pooler* no `PG_DSN`.
