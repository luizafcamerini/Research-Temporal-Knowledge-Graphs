"""Baixa o LongMemEval (oracle) e gera um subconjunto estratificado para a POC.

Exemplos:
    python src/prepare_data.py --download
    python src/prepare_data.py --per-type 5 --seed 42
    python src/prepare_data.py --per-type 3 --categories temporal-reasoning knowledge-update
"""
import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict

import requests

from utils import CATEGORIES, ORACLE_PATH, SUBSET_PATH, category_of, read_json, write_json

# Repositório atual dos autores ("replaces the original LongMemEval dataset", set/2025).
# A limpeza mexeu só nos históricos de _s/_m: o oracle é idêntico ao do repositório
# antigo (xiaowu0162/longmemeval, onde o arquivo hoje se chama "longmemeval_oracle", sem .json).
ORACLE_URL = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json"


def download(url: str, force: bool) -> None:
    if ORACLE_PATH.exists() and ORACLE_PATH.stat().st_size > 0 and not force:
        print(f"{ORACLE_PATH.name} já existe ({ORACLE_PATH.stat().st_size / 1e6:.1f} MB); use --force para baixar de novo")
        return
    ORACLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = ORACLE_PATH.with_suffix(".json.part")
    print(f"Baixando {url} ...")
    sha = hashlib.sha256()
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                sha.update(chunk)
    try:
        data = json.loads(tmp.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        tmp.unlink(missing_ok=True)
        sys.exit(f"O arquivo baixado não é um JSON válido ({e}); confira a URL")
    if not isinstance(data, list) or not data or "question_id" not in data[0]:
        tmp.unlink(missing_ok=True)
        sys.exit("O arquivo baixado não tem o formato do LongMemEval (lista de perguntas)")
    tmp.replace(ORACLE_PATH)
    print(f"Salvo em {ORACLE_PATH} ({ORACLE_PATH.stat().st_size / 1e6:.1f} MB, {len(data)} perguntas)")
    print(f"sha256: {sha.hexdigest()}  (anote para reprodutibilidade)")


def build_subset(data: list[dict], args: argparse.Namespace) -> list[dict]:
    if args.ids:
        wanted = set(args.ids)
        subset = [it for it in data if it["question_id"] in wanted]
        missing = wanted - {it["question_id"] for it in subset}
        if missing:
            print(f"[aviso] ids não encontrados: {sorted(missing)}", file=sys.stderr)
        return subset

    categories = args.categories or [c for c in CATEGORIES if not (args.no_abstention and c == "abstention")]
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for item in data:
        if args.max_sessions and len(item["haystack_sessions"]) > args.max_sessions:
            continue
        by_cat[category_of(item)].append(item)

    rng = random.Random(args.seed)
    subset = []
    for cat in categories:
        pool = sorted(by_cat.get(cat, []), key=lambda it: it["question_id"])  # determinístico
        if len(pool) < args.per_type:
            print(f"[aviso] {cat}: só {len(pool)} perguntas disponíveis", file=sys.stderr)
        subset.extend(rng.sample(pool, min(args.per_type, len(pool))))
    return subset


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--download", action="store_true", help="Baixa longmemeval_oracle.json e sai")
    p.add_argument("--url", default=ORACLE_URL)
    p.add_argument("--force", action="store_true", help="Sobrescreve o arquivo baixado")
    p.add_argument("--per-type", type=int, default=5, help="Perguntas por categoria (padrão: 5)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--categories", nargs="+", choices=CATEGORIES)
    p.add_argument("--no-abstention", action="store_true", help="Exclui perguntas sem resposta (_abs)")
    p.add_argument("--max-sessions", type=int, help="Descarta perguntas com mais sessões que isso (controle de custo)")
    p.add_argument("--ids", nargs="+", help="Usa exatamente estes question_id")
    args = p.parse_args()

    if args.download:
        download(args.url, args.force)
        return

    try:
        data = read_json(ORACLE_PATH)
    except FileNotFoundError:
        sys.exit(f"{ORACLE_PATH} está vazio. Rode antes: python src/prepare_data.py --download")

    subset = build_subset(data, args)
    write_json(SUBSET_PATH, subset)

    counts = Counter(category_of(it) for it in subset)
    n_sessions = sum(len(it["haystack_sessions"]) for it in subset)
    n_turns = sum(len(s) for it in subset for s in it["haystack_sessions"])
    print(f"{len(subset)} perguntas salvas em {SUBSET_PATH}")
    for cat in CATEGORIES:
        if counts.get(cat):
            print(f"  {cat:<28} {counts[cat]}")
    print(f"  total de sessões: {n_sessions} | turnos: {n_turns}")


if __name__ == "__main__":
    main()
