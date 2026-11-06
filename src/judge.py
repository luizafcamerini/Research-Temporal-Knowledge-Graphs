"""Avalia as respostas com LLM-as-judge (protocolo do LongMemEval) e imprime a acurácia.

Os prompts do juiz seguem os do script oficial de avaliação do LongMemEval
(src/evaluation/evaluate_qa.py, Wu et al., 2024), um por tipo de pergunta,
para manter os números comparáveis com a literatura.

Exemplos:
    python src/judge.py                 # julga o que falta e mostra o relatório
    python src/judge.py --report-only   # só o relatório
    python src/judge.py --rejudge --systems graph
"""
import argparse
import asyncio
import re
import sys
from collections import defaultdict

from config import JUDGE_MODEL, make_genai_client
from utils import (
    ANSWERS_PATH,
    CATEGORIES,
    JUDGMENTS_PATH,
    RESULTS_DIR,
    append_jsonl,
    gather_limited,
    generate_text,
    is_abstention,
    latest_by,
    now_iso,
    read_jsonl,
)

_BASE = (
    "I will give you a question, a correct answer, and a response from a model. Please answer yes if the "
    "response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct "
    "answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If "
    "the response only contains a subset of the information required by the answer, answer no."
)
_TAIL = "\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\nIs the model response correct? Answer yes or no only."

JUDGE_PROMPTS = {
    "single-session-user": _BASE + _TAIL,
    "single-session-assistant": _BASE + _TAIL,
    "multi-session": _BASE + _TAIL,
    "temporal-reasoning": _BASE + (
        " In addition, do not penalize off-by-one errors for the number of days. If the question asks for the "
        "number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when "
        "the answer is 18), the model's response is still correct."
    ) + _TAIL,
    "knowledge-update": (
        "I will give you a question, a correct answer, and a response from a model. Please answer yes if the "
        "response contains the correct answer. Otherwise, answer no. If the response contains some previous "
        "information along with an updated answer, the response should be considered as correct as long as the "
        "updated answer is the required answer."
    ) + _TAIL,
    "single-session-preference": (
        "I will give you a question, a rubric for desired personalized response, and a response from a model. "
        "Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does "
        "not need to reflect all the points in the rubric. The response is correct as long as it recalls and "
        "utilizes the user's personal information correctly."
        "\n\nQuestion: {question}\n\nRubric: {answer}\n\nModel Response: {response}\n\n"
        "Is the model response correct? Answer yes or no only."
    ),
    "abstention": (
        "I will give you an unanswerable question, an explanation, and a response from a model. Please answer "
        "yes if the model correctly identifies the question as unanswerable. The model could say that the "
        "information is incomplete, or some other information is given but the asked information is not."
        "\n\nQuestion: {question}\n\nExplanation: {answer}\n\nModel Response: {response}\n\n"
        "Does the model correctly identify the question as unanswerable? Answer yes or no only."
    ),
}

_YES_RE = re.compile(r"\byes\b", re.IGNORECASE)


def judge_prompt(answer_rec: dict) -> str:
    key = "abstention" if is_abstention(answer_rec["question_id"]) else answer_rec["question_type"]
    return JUDGE_PROMPTS[key].format(
        question=answer_rec["question"],
        answer=answer_rec["gold_answer"],
        response=answer_rec["hypothesis"],
    )


async def judge_one(client, answer_rec: dict) -> dict:
    llm = await generate_text(client, JUDGE_MODEL, judge_prompt(answer_rec))
    return {
        "question_id": answer_rec["question_id"],
        "system": answer_rec["system"],
        "question_type": answer_rec["question_type"],
        "category": answer_rec.get("category", answer_rec["question_type"]),
        "abstention": is_abstention(answer_rec["question_id"]),
        "label": bool(_YES_RE.search(llm.text)),
        "judge_model": JUDGE_MODEL,
        "judge_raw": llm.text,
        # Liga o julgamento à versão exata da resposta (permite detectar respostas refeitas).
        "answer_timestamp": answer_rec.get("timestamp"),
        "timestamp": now_iso(),
    }


# --------------------------------------------------------------------------- relatório

def report(systems: list[str] | None = None, csv: bool = False) -> None:
    judgments = latest_by(read_jsonl(JUDGMENTS_PATH), key=lambda r: (r["question_id"], r["system"]))
    answers = latest_by(read_jsonl(ANSWERS_PATH), key=lambda r: (r["question_id"], r["system"]))
    # Considera só julgamentos da versão atual de cada resposta.
    valid = [
        j for k, j in judgments.items()
        if k in answers and j.get("answer_timestamp") == answers[k].get("timestamp")
        and (not systems or j["system"] in systems)
    ]
    if not valid:
        print("Nenhum julgamento encontrado.")
        return

    table: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for j in valid:
        table[j["system"]][j["category"]].append(j["label"])

    cats = [c for c in CATEGORIES if any(c in table[s] for s in table)]
    sys_names = sorted(table)
    width = max(12, *(len(s) for s in sys_names)) + 2

    def cell(labels: list[bool]) -> str:
        return f"{sum(labels) / len(labels):6.1%} ({sum(labels)}/{len(labels)})" if labels else "-"

    header = f"{'categoria':<28}" + "".join(f"{s:>{width + 8}}" for s in sys_names)
    print("\nAcurácia (LLM-as-judge: " + JUDGE_MODEL + ")")
    print(header)
    print("-" * len(header))
    for c in cats:
        print(f"{c:<28}" + "".join(f"{cell(table[s].get(c, [])):>{width + 8}}" for s in sys_names))
    print("-" * len(header))
    overall = {s: [x for labels in table[s].values() for x in labels] for s in sys_names}
    print(f"{'GERAL':<28}" + "".join(f"{cell(overall[s]):>{width + 8}}" for s in sys_names))
    task_avg = {
        s: sum(sum(table[s][c]) / len(table[s][c]) for c in table[s]) / len(table[s]) for s in sys_names
    }
    print(f"{'média por categoria':<28}" + "".join(f"{task_avg[s]:>{width + 8}.1%}" for s in sys_names))

    # Custo/latência de resposta por sistema.
    print("\nCusto médio por resposta")
    for s in sys_names:
        recs = [a for (qid, sid), a in answers.items() if sid == s]
        if recs:
            n = len(recs)
            print(f"  {s:<{width}} tokens entrada={sum(r['input_tokens'] for r in recs) / n:,.0f} "
                  f"saída={sum(r['output_tokens'] for r in recs) / n:,.0f} | "
                  f"busca={sum(r['retrieval_s'] for r in recs) / n:.2f}s "
                  f"geração={sum(r['generation_s'] for r in recs) / n:.2f}s")

    if csv:
        path = RESULTS_DIR / "acuracia.csv"
        with path.open("w", encoding="utf-8") as f:
            f.write("system,category,n,correct,accuracy\n")
            for s in sys_names:
                for c in cats:
                    labels = table[s].get(c, [])
                    if labels:
                        f.write(f"{s},{c},{len(labels)},{sum(labels)},{sum(labels) / len(labels):.4f}\n")
                f.write(f"{s},overall,{len(overall[s])},{sum(overall[s])},{sum(overall[s]) / len(overall[s]):.4f}\n")
        print(f"\nCSV salvo em {path}")


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--systems", nargs="+", help="Restringe a estes sistemas (ex.: graph vector@k5)")
    p.add_argument("--rejudge", action="store_true", help="Julga de novo mesmo o que já tem julgamento")
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--csv", action="store_true", help="Salva results/acuracia.csv")
    p.add_argument("--concurrency", type=int, default=4)
    args = p.parse_args()

    if not args.report_only:
        answers = latest_by(read_jsonl(ANSWERS_PATH), key=lambda r: (r["question_id"], r["system"]))
        judged = latest_by(read_jsonl(JUDGMENTS_PATH), key=lambda r: (r["question_id"], r["system"]))
        pending = [
            a for k, a in answers.items()
            if (not args.systems or a["system"] in args.systems)
            and (args.rejudge or k not in judged or judged[k].get("answer_timestamp") != a.get("timestamp"))
        ]
        print(f"{len(answers)} respostas | {len(pending)} a julgar com {JUDGE_MODEL}")

        if pending:
            client = make_genai_client()
            errors = 0

            async def run(i: int, rec: dict) -> None:
                nonlocal errors
                try:
                    j = await judge_one(client, rec)
                except Exception as e:  # noqa: BLE001
                    errors += 1
                    print(f"[{i + 1}/{len(pending)}] {rec['system']} {rec['question_id']} ERRO: {e}", file=sys.stderr)
                    return
                append_jsonl(JUDGMENTS_PATH, j)
                print(f"[{i + 1}/{len(pending)}] {rec['system']} {rec['question_id']}: "
                      f"{'correta' if j['label'] else 'errada'}")

            await gather_limited(pending, run, args.concurrency)
            if errors:
                print(f"{errors} julgamentos falharam; rode de novo para completar.", file=sys.stderr)

    report(args.systems, args.csv)


if __name__ == "__main__":
    asyncio.run(main())
