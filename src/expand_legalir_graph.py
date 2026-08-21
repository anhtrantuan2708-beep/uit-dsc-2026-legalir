#!/usr/bin/env python3
"""Safely add one legal-reference neighbour to a LegalIR top-5 ranking.

This intentionally is not a learned graph model.  A graph neighbour may fill
only the last free slot; it never displaces the protected early V12 results.
That makes the test falsifiable and cheap before considering a larger model.
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def add_unique(output, values, limit):
    for value in values:
        value = str(value)
        if value not in output:
            output.append(value)
        if len(output) >= limit:
            return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path, help="Existing trusted top-5 JSON")
    parser.add_argument("candidates", type=Path, help="V12 top-20 candidate JSON")
    parser.add_argument("graph", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--keep-base", type=int, default=4)
    parser.add_argument("--seed-k", type=int, default=20)
    parser.add_argument("--max-seed-rank", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--relations", default="AMENDS,GUIDES,PURSUANT_TO,REFERS_TO")
    args = parser.parse_args()

    base, candidates, graph = load(args.base), load(args.candidates), load(args.graph)
    allowed = {name.strip() for name in args.relations.split(",") if name.strip()}
    # lower source rank is safer.  Relation priority only breaks ties, so it
    # cannot overwhelm retrieval evidence.
    relation_priority = {"AMENDS": 0, "GUIDES": 1, "PURSUANT_TO": 2, "REFERS_TO": 3}
    # Graph is serialized as {source_document_id: [edge, ...]} so it stays
    # compact and can be read directly without a graph database.
    adjacency = {
        str(source_id): [edge for edge in edges if edge.get("relation") in allowed]
        for source_id, edges in graph.items()
    }

    result, report = {}, Counter()
    examples = []
    for query_id, row in base.items():
        protected = [str(item) for item in row.get("answer", [])][: args.keep_base]
        ranked = candidates.get(query_id, {}).get("answer", [])[: args.seed_k]
        options = []
        for rank, source_id in enumerate(ranked, start=1):
            source_id = str(source_id)
            if rank > args.max_seed_rank:
                break
            for edge in adjacency.get(source_id, []):
                target = str(edge["target_id"])
                if target not in protected:
                    options.append((rank, relation_priority.get(edge["relation"], 9), target, edge["relation"], source_id))
        options.sort()
        answer = list(protected)
        if options:
            best = options[0]
            add_unique(answer, [best[2]], args.top_k)
            report["graph_added"] += 1
            report[f"relation_{best[3]}"] += 1
            if len(examples) < 10:
                examples.append({"query_id": query_id, "seed_id": best[4], "seed_rank": best[0], "relation": best[3], "added_id": best[2]})
        else:
            report["no_graph_option"] += 1
        # Backfill from the trusted ranking, not the weak graph candidate.
        add_unique(answer, row.get("answer", []), args.top_k)
        result[query_id] = {"answer": answer}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"queries": len(result), "settings": vars(args), "report": report, "examples": examples}, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
