"""Ordering helpers: prerequisite edges, cycle review and Kahn topological sort."""
from __future__ import annotations

import heapq

from .hashing import hash_obj, sha256_text


def prereq_item_id(card_id: str, text: str) -> str:
    return f"prereq:{card_id}:{sha256_text(text.strip())[:8]}"


def edge_item_id(src: str, dst: str) -> str:
    return f"edge:{src}->{dst}"


def compute_prereqs(cards: list[dict], embedder, threshold: float) -> dict:
    """Map free-text prerequisites to existing cards (excluding self) by embedding similarity."""
    import numpy as np

    from .embed import cosine_matrix

    queries = [(c["card_id"], p) for c in cards for p in c["prerequisites"] if p.strip()]
    candidates: dict[tuple, dict] = {}
    unresolved = []
    if queries and cards:
        cvec = embedder.encode([c["concept"] for c in cards])
        qvec = embedder.encode([p for _, p in queries])
        sim = cosine_matrix(qvec, cvec)
        ids = [c["card_id"] for c in cards]
        for qi, (cid, text) in enumerate(queries):
            row = sim[qi].copy()
            row[ids.index(cid)] = -np.inf
            best = int(np.argmax(row))
            score = float(row[best])
            if score >= threshold:
                k = (ids[best], cid)
                if k not in candidates or candidates[k]["similarity"] < score:
                    candidates[k] = {"from": ids[best], "to": cid, "similarity": round(score, 6),
                                     "prereq_text": text}
            else:
                unresolved.append({"item_id": prereq_item_id(cid, text), "card_id": cid,
                                   "prerequisite": text,
                                   "best_candidate": ids[best] if np.isfinite(score) else None,
                                   "similarity": round(score, 6) if np.isfinite(score) else None})
    edges = sorted(candidates.values(), key=lambda e: (e["from"], e["to"]))
    return {"threshold": threshold, "candidate_edges": edges, "unresolved_prerequisites": unresolved}


def strongly_connected(nodes: list[str], edges: list[tuple[str, str]]) -> list[list[str]]:
    """Kosaraju; returns components with more than one node, deterministic order."""
    adj = {n: [] for n in nodes}
    radj = {n: [] for n in nodes}
    for a, b in edges:
        adj[a].append(b)
        radj[b].append(a)
    seen, order = set(), []
    for n in nodes:
        if n in seen:
            continue
        stack = [(n, iter(sorted(adj[n])))]
        seen.add(n)
        while stack:
            v, it = stack[-1]
            nxt = next((w for w in it if w not in seen), None)
            if nxt is None:
                order.append(v)
                stack.pop()
            else:
                seen.add(nxt)
                stack.append((nxt, iter(sorted(adj[nxt]))))
    comp: dict[str, int] = {}
    comps: list[list[str]] = []
    for n in reversed(order):
        if n in comp:
            continue
        idx = len(comps)
        comps.append([])
        stack = [n]
        comp[n] = idx
        while stack:
            v = stack.pop()
            comps[idx].append(v)
            for w in radj[v]:
                if w not in comp:
                    comp[w] = idx
                    stack.append(w)
    return sorted((sorted(c) for c in comps if len(c) > 1), key=lambda c: c[0])


def kahn(nodes: list[str], edges: list[tuple[str, str]]) -> list[str]:
    """Topological order; among ready nodes the original order wins."""
    pos = {n: i for i, n in enumerate(nodes)}
    indeg = {n: 0 for n in nodes}
    adj = {n: [] for n in nodes}
    for a, b in edges:
        adj[a].append(b)
        indeg[b] += 1
    heap = [(pos[n], n) for n in nodes if indeg[n] == 0]
    heapq.heapify(heap)
    out = []
    while heap:
        _, n = heapq.heappop(heap)
        out.append(n)
        for m in adj[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                heapq.heappush(heap, (pos[m], m))
    if len(out) != len(nodes):
        raise ValueError("graph still has cycles")
    return out


def compute_order(cards: list[dict], cards_version: str, prereqs: dict, decisions: list[dict]) -> dict:
    """Apply review decisions to candidate edges, auto-break remaining cycles for the draft
    (lowest similarity edge inside the cycle, ties by id) and sort with Kahn."""
    ids = [c["card_id"] for c in cards]
    idset = set(ids)
    by_item = {d.get("item_id"): d for d in decisions}
    candidates = prereqs["candidate_edges"]
    human_removed, mapped = [], []
    edges: dict[tuple, dict] = {}
    for e in candidates:
        d = by_item.get(edge_item_id(e["from"], e["to"]))
        if d and d.get("decision") == "remove_edge":
            human_removed.append({"from": e["from"], "to": e["to"], "reason": d.get("reason", "")})
            continue
        edges[(e["from"], e["to"])] = {**e, "origin": "candidate"}
    for u in prereqs["unresolved_prerequisites"]:
        d = by_item.get(u["item_id"])
        if d and d.get("decision") == "map_prerequisite":
            tgt = d.get("target_card_id")
            if tgt in idset and tgt != u["card_id"]:
                k = (tgt, u["card_id"])
                if edge_item_id(*k) in by_item and by_item[edge_item_id(*k)].get("decision") == "remove_edge":
                    continue
                edges[k] = {"from": tgt, "to": u["card_id"], "similarity": 1.0,
                            "prereq_text": u["prerequisite"], "origin": "review_decision"}
                mapped.append({"from": tgt, "to": u["card_id"], "item_id": u["item_id"]})
    cycles = strongly_connected(ids, list(edges))
    auto_removed = []
    while True:
        sccs = strongly_connected(ids, list(edges))
        if not sccs:
            break
        for scc in sccs:
            members = set(scc)
            inner = [e for k, e in edges.items() if k[0] in members and k[1] in members]
            victim = min(inner, key=lambda e: (e["similarity"], e["from"], e["to"]))
            del edges[(victim["from"], victim["to"])]
            auto_removed.append({"item_id": edge_item_id(victim["from"], victim["to"]),
                                 "from": victim["from"], "to": victim["to"],
                                 "similarity": victim["similarity"], "cycle": scc,
                                 "confirmed": False})
    effective = sorted(edges.values(), key=lambda e: (e["from"], e["to"]))
    order = kahn(ids, [(e["from"], e["to"]) for e in effective])
    return {
        "cards_version": cards_version,
        "prereq_similarity": prereqs["threshold"],
        "candidate_edges": candidates,
        "human_removed_edges": human_removed,
        "mapped_edges": mapped,
        "cycles": cycles,
        "removed_edges": auto_removed,
        "effective_edges": effective,
        "unresolved_prerequisites": prereqs["unresolved_prerequisites"],
        "order": order,
    }


def order_hash(order: dict) -> str:
    return hash_obj(order)
