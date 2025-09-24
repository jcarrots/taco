# ordered_cumulants.py
# Minimal tools for: (i) Möbius inversion on ordered partitions (chronological cumulants),
#                    (ii) cancellation rules used next (zero-mean & connectedness for pairings).
#
# Requires: sympy

from __future__ import annotations
import itertools as it
import sympy as sp
from pathlib import Path


# ---------- Ordered partitions (compositions) ----------
def ordered_partitions(n: int):
    """
    Yield ordered partitions (into consecutive blocks) of {1,...,n}.
    Each partition is a tuple of blocks; each block is a tuple of indices.
    Example (n=4): ( (1,2,3,4), ), ( (1,), (2,3,4) ), ( (1,2), (3,4) ), ... ( (1,), (2,), (3,), (4,) )
    """
    if n <= 0:
        yield tuple()
        return
    # choose cut positions among {1,...,n-1}; Boolean lattice B_{n-1}
    cuts_positions = list(range(1, n))  # possible cuts after j
    for cut_set_size in range(0, n):    # 0..n-1 cuts
        for cuts in it.combinations(cuts_positions, cut_set_size):
            blocks = []
            prev = 0
            for c in cuts + (n,):       # append n as final cut
                block = tuple(range(prev + 1, c + 1))
                blocks.append(block)
                prev = c
            yield tuple(blocks)

# ---------- Möbius function on the ordered-partition poset ----------
def mobius_weight_ordered_partition(partition) -> int:
    """
    Möbius function μ(π) on the poset of ordered partitions into consecutive blocks.
    Because this poset is isomorphic to the Boolean lattice B_{n-1}, μ depends only on #blocks:
        μ(π) = (-1)^{k-1},   k = number of blocks in π
    """
    k = len(partition)
    return (-1) ** (k - 1) if k >= 1 else 0

# ---------- Symbols for ordered moments ----------
def moment_symbol(block, base: str = "M") -> sp.Symbol:
    """
    Return a SymPy symbol for the ordered moment of a single block:
        M_{i1_i2_..._ik}  representing  ⟨ X_{i1} X_{i2} ... X_{ik} ⟩
    """
    name = base + "_" + "_".join(str(i) for i in block)
    return sp.Symbol(name)

def moment_product(partition, base: str = "M") -> sp.Expr:
    """Product over blocks of their moment symbols."""
    expr = sp.Integer(1)
    for B in partition:
        expr *= moment_symbol(B, base=base)
    return expr

# ---------- Ordered (chronological) cumulant via Möbius inversion ----------
def ordered_cumulant_expr(n: int, base: str = "M") -> sp.Expr:
    """
    Return κ_n(X1,...,Xn) = sum_{π in OP(n)} μ(π) ∏_{B in π} ⟨X_B⟩,
    as a SymPy expression in abstract symbols M_block.
    """
    total = sp.Integer(0)
    for pi in ordered_partitions(n):
        mu = mobius_weight_ordered_partition(pi)
        total += mu * moment_product(pi, base=base)
    return sp.simplify(total)

# ---------- Zero-mean cancellation on ordered moments ----------
def apply_zero_mean(expr: sp.Expr, n: int, base: str = "M") -> sp.Expr:
    """
    Replace any singleton moment M_i (i.e. block of length 1) by 0 and simplify.
    """
    subs = { sp.Symbol(f"{base}_{i}") : 0 for i in range(1, n+1) }
    return sp.simplify(expr.subs(subs))

def ordered_partition_cancels_zero_mean(partition) -> bool:
    """True iff the ordered partition has any singleton block (k=1), hence vanishes for zero-mean baths."""
    return any(len(B) == 1 for B in partition)

# ---------- Pairing cancellation (connectedness for Wick pairings) ----------
def normalize_pairs(pairs):
    """Normalize and sort arc endpoints (i<j) and by left endpoint."""
    return sorted([(min(i,j), max(i,j)) for (i,j) in pairs], key=lambda p: (p[0], p[1]))

def pair_relation(p1, p2):
    """Classify two arcs p1=(i,j), p2=(k,l) with i<j, k<l."""
    (i,j) = p1; (k,l) = p2
    if j < k or l < i:
        return 'parallel'
    if (i < k and l < j) or (k < i and j < l):
        return 'nested'
    return 'crossed'

def pairing_overlap_graph_connected(pairs) -> bool:
    """
    Build graph whose vertices are arcs; edge if two arcs are NOT parallel.
    Return True iff the graph is connected (linked), i.e., survives the ordered cumulant.
    """
    arcs = normalize_pairs(pairs)
    n = len(arcs)
    if n == 0: return True
    adj = {i:set() for i in range(n)}
    for i in range(n):
        for j in range(i+1, n):
            if pair_relation(arcs[i], arcs[j]) != 'parallel':
                adj[i].add(j); adj[j].add(i)
    # BFS
    seen = {0}; stack = [0]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v); stack.append(v)
    return len(seen) == n

def pairing_cancels_by_cumulant(pairs) -> bool:
    """True iff the pairing is disconnected (e.g., pure 'parallel blocks') and thus cancels in TCL."""
    return not pairing_overlap_graph_connected(pairs)

# ---------- Tiny self-checks (optional) ----------
if __name__ == "__main__":
    # κ3 and κ4 (generic)
    k3 = ordered_cumulant_expr(3)
    k4 = ordered_cumulant_expr(4)
    k3_zm = apply_zero_mean(k3, 3)
    k4_zm = apply_zero_mean(k4, 4)

    # Example: TCL4 matchings
    nest  = [(1,4),(2,3)]
    cross = [(1,3),(2,4)]
    para  = [(1,2),(3,4)]

    print("nest connected? ", pairing_overlap_graph_connected(nest))
    print("cross connected?", pairing_overlap_graph_connected(cross))
    print("parallel connected?", pairing_overlap_graph_connected(para))

    results = []
    results.append(("kappa_3", str(k3)))
    results.append(("kappa_3 (zero mean)", str(k3_zm)))
    results.append(("kappa_4", str(k4)))
    results.append(("kappa_4 (zero mean)", str(k4_zm)))
    results.append(("nest connected?", pairing_overlap_graph_connected(nest)))
    results.append(("cross connected?", pairing_overlap_graph_connected(cross)))
    results.append(("parallel connected?", pairing_overlap_graph_connected(para)))

    # --- Export as plain text ---
    out_txt = Path("cumulant_results.txt")
    lines = [f"{label} = {value}" for label, value in results]
    out_txt.write_text("\n".join(lines), encoding="utf-8")

    # --- Export as CSV ---
    import csv
    out_csv = Path("cumulant_results.csv")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "value"])
        for label, value in results:
            writer.writerow([label, value])

    print(f"Results written to {out_txt.resolve()} and {out_csv.resolve()}")