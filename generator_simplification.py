"""
Ultra-compact symbolic TCL scaffolding.

Goal: keep **everything formal and lightweight**. All kernels are symbolic
placeholders; we do not expand integrals. You assemble TCL2/4/6 using a
minimal API of abstract functions:

  Γ_{ab}(ω,t)   := Gamma(a,b, ω, t)          (SymPy Function placeholder)
  I2_{ab}(ω1,ω2;t) := I2(a,b, ω1, ω2, t)    (defined *via* Γ, but kept symbolic)
  I3_{ab}(ω1,ω2,ω3;t) := I3(a,b, ω1, ω2, ω3, t)

You can later substitute concrete forms (e.g., sums of exponentials) with
`sympy.Substitution` if needed, but the default stays compact.

Author: (c) 2025 | MIT
"""
from __future__ import annotations

import sympy as sp
from dataclasses import dataclass
from typing import Dict, List, Tuple

I = sp.I

# --------------------------- Core symbolic atoms -----------------------------
# Abstract kernel symbols (do NOT expand)
Gamma_ab = sp.Function('Gamma_ab')  # Gamma_ab(a,b, w, t)
I2_ab    = sp.Function('I2_ab')     # I2_ab(a,b, w1, w2, t)
I3_ab    = sp.Function('I3_ab')     # I3_ab(a,b, w1, w2, w3, t)

# Optional: compact identities if you want I2/I3 written via Γ, but still symbolic
# Toggle this flag True→ express I2/I3 in terms of Γ; False→ keep them as atomic functions.
USE_REDUCED_FORMS = True

# --------------------------- Linear algebra helpers --------------------------

def dagger(A: sp.Matrix) -> sp.Matrix:
    return A.conjugate().T

def L_left(A: sp.Matrix) -> sp.Matrix:
    n = A.shape[0]
    return sp.kronecker_product(sp.eye(n), A)

def L_right(B: sp.Matrix) -> sp.Matrix:
    n = B.shape[0]
    return sp.kronecker_product(B.T, sp.eye(n))

def comm_super(O: sp.Matrix) -> sp.Matrix:
    """Superoperator for [O,·] as a Kronecker matrix."""
    n = O.shape[0]
    return sp.kronecker_product(sp.eye(n), O) - sp.kronecker_product(O.T, sp.eye(n))

# --------------------------- Formal kernels API ------------------------------

@dataclass
class FormalKernels:
    """Return abstract kernel expressions (compact placeholders)."""
    def Gamma(self, a:int, b:int, w:sp.Symbol, t:sp.Symbol) -> sp.Expr:
        return Gamma_ab(sp.Integer(a), sp.Integer(b), w, t)

    def I2(self, a:int, b:int, w1:sp.Symbol, w2:sp.Symbol, t:sp.Symbol) -> sp.Expr:
        if not USE_REDUCED_FORMS:
            return I2_ab(sp.Integer(a), sp.Integer(b), w1, w2, t)
        # Reduced (Hadamard) form but still symbolic in Γ
        return ( self.Gamma(a,b, w2, t) - sp.exp(I*w1*t)*self.Gamma(a,b, w2 - w1, t) ) / (I*w1)

    def I3(self, a:int, b:int, w1:sp.Symbol, w2:sp.Symbol, w3:sp.Symbol, t:sp.Symbol) -> sp.Expr:
        if not USE_REDUCED_FORMS:
            return I3_ab(sp.Integer(a), sp.Integer(b), w1, w2, w3, t)
        G = lambda W: self.Gamma(a,b, W, t)
        num = G(w3) - sp.exp(I*w2*t)*G(w3 - w2) - sp.exp(I*w1*t)*G(w3 - w1) \
              + sp.exp(I*(w1+w2)*t)*G(w3 - w1 - w2)
        return num / ((I*w1)*(I*w2))

# --------------------------- TCL2 superoperator (formal) ---------------------

def assemble_K2_symbolic(H: sp.Matrix,
                          A_parts: List[Dict[sp.Symbol, sp.Matrix]],
                          kernels: FormalKernels,
                          t: sp.Symbol) -> sp.Matrix:
    """
    Build K2(t) using abstract Γ. Input spectral parts per channel:
      A_parts[α] : { ω_symbol -> A_α(ω) }
    Output: (N^2×N^2) SymPy matrix with Γ placeholders (no expansion).
    """
    N = H.shape[0]
    K = sp.zeros(N*N, N*N)
    nchan = len(A_parts)
    for a in range(nchan):
        for b in range(nchan):
            Ab = A_parts[b]
            Aa = A_parts[a]
            A_a_full = sum(Aa.values(), sp.zeros(N)) if Aa else sp.zeros(N)
            for w, Ab_w in Ab.items():
                Ap_a_mw = Aa.get(-w, dagger(Aa.get(w, sp.zeros(N))))
                G_ab = kernels.Gamma(a,b, w, t)
                term1 = G_ab * ( sp.kronecker_product(dagger(Ap_a_mw).T, Ab_w)
                                - sp.kronecker_product(sp.eye(N), dagger(Ap_a_mw)*Ab_w) )
                G_ba_star = sp.conjugate(kernels.Gamma(b,a, w, t))
                term2 = G_ba_star * ( sp.kronecker_product(dagger(Ab_w).T, A_a_full)
                                     - sp.kronecker_product(sp.eye(N), dagger(Ab_w)*A_a_full) )
                K += term1 + term2
    return sp.simplify(K)

# --------------------------- TCL6 chain term (formal) -----------------------

def nested_comm_super(ops: List[sp.Matrix]) -> sp.Matrix:
    """Return [O1,[O2,[O3,·]]] superoperator (extend length as needed)."""
    assert len(ops) >= 1
    S = comm_super(ops[-1])
    for O in reversed(ops[:-1]):
        S = comm_super(O) * S
    return sp.simplify(S)


def assemble_TCL6_chain_term(A_list: List[sp.Matrix],
                              w1: sp.Symbol, w2: sp.Symbol, w3: sp.Symbol,
                              kernels: FormalKernels, t: sp.Symbol,
                              chan: Tuple[int,int]) -> sp.Matrix:
    """
    Representative TCL6 "chain" contribution (no H.c., no permutations):
      K6_chain  ~  I3_{ab}(w1,w2,w3;t) * [A1(ω1), [A2(ω2), [A3(ω3), · ] ] ]
    `chan` provides (a,b) for the kernel (typical mapping after Wick contraction).
    """
    N = A_list[0].shape[0]
    I3 = kernels.I3(chan[0], chan[1], w1, w2, w3, t)
    S  = nested_comm_super(A_list)
    return sp.simplify(I3 * S)

# ------------------------------- Usage sketch --------------------------------
if __name__ == "__main__":
    # Symbols
    t = sp.symbols('t', real=True, nonnegative=True)
    w1, w2, w3 = sp.symbols('w1 w2 w3', real=True)

    # Example 2×2 placeholders
    H11,H22,hx = sp.symbols('H11 H22 hx')
    H = sp.Matrix([[H11, hx],[hx, H22]])
    A_w1 = sp.MatrixSymbol('A_w1', 2, 2)
    A_wm1 = sp.MatrixSymbol('A_-w1', 2, 2)

    # Spectral parts for one channel (purely formal)
    A_parts = [ { w1: sp.Matrix(A_w1), -w1: sp.Matrix(A_wm1) } ]

    K = assemble_K2_symbolic(H, A_parts, kernels=FormalKernels(), t=t)
    # For TCL6 chain prototype:
    K6_chain = assemble_TCL6_chain_term([sp.Matrix(A_w1), sp.Matrix(A_w1), sp.Matrix(A_w1)],
                                        w1, w2, w3, kernels=FormalKernels(), t=t, chan=(0,0))
