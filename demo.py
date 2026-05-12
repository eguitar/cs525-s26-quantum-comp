"""
demo.py  --  CS525 Final Project
Runnable demonstration of the quantum computer simulator.

Run:
    python demo.py

If characters look wrong, set your terminal to UTF-8:
    Windows PowerShell: chcp 65001
    or run:  python -X utf8 demo.py
"""

import numpy as np
from quantum_computer import QuantumCircuit
from algorithms import (create_bell_state, quantum_teleportation,
                         deutsch_jozsa, bernstein_vazirani, grover_search)

np.random.seed(42)   # reproducible output for demos / screenshots


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def header(title: str):
    bar = "=" * 64
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)

def subheader(title: str):
    print(f"\n  -- {title} --")


# ---------------------------------------------------------------------------
# Demo 1: Superposition
# ---------------------------------------------------------------------------

def demo_superposition():
    header("DEMO 1: SUPERPOSITION (Hadamard Gate)")
    print("""
  A classical bit is always exactly 0 or 1.
  A qubit can be in a superposition of |0> and |1> simultaneously.
  The Hadamard gate (H) puts |0> into an equal superposition.
""")
    qc = QuantumCircuit(1)
    print("  Initial state |0>:")
    qc.print_state()

    qc.h(0)
    print("  After H gate  |+> = (|0> + |1>) / sqrt(2):")
    qc.print_state()

    counts = qc.sample(1000)
    print(f"\n  Sampling 1000 times: {counts}")
    print("  -> Each measurement is random; ~50% land on 0, ~50% on 1.")
    qc.print_circuit()


# ---------------------------------------------------------------------------
# Demo 2: Entanglement
# ---------------------------------------------------------------------------

def demo_entanglement():
    header("DEMO 2: ENTANGLEMENT (CNOT Gate)")
    print("""
  Apply H to qubit 0, then CNOT(0->1).
  The two qubits become entangled: their outcomes are always correlated.
  Measuring one qubit instantly determines the other -- no matter how
  far apart they are.  (Einstein called this "spooky action at a distance".)
""")
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.print_circuit()
    print("  Entangled state |Phi+> = (|00> + |11>) / sqrt(2):")
    qc.print_state()

    counts = qc.sample(1000)
    print(f"\n  1000 samples: {counts}")
    print("  -> Only |00> and |11> appear.  Never |01> or |10>.")


# ---------------------------------------------------------------------------
# Demo 3: All Four Bell States
# ---------------------------------------------------------------------------

def demo_bell_states():
    header("DEMO 3: THE FOUR BELL STATES")
    print("""
  The four Bell states are the maximally entangled 2-qubit basis.
  They differ only in relative phase and which outcomes are correlated.
""")
    bell_variants = [
        ('phi+', '|Phi+> = (|00> + |11>) / sqrt(2)',
         'Same bits, positive phase'),
        ('phi-', '|Phi-> = (|00> - |11>) / sqrt(2)',
         'Same bits, negative phase'),
        ('psi+', '|Psi+> = (|01> + |10>) / sqrt(2)',
         'Opposite bits, positive phase'),
        ('psi-', '|Psi-> = (|01> - |10>) / sqrt(2)',
         'Opposite bits, negative phase'),
    ]
    for variant, formula, desc in bell_variants:
        subheader(formula)
        print(f"    {desc}")
        qc = create_bell_state(variant)
        qc.print_state()
        counts = qc.sample(400)
        print(f"    Samples (400 shots): {counts}")


# ---------------------------------------------------------------------------
# Demo 4: Quantum Teleportation
# ---------------------------------------------------------------------------

def demo_teleportation():
    header("DEMO 4: QUANTUM TELEPORTATION")
    print("""
  Alice wants to send an unknown qubit state to Bob.
  They share a pre-entangled Bell pair.

  Protocol:
    1. Alice performs a Bell measurement on her qubit + her half of the pair.
    2. She sends the 2 classical measurement bits to Bob (classical channel).
    3. Bob applies X and/or Z corrections.
    4. Bob's qubit is now in exactly Alice's original state.

  Key insight: the quantum state is DESTROYED at Alice's end and
  RECREATED at Bob's.  No faster-than-light communication occurs.
""")
    test_states = [
        ('zero',  '|0>',  (1.0, 0.0)),
        ('one',   '|1>',  (0.0, 1.0)),
        ('plus',  '|+>',  (0.5, 0.5)),
        ('minus', '|->',  (0.5, 0.5)),
    ]

    for state_key, label, (expected_0, expected_1) in test_states:
        result = quantum_teleportation(state_key)
        m0, m1 = result['classical_bits']
        p0, p1 = result['bob_prob_0'], result['bob_prob_1']
        ok0 = abs(p0 - expected_0) < 0.01
        ok1 = abs(p1 - expected_1) < 0.01
        status = "OK" if (ok0 and ok1) else "FAIL"

        print(f"  Teleporting {label}:  classical bits=({m0},{m1})  "
              f"Bob P(|0>)={p0:.3f}  P(|1>)={p1:.3f}  [{status}]")

    print("\n  -> Bob's qubit always matches Alice's original state.")


# ---------------------------------------------------------------------------
# Demo 5: Deutsch-Jozsa
# ---------------------------------------------------------------------------

def demo_deutsch_jozsa():
    header("DEMO 5: DEUTSCH-JOZSA ALGORITHM")
    print("""
  Problem: given black-box f:{0,1}^n -> {0,1}, determine whether f is
    CONSTANT  (same output for every input) or
    BALANCED  (outputs 0 for exactly half the inputs, 1 for the other half).

  Classical worst case: 2^(n-1) + 1 queries.
  Quantum:              1 query.   <- Exponential speedup!

  Key idea: Hadamard gates create interference.  After the oracle,
  constructive interference on |00...0> signals CONSTANT;
  any other outcome signals BALANCED.
""")
    for n in [2, 4, 8]:
        print(f"  n = {n} (database size {2**n}, "
              f"classical worst case = {2**(n-1)+1} queries):")
        for oracle in ('constant_0', 'constant_1', 'balanced'):
            r = deutsch_jozsa(n, oracle)
            mark = "OK" if r['correct'] else "FAIL"
            print(f"    oracle={oracle:<12}  measured={r['measured']}  "
                  f"verdict={r['verdict']}  [{mark}]")
        print()


# ---------------------------------------------------------------------------
# Demo 6: Bernstein-Vazirani
# ---------------------------------------------------------------------------

def demo_bernstein_vazirani():
    header("DEMO 6: BERNSTEIN-VAZIRANI ALGORITHM")
    print("""
  Problem: given black-box f(x) = s . x (mod 2) for hidden string s,
  find s.

  Classical: n queries -- each query reveals one bit of s.
  Quantum:   1 query.   <- Linear speedup!

  Key idea: same phase-kickback trick as Deutsch-Jozsa.  After a single
  oracle call the state collapses directly to |s>.
""")
    secrets = ['1', '01', '101', '1010', '11001', '01101001']
    for secret in secrets:
        r = bernstein_vazirani(secret)
        mark = "OK" if r['correct'] else "FAIL"
        print(f"  secret='{secret}'  recovered='{r['recovered']}'  [{mark}]")

    n = len(secrets[-1])
    print(f"\n  -> For n={n} bits: classical needs {n} queries, quantum needs 1.")


# ---------------------------------------------------------------------------
# Demo 7: Grover's Search
# ---------------------------------------------------------------------------

def demo_grover():
    header("DEMO 7: GROVER'S SEARCH ALGORITHM")
    print("""
  Problem: find a single marked item in an UNSORTED database of N=2^n items.

  Classical average: N/2 queries.
  Grover's:          ~pi/4 * sqrt(N) queries.  <- Quadratic speedup!

  Mechanism:
    - Phase oracle flips the sign of the marked amplitude.
    - Diffusion operator (2|psi><psi| - I) reflects amplitudes
      about the mean, amplifying the marked item each iteration.
    - After ~sqrt(N) iterations the marked item has near-certain probability.
""")
    configs = [
        (3,  5,  "  8-item database "),
        (4,  11, " 16-item database "),
        (5,  23, " 32-item database "),
        (6,  47, " 64-item database "),
        (8, 200, "256-item database "),
    ]
    print(f"  {'Database':20}  {'Target':8}  {'Iters':6}  "
          f"{'Classical avg':14}  {'Success prob':12}")
    print("  " + "-" * 70)

    for n, target, label in configs:
        r = grover_search(target, n, shots=4096)
        print(f"  {label}  "
              f"|{r['target_bits']}> ({target:3d})  "
              f"{r['iterations']:6d}  "
              f"{r['classical_avg_queries']:14d}  "
              f"{r['success_probability']:>11.1%}")

    print("""
  -> Grover's algorithm is provably OPTIMAL for unstructured search.
     It is used as a subroutine in many other quantum algorithms and
     in attacks on symmetric cryptography (effectively halving key length).
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("""
+--------------------------------------------------------------+
|       QUANTUM COMPUTER SIMULATOR  --  Python                 |
|             CS525 Final Project                              |
|                                                              |
|  Simulates an n-qubit quantum computer via state vectors.    |
|  Demonstrates superposition, entanglement, measurement,      |
|  and five foundational quantum algorithms.                   |
+--------------------------------------------------------------+
""")

    demo_superposition()
    demo_entanglement()
    demo_bell_states()
    demo_teleportation()
    demo_deutsch_jozsa()
    demo_bernstein_vazirani()
    demo_grover()

    print("\n" + "=" * 64)
    print("  All demos complete.")
    print("  Source files:")
    print("    quantum_computer.py  -- QuantumCircuit simulator class")
    print("    algorithms.py        -- 5 quantum algorithms")
    print("    demo.py              -- this demo script")
    print("=" * 64 + "\n")
