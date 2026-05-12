"""
algorithms.py  --  CS525 Final Project
Quantum algorithms built on top of QuantumCircuit.

Algorithms:
  1. Bell States          -- maximally entangled 2-qubit states
  2. Quantum Teleportation -- transfer qubit state via entanglement
  3. Deutsch-Jozsa        -- constant vs balanced in 1 query (exponential speedup)
  4. Bernstein-Vazirani   -- recover hidden string in 1 query (linear speedup)
  5. Grover's Search      -- find marked item in O(sqrt(N)) steps (quadratic speedup)
"""

import numpy as np
from quantum_computer import QuantumCircuit


# ---------------------------------------------------------------------------
# 1.  Bell States
# ---------------------------------------------------------------------------

def create_bell_state(variant: str = 'phi+') -> QuantumCircuit:
    """
    Create one of the four maximally entangled Bell states.

    |Phi+> = (|00> + |11>) / sqrt(2)   variant='phi+'
    |Phi-> = (|00> - |11>) / sqrt(2)   variant='phi-'
    |Psi+> = (|01> + |10>) / sqrt(2)   variant='psi+'
    |Psi-> = (|01> - |10>) / sqrt(2)   variant='psi-'

    Circuit: optional Z/X to set phase/bit -> H(0) -> CNOT(0,1)
    """
    qc = QuantumCircuit(2)
    if variant in ('phi-', 'psi-'):
        qc.z(0)           # add relative minus sign
    if variant in ('psi+', 'psi-'):
        qc.x(1)           # swap |0> and |1> on qubit 1
    qc.h(0)
    qc.cx(0, 1)
    return qc


# ---------------------------------------------------------------------------
# 2.  Quantum Teleportation
# ---------------------------------------------------------------------------

def quantum_teleportation(state: str = 'plus') -> dict:
    """
    Teleport a qubit from Alice (qubit 0) to Bob (qubit 2).

    Steps:
      - Create shared Bell pair on qubits 1 (Alice) and 2 (Bob).
      - Alice performs Bell measurement on qubits 0 & 1.
      - Alice sends 2 classical bits to Bob.
      - Bob applies X and/or Z corrections; his qubit matches original state.

    state: 'zero' | 'one' | 'plus' | 'minus'
    Returns dict with measurement outcomes, Bob's probabilities, and circuit.
    """
    qc = QuantumCircuit(3)

    # Prepare the message qubit (qubit 0)
    if state == 'one':
        qc.x(0)
    elif state == 'plus':
        qc.h(0)
    elif state == 'minus':
        qc.h(0)
        qc.z(0)
    # 'zero' -> do nothing, qubit 0 stays in |0>

    # Create entangled Bell pair between qubits 1 and 2
    qc.h(1)
    qc.cx(1, 2)

    # Alice's Bell measurement
    qc.cx(0, 1)
    qc.h(0)
    m0 = qc.measure(0)   # Alice's classical bit 0
    m1 = qc.measure(1)   # Alice's classical bit 1

    # Bob applies corrections based on classical bits from Alice
    if m1:
        qc.x(2)
    if m0:
        qc.z(2)

    # Probability of qubit 2 being |0> vs |1>
    probs = qc.get_probabilities()
    p_bob_0 = sum(v for k, v in probs.items() if k[2] == '0')
    p_bob_1 = sum(v for k, v in probs.items() if k[2] == '1')

    return {
        'state': state,
        'classical_bits': (m0, m1),
        'bob_prob_0': p_bob_0,
        'bob_prob_1': p_bob_1,
        'circuit': qc,
    }


# ---------------------------------------------------------------------------
# 3.  Deutsch-Jozsa
# ---------------------------------------------------------------------------

def deutsch_jozsa(n: int, oracle: str = 'balanced') -> dict:
    """
    Determine if f: {0,1}^n -> {0,1} is constant or balanced.

    Classical worst case: 2^(n-1) + 1 queries.
    Quantum: exactly 1 query.  <- Exponential speedup!

    oracle: 'constant_0' | 'constant_1' | 'balanced'
      constant_0 : f(x) = 0 for all x
      constant_1 : f(x) = 1 for all x
      balanced   : f(x) = x_0  (0 for half inputs, 1 for other half)
    """
    qc = QuantumCircuit(n + 1)        # n input qubits + 1 ancilla

    # Put ancilla in |-> = (|0> - |1>)/sqrt(2) for phase kickback trick
    qc.x(n)
    qc.h(n)

    # Hadamard all input qubits -> uniform superposition over {0,1}^n
    for i in range(n):
        qc.h(i)

    # Oracle encodes f(x) as a phase: |x>|-> -> (-1)^f(x) |x>|->
    if oracle == 'constant_1':
        qc.x(n)          # global phase flip: f(x)=1 everywhere
    elif oracle == 'balanced':
        qc.cx(0, n)      # f(x) = x_0 (first input bit)
    # constant_0: identity -- f(x)=0 everywhere, no gate needed

    # Hadamard again on input qubits -> interference
    for i in range(n):
        qc.h(i)

    # Measure input qubits
    results = [qc.measure(i) for i in range(n)]

    # All zeros -> constant; any 1 -> balanced (provably)
    is_constant = all(r == 0 for r in results)
    expected_constant = oracle in ('constant_0', 'constant_1')

    return {
        'oracle': oracle,
        'n_qubits': n,
        'measured': results,
        'verdict': 'constant' if is_constant else 'balanced',
        'correct': (is_constant == expected_constant),
        'classical_queries_worst_case': 2 ** (n - 1) + 1,
        'circuit': qc,
    }


# ---------------------------------------------------------------------------
# 4.  Bernstein-Vazirani
# ---------------------------------------------------------------------------

def bernstein_vazirani(secret: str) -> dict:
    """
    Find the hidden bit string s such that f(x) = s . x (mod 2).

    Classical: n queries (must test each bit independently).
    Quantum: 1 query.  <- Linear speedup!

    secret: binary string e.g. '101'
    """
    n = len(secret)
    qc = QuantumCircuit(n + 1)

    # Ancilla in |-> for phase kickback
    qc.x(n)
    qc.h(n)

    # Hadamard all input qubits
    for i in range(n):
        qc.h(i)

    # Oracle: for each bit of s that is 1, apply CNOT(i -> ancilla)
    for i, bit in enumerate(secret):
        if bit == '1':
            qc.cx(i, n)

    # Hadamard again -> state collapses to |s>
    for i in range(n):
        qc.h(i)

    # Measure directly recovers s
    recovered = ''.join(str(qc.measure(i)) for i in range(n))

    return {
        'secret': secret,
        'recovered': recovered,
        'correct': recovered == secret,
        'circuit': qc,
    }


# ---------------------------------------------------------------------------
# 5.  Grover's Search
# ---------------------------------------------------------------------------

def grover_search(target: int, n_qubits: int = 3,
                  shots: int = 1024) -> dict:
    """
    Amplitude amplification search over 2^n_qubits items.

    Classical average: N/2 = 2^(n-1) queries.
    Grover's: O(sqrt(N)) queries.  <- Quadratic speedup!

    The phase oracle marks the target by flipping the sign of its amplitude.
    The diffusion operator (2|psi><psi| - I) then amplifies it.
    After ~pi/4 * sqrt(N) iterations the target has near-certain probability.

    target: integer index of the item to find (0 <= target < 2^n_qubits)
    """
    N = 2 ** n_qubits
    if not (0 <= target < N):
        raise ValueError(f"target {target} out of range [0, {N})")

    optimal_iters = max(1, round(np.pi / 4 * np.sqrt(N)))

    qc = QuantumCircuit(n_qubits)

    # Initialize uniform superposition: H on every qubit
    for i in range(n_qubits):
        qc.h(i)

    # Precompute uniform state vector for diffusion operator
    uniform = np.ones(N, dtype=complex) / np.sqrt(N)

    for _ in range(optimal_iters):
        # Phase oracle: flip the sign of the marked amplitude
        qc.state[target] *= -1

        # Grover diffusion operator: 2|psi><psi| - I
        # Reflects amplitudes about the average
        overlap = np.dot(uniform.conj(), qc.state)
        qc.state = 2 * overlap * uniform - qc.state

    counts = qc.sample(shots)
    target_bits = format(target, f'0{n_qubits}b')
    success_prob = counts.get(target_bits, 0) / shots

    return {
        'target': target,
        'target_bits': target_bits,
        'n_qubits': n_qubits,
        'database_size': N,
        'iterations': optimal_iters,
        'classical_avg_queries': N // 2,
        'counts': counts,
        'success_probability': success_prob,
        'circuit': qc,
    }
