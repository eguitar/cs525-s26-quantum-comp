"""
quantum_computer.py  --  CS525 Final Project
State-vector quantum computer simulator.

Represents an n-qubit system as a complex vector of length 2^n.
Qubit 0 is the most-significant bit (leftmost in ket notation).
  e.g. for 3 qubits, state index 5 = 101b -> |q0=1, q1=0, q2=1>
"""

import numpy as np
from collections import Counter

# ---------------------------------------------------------------------------
# Standard single-qubit gate matrices
# ---------------------------------------------------------------------------

_I  = np.eye(2, dtype=complex)
_X  = np.array([[0, 1], [1, 0]], dtype=complex)
_Y  = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z  = np.array([[1, 0], [0, -1]], dtype=complex)
_H  = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_S  = np.array([[1, 0], [0, 1j]], dtype=complex)
_T  = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

# Projection operators for controlled gates
_P0 = np.array([[1, 0], [0, 0]], dtype=complex)  # |0><0|
_P1 = np.array([[0, 0], [0, 1]], dtype=complex)  # |1><1|


# ---------------------------------------------------------------------------
# QuantumCircuit
# ---------------------------------------------------------------------------

class QuantumCircuit:
    """
    Simulates a quantum circuit using explicit state-vector evolution.

    Usage:
        qc = QuantumCircuit(2)
        qc.h(0).cx(0, 1)   # Bell state |Phi+>
        qc.print_state()
        counts = qc.sample(1000)
    """

    def __init__(self, num_qubits: int):
        if not (1 <= num_qubits <= 20):
            raise ValueError(f"num_qubits must be 1-20 (got {num_qubits})")
        self.num_qubits = num_qubits
        self._dim = 2 ** num_qubits
        self.state = np.zeros(self._dim, dtype=complex)
        self.state[0] = 1.0          # start in |00...0>
        self._ops: list = []         # record of operations for circuit drawing

    def reset(self):
        """Reset to |00...0>."""
        self.state[:] = 0
        self.state[0] = 1.0
        self._ops.clear()
        return self

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _apply(self, matrix: np.ndarray):
        """Multiply the full 2^n x 2^n operator into the state vector."""
        self.state = matrix @ self.state

    def _single_qubit_op(self, gate: np.ndarray, qubit: int) -> np.ndarray:
        """Lift a 2x2 gate to the full n-qubit space via Kronecker products."""
        op = np.array([[1.0 + 0j]])
        for i in range(self.num_qubits):
            op = np.kron(op, gate if i == qubit else _I)
        return op

    def _controlled_op(self, ctrl: int, tgt: int,
                        gate: np.ndarray) -> np.ndarray:
        """Apply `gate` to `tgt` when `ctrl` is |1>; identity otherwise."""
        def _build(proj, tgt_mat):
            op = np.array([[1.0 + 0j]])
            for i in range(self.num_qubits):
                if i == ctrl:
                    op = np.kron(op, proj)
                elif i == tgt:
                    op = np.kron(op, tgt_mat)
                else:
                    op = np.kron(op, _I)
            return op
        return _build(_P0, _I) + _build(_P1, gate)

    def _toffoli_op(self, c1: int, c2: int, tgt: int) -> np.ndarray:
        """Toffoli (CCX): apply X to tgt when both c1 and c2 are |1>."""
        def _build(p1, p2, t):
            op = np.array([[1.0 + 0j]])
            for i in range(self.num_qubits):
                if i == c1:
                    op = np.kron(op, p1)
                elif i == c2:
                    op = np.kron(op, p2)
                elif i == tgt:
                    op = np.kron(op, t)
                else:
                    op = np.kron(op, _I)
            return op
        return (_build(_P0, _P0, _I) + _build(_P0, _P1, _I) +
                _build(_P1, _P0, _I) + _build(_P1, _P1, _X))

    # -----------------------------------------------------------------------
    # Single-qubit gates
    # -----------------------------------------------------------------------

    def h(self, qubit: int):
        """Hadamard -- creates superposition."""
        self._apply(self._single_qubit_op(_H, qubit))
        self._ops.append(('H', qubit))
        return self

    def x(self, qubit: int):
        """Pauli-X -- quantum NOT (bit flip)."""
        self._apply(self._single_qubit_op(_X, qubit))
        self._ops.append(('X', qubit))
        return self

    def y(self, qubit: int):
        """Pauli-Y."""
        self._apply(self._single_qubit_op(_Y, qubit))
        self._ops.append(('Y', qubit))
        return self

    def z(self, qubit: int):
        """Pauli-Z -- phase flip."""
        self._apply(self._single_qubit_op(_Z, qubit))
        self._ops.append(('Z', qubit))
        return self

    def s(self, qubit: int):
        """S gate -- pi/2 phase rotation (sqrt of Z)."""
        self._apply(self._single_qubit_op(_S, qubit))
        self._ops.append(('S', qubit))
        return self

    def t(self, qubit: int):
        """T gate -- pi/4 phase rotation (4th root of Z)."""
        self._apply(self._single_qubit_op(_T, qubit))
        self._ops.append(('T', qubit))
        return self

    def rx(self, qubit: int, theta: float):
        """Rotation around the X axis by angle theta."""
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        gate = np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
        self._apply(self._single_qubit_op(gate, qubit))
        self._ops.append(('Rx', qubit))
        return self

    def ry(self, qubit: int, theta: float):
        """Rotation around the Y axis by angle theta."""
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        gate = np.array([[c, -s], [s, c]], dtype=complex)
        self._apply(self._single_qubit_op(gate, qubit))
        self._ops.append(('Ry', qubit))
        return self

    def rz(self, qubit: int, theta: float):
        """Rotation around the Z axis by angle theta."""
        gate = np.array([[np.exp(-1j * theta / 2), 0],
                          [0,  np.exp(1j * theta / 2)]], dtype=complex)
        self._apply(self._single_qubit_op(gate, qubit))
        self._ops.append(('Rz', qubit))
        return self

    # -----------------------------------------------------------------------
    # Two-qubit gates
    # -----------------------------------------------------------------------

    def cx(self, control: int, target: int):
        """CNOT -- flips target when control is |1>."""
        self._apply(self._controlled_op(control, target, _X))
        self._ops.append(('CX', control, target))
        return self

    def cz(self, control: int, target: int):
        """Controlled-Z -- phase flip on target when control is |1>."""
        self._apply(self._controlled_op(control, target, _Z))
        self._ops.append(('CZ', control, target))
        return self

    def swap(self, q1: int, q2: int):
        """SWAP -- exchange two qubit states (via 3 CNOTs)."""
        self.cx(q1, q2).cx(q2, q1).cx(q1, q2)
        return self

    # -----------------------------------------------------------------------
    # Three-qubit gate
    # -----------------------------------------------------------------------

    def ccx(self, c1: int, c2: int, target: int):
        """Toffoli (CCX) -- flips target when both controls are |1>."""
        self._apply(self._toffoli_op(c1, c2, target))
        self._ops.append(('CCX', c1, c2, target))
        return self

    # -----------------------------------------------------------------------
    # Measurement
    # -----------------------------------------------------------------------

    def measure(self, qubit: int) -> int:
        """
        Measure a single qubit in the computational basis.
        Collapses the state and returns 0 or 1.
        """
        probs = np.abs(self.state) ** 2
        n = self.num_qubits

        # Probability of |0> outcome for this qubit
        prob_0 = sum(
            probs[i] for i in range(self._dim)
            if not ((i >> (n - 1 - qubit)) & 1)
        )

        result = 0 if np.random.random() < prob_0 else 1

        # Project onto the measured subspace and renormalize
        new_state = np.zeros(self._dim, dtype=complex)
        for i in range(self._dim):
            if ((i >> (n - 1 - qubit)) & 1) == result:
                new_state[i] = self.state[i]
        norm = np.linalg.norm(new_state)
        self.state = new_state / norm

        self._ops.append(('M', qubit))
        return result

    def measure_all(self) -> list:
        """Measure all qubits. Returns list of bits (MSB first)."""
        return [self.measure(q) for q in range(self.num_qubits)]

    def sample(self, shots: int = 1024) -> dict:
        """
        Draw `shots` samples from the current probability distribution
        WITHOUT collapsing the state.
        Returns {bitstring: count}.
        """
        probs = np.abs(self.state) ** 2
        probs /= probs.sum()   # guard against floating-point drift
        indices = np.random.choice(self._dim, size=shots, p=probs)
        n = self.num_qubits
        return {format(k, f'0{n}b'): v
                for k, v in sorted(Counter(indices).items())}

    # -----------------------------------------------------------------------
    # State inspection
    # -----------------------------------------------------------------------

    def get_statevector(self) -> np.ndarray:
        return self.state.copy()

    def get_probabilities(self) -> dict:
        """Return {bitstring: probability} for all non-negligible amplitudes."""
        n = self.num_qubits
        probs = np.abs(self.state) ** 2
        return {format(i, f'0{n}b'): float(probs[i])
                for i in range(self._dim) if probs[i] > 1e-10}

    def print_state(self, threshold: float = 1e-6):
        """Print all basis states with non-negligible amplitude."""
        n = self.num_qubits
        print("  State vector:")
        found = False
        for i, amp in enumerate(self.state):
            if abs(amp) > threshold:
                bits = format(i, f'0{n}b')
                prob = abs(amp) ** 2
                if abs(amp.imag) > threshold:
                    amp_str = f"{amp.real:+.4f}{amp.imag:+.4f}j"
                else:
                    amp_str = f"{amp.real:+.4f}     "
                print(f"    |{bits}> : {amp_str}   prob={prob:.4f}")
                found = True
        if not found:
            print("    (zero state)")

    def print_circuit(self):
        """Render a basic ASCII circuit diagram."""
        n = self.num_qubits
        lines = [f"  q{i}: -" for i in range(n)]

        for op in self._ops:
            name = op[0]

            if len(op) == 2:                        # single-qubit gate
                qubit = op[1]
                label = f"[{name}]"
                w = len(label) + 2
                for i in range(n):
                    if i == qubit:
                        lines[i] += label + "-"
                    else:
                        lines[i] += "-" * w

            elif name in ('CX', 'CZ') and len(op) == 3:   # 2-qubit gate
                ctrl, tgt = op[1], op[2]
                sym = {'CX': '(+)', 'CZ': '[Z]'}[name]
                lo, hi = min(ctrl, tgt), max(ctrl, tgt)
                for i in range(n):
                    if i == ctrl:
                        lines[i] += "-[*]-"
                    elif i == tgt:
                        lines[i] += f"-{sym}-"
                    elif lo < i < hi:
                        lines[i] += "--|-"
                    else:
                        lines[i] += "-----"

            elif name == 'CCX' and len(op) == 4:           # Toffoli
                c1, c2, tgt = op[1], op[2], op[3]
                for i in range(n):
                    if i in (c1, c2):
                        lines[i] += "-[*]-"
                    elif i == tgt:
                        lines[i] += "-(+)-"
                    else:
                        lines[i] += "-----"

            elif name == 'M':                              # measurement
                qubit = op[1]
                for i in range(n):
                    lines[i] += "-[M]-" if i == qubit else "-----"

        print("\n  Circuit:")
        for line in lines:
            print(line)
        print()
