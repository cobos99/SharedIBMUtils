from qiskit_addon_utils.noise_management.post_selection import PostSelector
from qiskit_ibm_runtime import QiskitRuntimeService, RuntimeJobV2
from qiskit_ibm_runtime.quantum_program import SamplexItem
from qiskit_ibm_runtime.results import QuantumProgramResult
from qiskit.quantum_info import Pauli, PauliList
import hardware.utils.bootstrap as bootstrap
from qiskit.exceptions import QiskitError
from hardware.utils.jobdb import SavedJob
from collections.abc import Sequence
from typing import Literal
import numpy.typing as npt
import numpy as np

def pauli_to_str(pauli: Pauli | str) -> str:
    # Qiskit stores the phase as the exponent q of (-i)^q
    phasestr_dict = {0:"", 1:"-i", 2:"-", 3:"i"}
    if isinstance(pauli, Pauli):
        return phasestr_dict[pauli.phase] + "".join([str(opel) for opel in invert_pauli_order(pauli)])
    elif isinstance(pauli, str):
        return pauli
    else:
        raise ValueError("pauli argument is not a Pauli object")

def pauli_weight(pauli: Pauli | str) -> int:
    pauli = Pauli(pauli)
    # A qubit is non-identity when its x or z bit is set
    return int(np.sum((pauli.x.astype(int) + pauli.z.astype(int)) != 0))

def pauli_equal_uptophase(pauli1: Pauli | str, pauli2: Pauli | str) -> bool:
    try:
        pauli1 = Pauli(pauli1)
        pauli2 = Pauli(pauli2)
    except QiskitError:
        raise ValueError("Both arguments must be convertible to Pauli objects")
    return bool((pauli1.x == pauli2.x).all() and (pauli1.z == pauli2.z).all())

def invert_pauli_order(pauli: Pauli | str) -> Pauli:
    if not isinstance(pauli, Pauli):
        pauli = Pauli(pauli)
    return Pauli((pauli.z[::-1], pauli.x[::-1], pauli.phase))

def is_diagonal(pauli: Pauli | str, basis: str = "Z") -> bool:
    if not isinstance(pauli, (Pauli, str)):
        raise ValueError("This function only works for Pauli objects or strings")
    pauli_str = pauli_to_str(pauli)
    for c in pauli_str:
        if c not in ["I", "-", "i", "-i", basis]:
            return False
    return True

def diagonal_operators_check(operators: Sequence[Pauli | str] | PauliList) -> tuple[bool, str | None]:
    operators = list(operators)
    different_paulis = set("".join([pauli_to_str(op) for op in operators]))
    if len(remaining_ops := (different_paulis - {"I", "-", "i", "-i"})) > 1:
        return False, None
    # Identity-only operators are diagonal in any basis, so fall back to Z
    basis = remaining_ops.pop() if remaining_ops else "Z"
    all_diagonal = all([is_diagonal(op, basis=basis) for op in operators])
    return all_diagonal, basis

def from_bits(bits: npt.NDArray[np.bool_], diag_observables: Sequence[Pauli | str], confidence: float | None = None, bootstrap_samples: int = 1000) -> npt.NDArray | tuple[npt.NDArray, npt.NDArray]:
    # Takes bits with shape (shots, qubits) or (n, ..., shots, qubits) and returns the expectation values with shape (..., observables)
    # and, with confidence, (..., observables, 2). The first axis of the latter is averaged over by pooling its entries with the shots.
    bits = np.asarray(bits)
    if bits.dtype != np.bool_:
        raise ValueError("bits must be a boolean array")
    if bits.ndim < 2:
        raise ValueError("bits must be at least a 2D array")
    diag_observables = PauliList(diag_observables)
    if diag_observables.num_qubits != bits.shape[-1]:
        raise ValueError("The last axis of bits must have one entry per qubit of the observables")
    all_diagonal, basis = diagonal_operators_check(diag_observables)
    if not all_diagonal or basis != "Z":
        raise ValueError("All operators must be diagonal in the Z basis")
    phases = (-1j)**diag_observables.phase
    real_phases = np.all(phases.imag == 0)

    # The float32 matmul runs on BLAS and counts the qubits in each observable's support exactly. [Claude's got the tricks! 🔥]
    support_counts = bits.astype(np.float32) @ diag_observables.z.T.astype(np.float32)
    bit_parities = (support_counts.astype(np.int32) & 1).astype(bool)
    if bit_parities.ndim > 2:
        bit_parities = np.moveaxis(bit_parities, 0, -3).reshape(*bit_parities.shape[1:-2], -1, bit_parities.shape[-1])
    expectation_values = (1 - 2*np.mean(bit_parities, axis=-2)) * phases
    if real_phases:
        expectation_values = expectation_values.real
    if confidence is None:
        return expectation_values

    confidence_intervals = bootstrap.bootstrap_bits_mean(bit_parities, confidence=confidence, n_resamples=bootstrap_samples)
    confidence_intervals = phases[:, np.newaxis] * confidence_intervals
    # Multiplying by -1 or -i reverses the order of the ends, so swap them back
    flipped = (phases.real + phases.imag) < 0
    confidence_intervals = np.where(flipped[:, np.newaxis], confidence_intervals[..., ::-1], confidence_intervals)
    if real_phases:
        confidence_intervals = confidence_intervals.real
    return expectation_values, confidence_intervals

def from_quantum_program(job: RuntimeJobV2 | SavedJob, diag_observables: Sequence[Pauli | str], confidence: float | None = None, bootstrap_samples: int = 1000, register: str = "meas", non_markovian_post_strategy: Literal["none", "node", "edge"] = "none", service: QiskitRuntimeService | None = None) -> npt.NDArray | tuple[npt.NDArray, npt.NDArray]:
    result = job.result if isinstance(job, SavedJob) else job.result()
    if not isinstance(result, QuantumProgramResult):
        raise ValueError("The job result is not a QuantumProgramResult")
    try:
        bits = [d[register] for d in result]
    except KeyError:
        raise ValueError(f"At least one of the results does not contain a register named '{register}'")
    flips = [d.get(f"measurement_flips.{register}", np.zeros_like(d[register])) for d in result]
    items = job.inputs["quantum_program"].items
    # from_bits pools its first axis with the shots, so samplex items put their randomizations there and circuit items a dummy axis
    flipped_bits = [b ^ f if isinstance(item, SamplexItem) else (b ^ f)[np.newaxis] for b, f, item in zip(bits, flips, items, strict=True)]
    post_selected_items = [i for i, d in enumerate(result) if f"{register}_ps" in d.keys()] if non_markovian_post_strategy != "none" else []
    if not post_selected_items:
        return from_bits(np.stack(flipped_bits, axis=1), diag_observables, confidence=confidence, bootstrap_samples=bootstrap_samples)

    # A SavedJob only stores the name of its backend, so the service loads it
    if isinstance(job, SavedJob):
        backend = service.backend(job.backend) if service is not None and job.backend is not None else None
    else:
        backend = job.backend()
    if backend is None:
        raise ValueError("The post-selection needs the backend of the job, and a SavedJob needs a service to load it")
    # Post-selection keeps a different number of shots in each item and randomization, so the kept shots of each item are pooled
    # here and every item is read on its own
    for i in post_selected_items:
        if flipped_bits[i].ndim != 3:
            raise ValueError("Post-selection only supports samplex items of shape (randomizations,) and circuit items without parameter sweeps")
        post_selector = PostSelector.from_circuit(items[i].circuit, backend.coupling_map)
        mask = post_selector.compute_mask(result[i], strategy=non_markovian_post_strategy).reshape(flipped_bits[i].shape[:-1])
        flipped_bits[i] = flipped_bits[i][mask]
    item_results = [from_bits(b, diag_observables, confidence=confidence, bootstrap_samples=bootstrap_samples) for b in flipped_bits]
    if confidence is None:
        return np.stack(item_results)
    evs, cis = zip(*item_results)
    return np.stack(evs), np.stack(cis)
