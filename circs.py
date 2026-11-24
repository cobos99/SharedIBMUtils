from qiskit.converters import circuit_to_instruction, dag_to_circuit, circuit_to_dag
from samplomatic.transpiler import generate_boxing_pass_manager
from qiskit.transpiler import generate_preset_pass_manager
from samplomatic.utils import find_unique_box_instructions
# qiskit.circuit re-exports CircuitInstruction as a plain variable, which type checkers reject as a type,
# so import it from the compiled module that defines it
from qiskit._accelerate.circuit import CircuitInstruction
from hardware.utils.misc import numpy_array_of_witerables
from qiskit.transpiler.passes import RemoveBarriers
from collections.abc import Callable, Iterable
from qiskit.dagcircuit import DAGCircuit
from qiskit.providers import BackendV2
from qiskit.circuit import Annotation
from typing import Any, SupportsIndex
from qiskit.quantum_info import Pauli
from qiskit import QuantumCircuit
import numpy.typing as npt
from qiskit import qpy
import samplomatic
import numpy as np
import os

# A circuit, or an arbitrarily nested list/tuple/array of circuits
type NestedCircuits = QuantumCircuit | list[NestedCircuits] | tuple[NestedCircuits, ...] | npt.NDArray

def count_non_idle_qubits(circ: QuantumCircuit | DAGCircuit) -> int:
    if type(circ) is not DAGCircuit:
        dag = circuit_to_dag(circ)
    else:
        dag = circ
    idle_wires = list(dag.idle_wires())
    return dag.num_qubits() - len(idle_wires)

def remove_barriers(circuit: QuantumCircuit) -> QuantumCircuit:
    # RemoveBarriers drops the TranspileLayout but moves no qubit, so the original layout is still valid
    layout = circuit.layout
    circuit = RemoveBarriers()(circuit)
    circuit._layout = layout
    return circuit

def simplify_logical_circuits(circuits: QuantumCircuit | list[QuantumCircuit], optimization_level: int = 3, backend: BackendV2 | None = None) -> QuantumCircuit | list[QuantumCircuit]:
    pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend)
    return pm.run(circuits)

def depth2qb(circ: NestedCircuits) -> int | npt.NDArray:
    if isinstance(circ, (list, tuple, np.ndarray)):
        return np.array([depth2qb(c) for c in circ])
    else:
        return circ.depth(lambda x: len(x.qubits)>1 and x.name!="barrier")

def load_circs(filepath: str | os.PathLike) -> list[QuantumCircuit]:
    with open(filepath, "rb") as f:
        circs = qpy.load(f)
    return circs

def save_circs(filepath: str | os.PathLike, circs: QuantumCircuit | list[QuantumCircuit]) -> None:
    with open(filepath, "wb") as f:
        qpy.dump(circs, f)

def numpy_array_of_circuits(circuits: NestedCircuits) -> npt.NDArray:
    return numpy_array_of_witerables(circuits)

def get_pauli_string_basischange(pauli: Pauli | str) -> QuantumCircuit:
    pauli = Pauli(pauli)
    change_of_basis = QuantumCircuit(pauli.num_qubits)
    # Read each qubit from the x/z bits, so neither the phase nor the label order can shift the qubit indices
    for i, (x, z) in enumerate(zip(pauli.x, pauli.z)):
        match (bool(x), bool(z)):
            case (True, False):  # X
                change_of_basis.h(i)
            case (True, True):  # Y
                change_of_basis.sdg(i)
                change_of_basis.h(i)
            case _:
                pass
    return change_of_basis

def choose_layout_from_longest_circuit(backend: BackendV2, circuits: NestedCircuits) -> dict[int, int]:
    circuits = numpy_array_of_circuits(circuits).flatten()
    depths = depth2qb(circuits)
    max_depth_idx = np.argmax(depths)
    longest_circuit = circuits[max_depth_idx]
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    transpiled_circuit = pm.run(longest_circuit)
    return transpiled_circuit.layout.final_index_layout() if transpiled_circuit.layout is not None else None

def transpile_circuits_with_consistent_layout(logical_circuits: NestedCircuits, layout_depth_index_func: Callable[[npt.NDArray], SupportsIndex] = np.argmax, **pm_kwargs: Any) -> npt.NDArray:
    logical_circuits = numpy_array_of_circuits(logical_circuits)
    ogshape = logical_circuits.shape
    logical_circuits = logical_circuits.flatten().tolist()
    pm = generate_preset_pass_manager(**pm_kwargs)
    depths = depth2qb(logical_circuits)
    max_depth_idx = layout_depth_index_func(depths)
    if pm_kwargs.get("initial_layout", None) is None:
        transpiled_circuit = pm.run(logical_circuits[max_depth_idx])
        if transpiled_circuit.layout is not None:
            last_layout = transpiled_circuit.layout.final_index_layout()
        else:
            last_layout = None
        pm_kwargs["initial_layout"] = last_layout
        pm = generate_preset_pass_manager(**pm_kwargs)
    transpiled_circuits = pm.run(logical_circuits)
    transpiled_circuits = numpy_array_of_circuits(transpiled_circuits).reshape(ogshape)
    return transpiled_circuits

def box_circuits(circ_arr: NestedCircuits, **box_kwargs: Any) -> npt.NDArray:
    circ_arr = numpy_array_of_circuits(circ_arr)
    bpm = generate_boxing_pass_manager(**box_kwargs)
    ogshape = circ_arr.shape
    circ_arr = circ_arr.flatten().tolist()
    boxed_circuits = bpm.run(circ_arr)
    boxed_circuits = numpy_array_of_circuits(boxed_circuits).reshape(ogshape)
    return boxed_circuits.squeeze().tolist()

def get_unique_boxes_many(circ_arr: QuantumCircuit | Iterable[QuantumCircuit], undress_boxes: bool = True, normalize_annotations: Callable[[Iterable[Annotation]], list[Annotation]] | None = None) -> list[list[CircuitInstruction]]:
    if isinstance(circ_arr, QuantumCircuit):
        circ_arr = [circ_arr]
    elif isinstance(circ_arr, (list, tuple)):
        if not all(isinstance(c, QuantumCircuit) for c in circ_arr):
            raise ValueError("All elements of circ_arr must be QuantumCircuits")
    else:
        raise ValueError("circ_arr must be a QuantumCircuit or a list/tuple of QuantumCircuits")
    unique_instructions = []
    for circ in circ_arr:
        unique_instructions.append(find_unique_box_instructions(circ.data, undress_boxes=undress_boxes, normalize_annotations=normalize_annotations))
    return unique_instructions

def samplomatic_build(circuits: NestedCircuits) -> tuple[npt.NDArray, npt.NDArray]:
    # With squeeze=False, templates and samplexes are object arrays with the shape of circuits, without dropping length-1 axes
    if isinstance(circuits, QuantumCircuit):
        circuits = [circuits]
    circuits = numpy_array_of_circuits(circuits)
    ogshape = circuits.shape
    templates = []
    samplexes = []
    for circ in circuits.flatten():
        this_template, this_samples = samplomatic.build(circ)
        templates.append(this_template)
        samplexes.append(this_samples)
    templates = numpy_array_of_circuits(templates).reshape(ogshape)
    samplexes_arr = np.empty(len(samplexes), dtype=object)
    samplexes_arr[:] = samplexes
    samplexes = samplexes_arr.reshape(ogshape)
    return templates.squeeze().tolist(), samplexes.squeeze().tolist()

def append_with_barrier(c1: QuantumCircuit, c2: QuantumCircuit) -> QuantumCircuit:
    if c1.num_qubits != c2.num_qubits:
        raise ValueError("Circuits must have the same number of qubits to be appended")
    c1 = c1.copy()
    c1.barrier()
    return c1.compose(c2)