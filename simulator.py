from qiskit.transpiler import generate_preset_pass_manager
from hardware.utils.misc import numpy_array_of_witerables
from qiskit_aer.primitives import SamplerV2, EstimatorV2
from qiskit_ibm_runtime.ibm_backend import IBMBackend
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit_aer.noise import NoiseModel
from collections.abc import Sequence
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit
from typing import Literal

AerMethod = Literal[
    "automatic",
    "statevector",
    "density_matrix",
    "stabilizer",
    "extended_stabilizer",
    "matrix_product_state",
    "unitary",
    "superop",
    "tensor_network"
]

def simulate_sampler(circuits: QuantumCircuit, method: AerMethod = "automatic", backend: IBMBackend | None = None, run_options: dict = {}, **backend_kwargs):
    circuits = numpy_array_of_witerables(circuits)
    if backend is None:
        backend = AerSimulator(method=method, **backend_kwargs)
        options = {
            "backend_options": {
                "method": method,
                **backend_kwargs
            },
            "run_options": run_options
        }
    else:
        options = {
            "default_precision": 0,
            "backend_options": {
                **backend_kwargs
            },
            "run_options": run_options
        }
    sampler = SamplerV2.from_backend(backend, options=options)
    pm = generate_preset_pass_manager(backend=backend)
    transpiled_circuits = pm.run(circuits.tolist())
    job = sampler.run([(circ,) for circ in transpiled_circuits])
    samples = [result.data.meas.to_bool_array() for result in job.result()]
    return samples

def simulate_estimator(circuits: QuantumCircuit, observables: Sequence[Sequence[Pauli | SparsePauliOp]], method: AerMethod = "automatic", backend: IBMBackend | None = None, run_options: dict = {}, **backend_kwargs):
    """Expectation values of observables[i], e.g. a PauliList, on circuits[i]"""
    circuits = numpy_array_of_witerables(circuits)
    if backend is None:
        backend = AerSimulator(method=method, **backend_kwargs)
        options = {
            "backend_options": {
                "method": method,
                **backend_kwargs
            },
            "run_options": run_options
        }
    else:
        options = {
            "default_precision": 0,
            "backend_options": {
                **backend_kwargs
            },
            "run_options": run_options
        }
    estimator = EstimatorV2.from_backend(backend, options=options)
    pm = generate_preset_pass_manager(backend=backend)
    transpiled_circuits = pm.run(circuits.tolist())
    # The observables follow the qubits of each circuit through the layout chosen by the transpiler
    job = estimator.run([(circ, [ob.apply_layout(circ.layout) for ob in obs]) for circ, obs in zip(transpiled_circuits, observables, strict=True)])
    evs = [result.data.evs for result in job.result()]
    return evs