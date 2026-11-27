from qiskit_ibm_runtime import QuantumProgram, Executor, Batch, Session, QiskitRuntimeService
from qiskit_ibm_runtime.options_models import ExecutorOptions
import hardware.utils.characterization as characterization
from qiskit.transpiler import generate_preset_pass_manager
from collections.abc import Callable, Sequence
import hardware.utils.circs as circutils
from samplomatic.samplex import Samplex
from qiskit.providers import BackendV2
from hardware.utils.jobdb import JobDB
from samplomatic.utils import unbox
from typing import Any
import numpy as np

def backends_objs_to_names(backends_arr: BackendV2 | Sequence[BackendV2] | str | Sequence[str]) -> str | list[str]:
    if type(backends_arr) == str:
        backends_arr = [backends_arr]
    else:
        try:
            backends_arr = list(backends_arr)
        except TypeError:
            backends_arr = [backends_arr]
    backends_name_arr = []
    for backend in backends_arr:
        if type(backend) != str:
            backends_name_arr.append(backend.name)
        else:
            backends_name_arr.append(backend)
    return backends_name_arr if len(backends_name_arr) > 1 else backends_name_arr[0]

def _check_samplex_options_have_shape(shape: tuple[int, ...] | None, samplex_options_func: Callable[[Samplex], dict[str, Any]] | None) -> None:
    if samplex_options_func is not None and shape is None:
        raise ValueError("samplex_options_func sets the arguments of samplex items, which are only built when shape is given")

def programs_from_circuits(circuits: circutils.NestedCircuits, shots: int = 2**18, shape: tuple[int, ...] | None = None, samplex_options_func: Callable[[Samplex], dict[str, Any]] | None = None) -> np.ndarray:
    # Returns an object array of programs with the shape of circuits. With a shape each boxed circuit becomes a samplex item
    # whose randomizations have that shape, otherwise a circuit item with its boxes inlined
    _check_samplex_options_have_shape(shape, samplex_options_func)
    circuits = circutils.numpy_array_of_circuits(circuits)
    programs = np.empty(circuits.shape, dtype=object)
    for idx, circuit in np.ndenumerate(circuits):
        program = QuantumProgram(shots=shots)
        # build and unbox drop the TranspileLayout but move no qubit, and the job DB records the layout from the item circuit
        if shape is not None:
            template, circuit_samplex = circutils.samplomatic_build(circuit)
            template._layout = circuit.layout
            samplex_arguments = samplex_options_func(circuit_samplex) if samplex_options_func is not None else None
            program.append_samplex_item(template, circuit_samplex, samplex_arguments=samplex_arguments, shape=shape)
        else:
            unboxed = unbox(circuit)
            unboxed._layout = circuit.layout
            program.append_circuit_item(unboxed)
        programs[idx] = program
    return programs

def batch(backend: BackendV2, programs: Sequence[QuantumProgram], executor_options: dict | ExecutorOptions={}, jobdb: JobDB | None = None, db_args: dict = {}, verbose=True):
    programs = np.array(programs, dtype=object)
    og_shape = programs.shape
    programs = programs.flatten()
    jobs = []
    with Batch(backend=backend) as b:
        executor = Executor(mode=b, options=executor_options)
        for program in programs:
            jobs.append(executor.run(program))

    jobs = np.array(jobs, dtype=object)
    jobs = jobs.reshape(og_shape)

    if jobdb is not None:
        jobdb.add(db_args, jobs)

    return jobs

def layout_session(service: QiskitRuntimeService, backend: BackendV2, circuits: circutils.NestedCircuits, shots: int = 2**18, shape: tuple[int, ...] | None = None, executor_options: dict | ExecutorOptions = {}, transpiler_options: dict = {"optimization_level": 3}, samplex_options_func: Callable[[Samplex], dict[str, Any]] | None = None, jobdb: JobDB | None = None, db_args: dict = {}) -> np.ndarray:
    # circuits are logical and boxed: each is transpiled against the backend characterized in the session and then
    # turned into a program as in programs_from_circuits
    _check_samplex_options_have_shape(shape, samplex_options_func)
    circuits = circutils.numpy_array_of_circuits(circuits)
    og_shape = circuits.shape
    db_args = db_args.copy()

    with Session(backend=backend) as session:
        # Run calibration to choose the best layout for the circuits
        db_args["batch_id"] = session.session_id
        characterization_experiments = ["readout", "rb_1q", "rb_2q"]
        characterization_ids = characterization.run_characterization_experiments(backend=backend, experiments=characterization_experiments)
        db_args["characterization_ids"] = characterization_ids
        backend_updated = characterization.update_backend(service=service, backend=backend, experiments=characterization_experiments, job_ids=characterization_ids)

        pm = generate_preset_pass_manager(**(transpiler_options | {"backend": backend_updated}))
        executor = Executor(mode=session, options=executor_options)
        jobs = []
        # Each circuit is compiled right before its job is submitted, so the session keeps receiving jobs while the rest compile
        for circuit in circuits.flatten():
            tqc = pm.run(circuit)
            program = programs_from_circuits([tqc], shots=shots, shape=shape, samplex_options_func=samplex_options_func)[0]
            jobs.append(executor.run(program))

    jobs = np.array(jobs, dtype=object)
    jobs = jobs.reshape(og_shape)

    if jobdb is not None:
        jobdb.add(db_args, jobs)

    return jobs
