from qiskit_ibm_runtime import QuantumProgram, Executor, Batch, Session
from qiskit_ibm_runtime.options_models import ExecutorOptions
from samplomatic.samplex import Samplex
from qiskit.providers import BackendV2
from hardware.utils.jobdb import JobDB
from collections.abc import Sequence
from qiskit import QuantumCircuit
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

# def layout_session(backend: BackendV2, circuits: Sequence[QuantumCircuit], executor_options: dict | ExecutorOptions={}, samplex_options_func: callable | None = None, jobdb: JobDB | None = None, db_args: dict = {}):
#     circuits = np.array(circuits, dtype=object)
#     og_shape = circuits.shape
#     circuits = circuits.flatten()

#     # Run calibration to choose the best layout for the circuits
    