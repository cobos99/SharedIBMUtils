from qiskit_ibm_runtime.quantum_program.params_converters import QUANTUM_PROGRAM_PARAMS_CONVERTERS
from qiskit_ibm_runtime.decoders.executor.decoder import ExecutorResultDecoder
from qiskit_ibm_runtime.decoders.result_decoder import ResultDecoder
from qiskit_ibm_runtime import QiskitRuntimeService, RuntimeJobV2
from qiskit_ibm_runtime.results import QuantumProgramResult
from qiskit_ibm_runtime.json import RuntimeDecoder
from hardware.utils.circs import depth2qb
from collections.abc import Sequence
from typing import Any, NamedTuple
import numpy as np
import json
import os

type Batch = dict[str, Any]

class SavedJob(NamedTuple):
    job_id: str
    backend: str | None
    inputs: dict[str, Any]
    result: QuantumProgramResult

def save_job(job: RuntimeJobV2, path: str | os.PathLike) -> None:
    inputs = job.inputs
    converter = QUANTUM_PROGRAM_PARAMS_CONVERTERS[inputs["schema_version"]]
    backend = job.backend()
    with open(path, "w") as f:
        json.dump({
            "id": job.job_id(),
            "backend": backend.name if backend is not None else None,
            "program": {"id": "executor"},
            "params": converter.encoder(inputs["quantum_program"], inputs["options"]).model_dump(mode="json"),
            "result": job.result(decoder=ResultDecoder),
        }, f)

def load_job(path: str | os.PathLike) -> SavedJob:
    with open(path, "r") as f:
        saved = json.load(f, cls=RuntimeDecoder)
    result = ExecutorResultDecoder.decode(json.dumps(saved["result"]))
    if not isinstance(result, QuantumProgramResult):
        raise ValueError(f"{path} does not hold the result of an executor job")
    return SavedJob(saved["id"], saved["backend"], saved["params"], result)

def load_jobs_from_local_or_ibmq(job_ids: Sequence[str], ibmq_service: QiskitRuntimeService | None, jobs_local_folder: str | os.PathLike = "", verbose: bool = False) -> list[RuntimeJobV2 | SavedJob]:
    jobs = []
    for job_id in job_ids:
        local_path = os.path.join(jobs_local_folder, f"{job_id}.json")
        if os.path.exists(local_path):
            if verbose:
                print(f"Loading job {job_id} from local file {local_path}")
            jobs.append(load_job(local_path))
        else:
            if ibmq_service is None:
                raise ValueError(f"Job {job_id} is not saved in {local_path} and no ibmq_service was given to download it")
            if verbose:
                print(f"Loading job {job_id} from IBM Cloud")
            this_job = ibmq_service.job(job_id=job_id)
            save_job(this_job, local_path)
            jobs.append(load_job(local_path))
    return jobs

class JobDB:
    def __init__(self, path: str | os.PathLike, jobs_local_folder: str | os.PathLike = "") -> None:
        self.path = path
        self._data: list[Batch]
        self.jobs_local_folder = jobs_local_folder
        if os.path.exists(path):
            with open(path, "r") as f:
                self._data = json.load(f)
        else:
            self._data = []
            with open(path, "w") as f:
                json.dump(self._data, f, indent=4)

    def reload(self) -> None:
        with open(self.path, "r") as f:
            self._data = json.load(f)

    def save(self) -> None:
        text = json.dumps(self._data, indent=4)
        temporary_path = os.fspath(self.path) + ".tmp"
        with open(temporary_path, "w") as f:
            f.write(text)
        os.replace(temporary_path, self.path)

    @classmethod
    def generate_batch_db_data(cls, batch_args: dict[str, Any], jobs_arr: Sequence[RuntimeJobV2]) -> Batch:
        data_to_add = batch_args.copy()
        jobs_arr = np.array(jobs_arr, dtype=object)
        data_to_add["jobs_shape"] = jobs_arr.shape
        jobs_arr = jobs_arr.flatten()
        job_circuits = [[item.circuit for item in job.inputs["quantum_program"].items] for job in jobs_arr]
        data_to_add["jobs"] = [{
            "job_id": job.job_id(),
            "circuits": [{
                "depth": circuit.depth(),
                "depth2qb": depth2qb(circuit),
                "nops": circuit.count_ops(),
                "nqubits": circuit.num_qubits,
                "layout": circuit.layout.final_index_layout() if circuit.layout is not None else None
            } for circuit in job_circuits[i]]
        } for i, job in enumerate(jobs_arr)]
        return data_to_add

    def add(self, batch_args: dict[str, Any], jobs_arr: Sequence[RuntimeJobV2]) -> None:
        self.reload()
        thisid = 0 if len(self._data) == 0 else self._data[-1]["id"] + 1
        data_to_add = {"id": thisid}
        data_to_add = data_to_add | self.generate_batch_db_data(batch_args, jobs_arr)
        self._data.append(json.loads(json.dumps(data_to_add)))
        self.save()

    def remove(self, id: int) -> None:
        self.reload()
        index = self._search_batch_index_by_id(id)
        self._data.pop(index)
        self.save()

    def _search_batches_indices_by_params(self, batch_args: dict[str, Any], limit: int = 10) -> list[int]:
        batch_args = json.loads(json.dumps(batch_args))
        indices_to_return = []
        for i, batch in enumerate(self._data[::-1]):
            is_equal = ([
                batch.get(key, None) == val for key, val in batch_args.items()
            ])
            if all(is_equal): indices_to_return.append(len(self._data) - 1 - i)
            if len(indices_to_return) >= limit: break
        return indices_to_return

    def _search_batch_index_by_id(self, id: int) -> int:
        for i, batch in enumerate(self._data):
            if batch["id"] == id:
                return i
        raise ValueError(f"No batch found with id: {id}")

    def _search_batch_index_by_data(self, data: Batch) -> int:
        data = json.loads(json.dumps(data))
        for i, batch in enumerate(self._data):
            batch_without_id = batch.copy()
            del batch_without_id["id"]
            if batch_without_id == data:
                return i
        raise ValueError("No batch found with the provided data")

    def load_by_params(self, batch_args: dict[str, Any], ibmq_service: QiskitRuntimeService | None = None, limit: int = 10) -> list[list[RuntimeJobV2 | SavedJob]]:
        jobs_objs = []
        batches = [self._data[i] for i in self._search_batches_indices_by_params(batch_args, limit)]
        for batch in batches:
            this_jobs_ids = [job["job_id"] for job in batch["jobs"]]
            jobs_objs.append(load_jobs_from_local_or_ibmq(this_jobs_ids, ibmq_service, self.jobs_local_folder))
        return jobs_objs
    
    def load_by_id(self, id: int, ibmq_service: QiskitRuntimeService | None = None) -> list[RuntimeJobV2 | SavedJob]:
        data = self._data[self._search_batch_index_by_id(id)]
        jobs_ids = [job["job_id"] for job in data["jobs"]]
        return load_jobs_from_local_or_ibmq(jobs_ids, ibmq_service, self.jobs_local_folder)

    def load_by_data(self, data: Batch, ibmq_service: QiskitRuntimeService | None = None) -> list[RuntimeJobV2 | SavedJob]:
        data = self._data[self._search_batch_index_by_data(data)]
        jobs_ids = [job["job_id"] for job in data["jobs"]]
        return load_jobs_from_local_or_ibmq(jobs_ids, ibmq_service, self.jobs_local_folder)