from collections.abc import Callable, Iterable, Iterator, Sequence
from qiskit.quantum_info import Pauli, PauliList
from numpy.typing import ArrayLike, DTypeLike
import numpy.typing as npt
from typing import Any
import numpy as np

def iterate_nested_list(nested_list: Iterable[Any], extra_inclusions: tuple[type, ...] = ()) -> Iterator[Any]:
    for element in nested_list:
        if isinstance(element, (list, tuple, PauliList, *extra_inclusions)):
            yield from iterate_nested_list(element, extra_inclusions=extra_inclusions)
        else:
            yield element

def del_nested_element(lst: list[Any], indices: Sequence[int]) -> None:
    """
    Delete the element at the given indices in a nested list structure.
    Example:
        del_nested_element(lst, (2, 1))
    """
    if len(indices) == 1:
        del lst[indices[0]]
    else:
        del_nested_element(lst[indices[0]], indices[1:])

def del_nested_element_by_value(lst: list[Any], value: Any, *, eq: Callable[[Any, Any], bool] = lambda a, b: a == b) -> bool:
    """
    Delete the first occurrence of value in a nested list structure.
    Returns True if deleted, False otherwise.
    """
    if not isinstance(lst, list):
        return False
    for i, item in enumerate(lst):
        if eq(item, value):
            del lst[i]
            return True
        if isinstance(item, list):
            if del_nested_element_by_value(item, value, eq=eq):
                return True
    return False

def del_all_nested_elements_by_value(lst: list[Any], value: Any, *, eq: Callable[[Any, Any], bool] = lambda a, b: a == b) -> int:
    """
    Delete all occurrences of value in a nested list structure.
    Returns the number of deletions.
    """
    if not isinstance(lst, list):
        return 0
    count = 0
    i = 0
    while i < len(lst):
        item = lst[i]
        if eq(item, value):
            del lst[i]
            count += 1
            continue  # Don't increment i, as list shrinks
        if isinstance(item, list):
            count += del_all_nested_elements_by_value(item, value, eq=eq)
        i += 1
    return count

def numpy_array_of_witerables(weird_iterables: Sequence[Any] | npt.NDArray, extra_inclusions: tuple[type, ...] = ()) -> npt.NDArray:
    if isinstance(weird_iterables, np.ndarray):
        return weird_iterables
    try:
        len(weird_iterables[0])
    except TypeError:
        return np.array(weird_iterables, dtype=object)
    dimensions = 0
    last_element = weird_iterables
    dimensions_length = []
    types_arr = (tuple, list, np.ndarray, set, *extra_inclusions)
    if not isinstance(last_element, types_arr):
        # A single object (e.g. one QuantumCircuit) becomes a length-1 array, filled by hand because np.array would iterate it
        arr = np.empty(1, dtype=object)
        arr[0] = last_element
        return arr
    while isinstance(last_element, types_arr):
        try:
            el_lengths = [len(el) for el in last_element]
            dimensions_length.append(len(last_element))
            last_element = last_element[np.argmax(el_lengths)]
            dimensions += 1
        except TypeError:
            raise TypeError("weird_iterables is not a list of non tuple/list/set/ndarray iterables")
    arr = np.empty(dimensions_length, dtype=object)
    dimensions_length = np.array(dimensions_length)
    for i, qc in enumerate(iterate_nested_list(weird_iterables)):
        indices = np.unravel_index(i, dimensions_length)
        arr[indices] = qc
    return arr

def find_index_nested(lst: Iterable[Any], value: Any, *, eq: Callable[[Any, Any], bool] = lambda a, b: a == b) -> tuple[int, ...] | None:
    """Return a tuple path of indices to the first occurrence of value, or None if not found."""
    for i, item in enumerate(lst):
        if eq(item, value):
            return (i,)
        if isinstance(item, (list, tuple, PauliList)):
            sub = find_index_nested(item, value, eq=eq)
            if sub is not None:
                return (i,) + sub
    return None

def get_list_shape(lst: Any, extra_iterable_types: tuple[type, ...] = ()) -> tuple[int, ...]:
    if isinstance(lst, (list, tuple, np.ndarray, *extra_iterable_types)):
        lengths = [get_list_shape(item, extra_iterable_types=extra_iterable_types) for item in lst]
        max_shape = tuple(max(s[i] if len(s) > i else 0 for s in lengths) for i in range(max(map(len, lengths), default=0)))
        return (len(lst),) + max_shape
    else:
        return ()

def fill_array(arr: npt.NDArray, lst: Any, idx: tuple[int, ...] = (), extra_iterable_types: tuple[type, ...] = ()) -> None:
    if isinstance(lst, (list, tuple, np.ndarray, *extra_iterable_types)):
        for i, item in enumerate(lst):
            fill_array(arr, item, idx + (i,), extra_iterable_types=extra_iterable_types)
    else:
        arr[idx] = lst

def to_padded_array(nested_list: Sequence[Any] | npt.NDArray, fill_value: Any = np.nan, dtype: DTypeLike = float, extra_iterable_types: tuple[type, ...] = ()) -> npt.NDArray:
    shape = get_list_shape(nested_list, extra_iterable_types=extra_iterable_types)
    arr = np.full(shape, fill_value, dtype=dtype)
    fill_array(arr, nested_list, extra_iterable_types=extra_iterable_types)
    return arr

def padded_array_to_list(padded_array: npt.NDArray, pad_value: Any = np.nan) -> list[Any]:
    """Inverse of to_padded_array: nested lists with the trailing padding removed along every axis.
    Trailing sublists made only of padding are removed too, so an empty list at the end of the original is lost.
    """
    if not isinstance(padded_array, np.ndarray):
        raise ValueError("padded_array must be a numpy array.")
    pad_is_nan = isinstance(pad_value, float) and np.isnan(pad_value)

    def is_padding(item: Any) -> bool:
        if isinstance(item, list):
            return len(item) == 0
        if pad_is_nan:
            return isinstance(item, float) and np.isnan(item)
        return item is pad_value or item == pad_value

    def strip(item: Any) -> Any:
        if not isinstance(item, list):
            return item
        stripped = [strip(sub) for sub in item]
        while stripped and is_padding(stripped[-1]):
            stripped.pop()
        return stripped

    return strip(padded_array.tolist())


def roll_rows(a: npt.NDArray, shifts: ArrayLike) -> npt.NDArray:
    """Roll each row of 2D array `a` by the corresponding amount in `shifts`.
    Positive shift -> roll to the right. `shifts` can be scalar or 1D array-like len==a.shape[0].
    """
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError("a must be a 2D array.")
    row_shifts = np.asarray(shifts)
    if row_shifts.ndim > 1:
        raise ValueError("shifts must be a scalar or 1D array-like.")
    if row_shifts.ndim == 1 and row_shifts.shape[0] != a.shape[0]:
        raise ValueError("shifts must be a scalar or 1D array-like with length equal to number of rows in a.")
    # A scalar shift applies to every row
    row_shifts = np.broadcast_to(row_shifts, a.shape[0]) % a.shape[1]
    # build index array shape (n_rows, n_cols)
    cols = np.arange(a.shape[1])
    idx = (cols - row_shifts[:, None]) % a.shape[1]
    return np.take_along_axis(a, idx, axis=1)