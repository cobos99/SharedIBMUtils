from hardware.utils.execution import backends_objs_to_names
from collections.abc import Mapping
from datetime import datetime
from typing import Any
import pandas as pd
import numpy as np
import os

class ExecutionRecord:
    def __init__(self, filepath: str | os.PathLike, columns: list[str]) -> None:
        self.filepath = filepath
        if os.path.exists(filepath):
            self.df = pd.read_excel(filepath)
            if not all(argcol == col for argcol, col in zip(columns, self.df.columns)):
                raise ValueError(f"Columns in the file do not match the provided columns.")
        else:
            self.df = pd.DataFrame(columns=columns)
            self.df.to_excel(filepath, index=False)

    def save(self) -> None:
        self.df.to_excel(self.filepath, index=False)

    def add_record(self, record: Mapping[str, Any]) -> None:
        new_row = [None]*len(self.df.columns)
        for key, value in record.items():
            if key in self.df.columns:
                new_row[self.df.columns.get_loc(key)] = value
        self.df.loc[-1] = new_row
        self.df.index = self.df.index + 1
        self.df = self.df.sort_index()
        self.save()

    def add_separator(self, lines: int = 3) -> None:
        for i in range(lines):
            self.df.loc[-1] = [None]*len(self.df.columns)
            self.df.index = self.df.index + 1
            self.df = self.df.sort_index()
        self.save()