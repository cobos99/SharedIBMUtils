from hardware.utils.execution import backends_objs_to_names
from collections.abc import Iterable, Sequence
from matplotlib.typing import ColorType
from qiskit.providers import BackendV2
from functools import cached_property
import matplotlib.pyplot as plt
import numpy.typing as npt
import matplotlib as mpl
import numpy as np
import os

type Coords = tuple[float, float]

def _num_half_integer_entries(coords: Coords | Sequence[Coords] | npt.NDArray) -> int | npt.NDArray:
    """Number of half-integer entries of a (row, column) tuple, or of each row of an N x 2 array:
    0 for a node, 1 for a link and 2 for an ancilla
    """
    single = isinstance(coords, tuple)
    coords_arr = np.asarray([coords] if single else coords, dtype=float)
    if coords_arr.ndim != 2 or coords_arr.shape[1] != 2:
        raise np.exceptions.AxisError("Coords is not a valid tuple or N x 2 array")
    fractional = coords_arr % 1
    if not np.all((fractional == 0) | (fractional == 0.5)):
        raise ValueError("Coordinates must be integers or half-integers")
    counts = np.count_nonzero(fractional == 0.5, axis=1)
    return int(counts[0]) if single else counts

def is_node_coords(coords: Coords | Sequence[Coords] | npt.NDArray) -> bool | npt.NDArray:
    return _num_half_integer_entries(coords) == 0

def is_link_coords(coords: Coords | Sequence[Coords] | npt.NDArray) -> bool | npt.NDArray:
    return _num_half_integer_entries(coords) == 1

def is_ancilla_coords(coords: Coords | Sequence[Coords] | npt.NDArray) -> bool | npt.NDArray:
    return _num_half_integer_entries(coords) == 2
    
class Lattice:
    def __init__(self, coords: Iterable[Coords]) -> None:
        self.coords = sorted(coords)
        kinds = _num_half_integer_entries(self.coords)
        self.node_qbinds: list[int] = np.flatnonzero(kinds == 0).tolist()
        self.link_qbinds: list[int] = np.flatnonzero(kinds == 1).tolist()
        self.ancilla_qbinds: list[int] = np.flatnonzero(kinds == 2).tolist()
        self.node_coords = [self.coords[i] for i in self.node_qbinds]
        self.link_coords = [self.coords[i] for i in self.link_qbinds]
        self.ancilla_coords = [self.coords[i] for i in self.ancilla_qbinds]

    @cached_property
    def max_x(self) -> float:
        return max([coords[1] for coords in self.coords])
    
    @cached_property
    def min_x(self) -> float:
        return min([coords[1] for coords in self.coords])
    
    @cached_property
    def max_y(self) -> float:
        return max([coords[0] for coords in self.coords])
    
    @cached_property
    def min_y(self) -> float:
        return min([coords[0] for coords in self.coords])
    
    def __len__(self) -> int:
        return len(self.coords)

    def coords_to_logical_qb(self, coords: Coords | Sequence[Coords]) -> int | list[int]:
        if isinstance(coords, tuple):
            coords = [tuple(coords)]
        to_return = [self.coords.index((coord[0], coord[1])) for coord in coords]
        return to_return if len(to_return) > 1 else to_return[0]

    def neighborhood(self, coords: Coords) -> list[Coords]:
        """Returns the coordinates of the qubits connected to the qubit at coords"""
        min_dist = 0.5 if len(self.link_coords) > 0 or len(self.ancilla_coords) > 0 else 1
        candidate_directions = [(0, min_dist), (0, -min_dist), (min_dist, 0), (-min_dist, 0)]
        candidate_coords = coords + np.array(candidate_directions)
        return sorted([tuple(c) for c in candidate_coords if tuple(c) in self.coords])
    
    def plot(self, scale: float = 1.5, number_qubits: bool = False, first_qubit: int | None = 0, link_boxes: bool = True, highlighted_qubits: Sequence[int] | None = None, colors: ColorType | list[ColorType] | npt.NDArray | None = None, filepath: str | os.PathLike | None = None) -> None:
        plt.rc("font", family="serif")
        node_x = np.array([n[1] for n in self.node_coords], dtype=int)
        node_y = np.array([self.max_y - n[0] for n in self.node_coords], dtype=int)
        # reshape keeps the N x 2 shape when the lattice has no links
        links_endpoints_x = np.array([[np.floor(link[1]), np.ceil(link[1])] for link in self.link_coords], dtype=int).reshape(-1, 2)
        links_endpoints_y = np.array([[self.max_y - np.floor(link[0]), self.max_y - np.ceil(link[0])] for link in self.link_coords], dtype=int).reshape(-1, 2)
        links_boxes_x = np.mean(links_endpoints_x, axis=1)
        links_boxes_y = np.mean(links_endpoints_y, axis=1)
        ancilla_x = np.array([a[1] for a in self.ancilla_coords])
        ancilla_y = np.array([self.max_y - a[0] for a in self.ancilla_coords])

        fig, ax = plt.subplots(figsize=[scale*self.max_x, scale*self.max_y])
        ax.set_aspect('equal')
        for lxs, lys in zip(links_endpoints_x, links_endpoints_y):
            plt.plot(lxs, lys, color="black")
        # Dashed lines from each ancilla to its nearest qubits, at distance 0.5 (not the diagonal ones at 0.5*sqrt(2))
        for a in self.ancilla_coords:
            for c in self.coords:
                if np.isclose(np.hypot(c[0] - a[0], c[1] - a[1]), 0.5):
                    plt.plot([a[1], c[1]], [self.max_y - a[0], self.max_y - c[0]], color="black", linestyle="--", zorder=1)
        plt.scatter(node_x, node_y, 350*scale, marker="o", c="white", edgecolors="black", zorder=2)
        if link_boxes:
            plt.scatter(links_boxes_x, links_boxes_y, 400*scale, marker=(4, 0, 45), c="white", edgecolors="black", zorder=2)
        if self.ancilla_coords:
            plt.scatter(ancilla_x, ancilla_y, 350*scale, marker="h", c="white", edgecolors="black", zorder=2)
        if highlighted_qubits is not None:
            if colors is not None:
                if isinstance(colors, (str, tuple)):
                    colors = [colors]*len(highlighted_qubits)
                elif isinstance(colors, (list, tuple, np.ndarray)):
                    if len(colors) != len(highlighted_qubits):
                        raise ValueError("Length of colors must match length of highlighted_qubits")
                else:
                    raise ValueError("colors must be a string or a list/tuple/ndarray of strings")
            else:
                colors = ["#2f64f7"]*len(highlighted_qubits)
            for qb_ind in highlighted_qubits:
                coords = self.coords[qb_ind]
                x = coords[1]
                y = self.max_y - coords[0]
                color = colors[highlighted_qubits.index(qb_ind)]
                if is_node_coords(coords):
                    plt.scatter(x, y, 350*scale, marker="o", c=color, edgecolors="black", zorder=3)
                elif is_link_coords(coords):
                    plt.scatter(x, y, 400*scale, marker=(4, 0, 45), c=color, edgecolors="black", zorder=3)
                else:
                    plt.scatter(x, y, 350*scale, marker="h", c=color, edgecolors="black", zorder=3)
        if number_qubits:
            start = 0 if first_qubit is None else first_qubit
            labels = [str(start + i) for i in range(len(self.coords))]
            ttransform = mpl.transforms.Affine2D().translate(0, -1*scale)
            for i, (y, x) in enumerate(self.coords):
                text = plt.text(x, self.max_y - y, labels[i], horizontalalignment="center", verticalalignment="center", fontdict={"size": 5.5*scale})
                text.set_transform(text.get_transform() + ttransform)
        plt.axis("off")
        plt.xlim([self.min_x-0.2, self.max_x+0.2])
        plt.ylim([-0.2, self.max_y+0.2])
        # plt.tight_layout()
        if filepath is not None:
            plt.savefig(filepath, dpi=300, facecolor="none")
        plt.rcdefaults()

class HeavyHexLattice(Lattice):
    def __init__(self, plaq_width: int, plaq_height: int) -> None:
        self.plaquettes_width = plaq_width
        self.plaquettes_height = plaq_height
        width=plaq_width*2+1
        height=plaq_height
        link_coords = []
        node_coords = []
        for i in range(height+1):
            for j in range(width+1):
                if i==0 and j == 0:
                    continue
                if i == height:
                    if j == 0 and i%2 != 0:
                        continue
                    if j == width and i%2 == 0:
                        continue
                node_coords.append((i,j))
        for i in range(height+1):
            for j in range(width):
                if i==0 and j == 0:
                    continue
                if i == height:
                    if j == 0 and i%2 != 0:
                        continue
                    if j == width-1 and i%2 == 0:
                        continue
                link_coords.append((i, j+0.5))
        for i in range(height):
            if i%2 == 0:
                for j in range((width+1)//2):
                    link_coords.append((i+0.5, 2*j+1))
            else:
                for j in range(width//2+1):
                    link_coords.append((i+0.5, 2*j))
        super().__init__(node_coords + link_coords)

    @staticmethod
    def get_backend_coordinates(backend: BackendV2 | str) -> list[tuple[int, int]]:
        """Returns a list with the qubit coordinates of the specified backend
        """
        backend_name = backends_objs_to_names(backend)
        ibm_qubit_coords = []
        if backend_name in ["ibm_fez", "ibm_marrakesh", "ibm_aachen", "ibm_kingston", "ibm_basquecountry"]:
            for i in range(8):
                for j in range(16):
                    ibm_qubit_coords.append((2*i, j))
            for i in range(4):
                for j in range(4):
                    ibm_qubit_coords.append((4*i+1, 4*j+3))
            for i in range(3):
                for j in range(4):
                    ibm_qubit_coords.append((4*i+3, 4*j+1))
        elif backend_name == 'ibm_torino':
            for i in range(7):
                for j in range(15):
                    ibm_qubit_coords.append((2*i, j))
            for i in range(4):
                for j in range(4):
                    ibm_qubit_coords.append((4*i+1, 4*j))
            for i in range(3):
                for j in range(4):
                    ibm_qubit_coords.append((4*i+3, 4*j+2))
        else: # Eagle r3
            for i in range(7):
                for j in range(15):
                    if i == 0 and j == 14:
                        continue
                    if i == 6 and j == 0:
                        continue
                    ibm_qubit_coords.append((2*i, j))
            for i in range(3):
                for j in range(4):
                    ibm_qubit_coords.append((4*i+1, 4*j))
            for i in range(3):
                for j in range(4):
                    ibm_qubit_coords.append((4*i+3, 4*j+2))
        return sorted(ibm_qubit_coords)

    def get_all_layouts(self, backend: BackendV2 | str, exclude_automorphisms: bool = False) -> list[list[int]]:
        """Returns a list of all possible mappings from own qubit indices to ibm physical qubits
        Possible isomorphisms are reflections and 60º rotations
        Arguments:
            backend:
                Device backend
        """
        # Gets the new rows and columns after a 60º rotation
        # A bit cumbersome due to the brick-wall layout instead of regular hexagons
        def rotated_row(r: float, c: float) -> float:
            if r != int(r):
                return int(c)//2+1+int(r-0.5)//2
            return [0,0.5,1,1][int(2*r+2*c)%4]+int(r)//2+int(2*(c+r%2))//4
        def rotated_column(r: float, c: float) -> float:
            if r != int(r):
                return -(int(c)//2)+0.5+3*(int(r-0.5)//2)+2*(int(r-0.5)%2)
            return [0,0,0,-0.5][int(2*r+2*c)%4]+int(3*r)//2-(int(2*(c-r%2))//4)
        layouts = []
        sorted_layouts = []
        ibm_qubit_coords = HeavyHexLattice.get_backend_coordinates(backends_objs_to_names(backend))
        for (row_ibm, col_ibm) in ibm_qubit_coords:
            for dir_col in [1, -1]: # left-right mirroring
                for dir_row in [1, -1]: # up-down mirroring
                    layout = []
                    fits = False
                    if (row_ibm+dir_row, col_ibm) in ibm_qubit_coords:
                        if exclude_automorphisms and dir_row == -1: # skip up-down mirroring as it is an automorphism
                            continue
                        fits = True
                        (row_orig, col_orig) = self.coords[0] # Coordinates of qubit 0
                        for (row_lattice, col_lattice) in self.coords:
                            offset = (int(2*(row_lattice-row_orig)), int(2*(col_lattice-col_orig)))
                            ibm_c = (row_ibm+dir_row*offset[0], col_ibm+dir_col*offset[1])
                            if ibm_c not in ibm_qubit_coords:
                                fits = False
                                break
                            layout.append(ibm_qubit_coords.index(ibm_c))
                    elif (row_ibm+dir_row, col_ibm+2*dir_col) in ibm_qubit_coords: # 60º rotated layouts
                        fits = True
                        (row_orig, col_orig) = self.coords[0] # Coordinates of qubit 0
                        (row_orig, col_orig) = (rotated_row(row_orig, col_orig), rotated_column(row_orig, col_orig))
                        for (row_lattice, col_lattice) in self.coords:
                            (row_lattice, col_lattice) = rotated_row(row_lattice, col_lattice), rotated_column(row_lattice, col_lattice)
                            offset = (int(2*(row_lattice-row_orig)), int(2*(col_lattice-col_orig)))
                            ibm_c = (row_ibm+dir_row*offset[0], col_ibm+dir_col*offset[1])
                            if ibm_c not in ibm_qubit_coords:
                                fits = False
                                break
                            layout.append(ibm_qubit_coords.index(ibm_c))
                    if fits:
                        if exclude_automorphisms:
                            sort = sorted(layout)
                            if sort not in sorted_layouts:
                                layouts.append(layout)
                                sorted_layouts.append(sort)
                        elif layout not in layouts:
                            layouts.append(layout)
        return layouts

class HeronHeavyHex(Lattice):
    def __init__(self, plaq_width: int, plaq_height: int) -> None:
        self.plaquettes_width = plaq_width
        self.plaquettes_height = plaq_height
        width=plaq_width*2+1
        height=plaq_height
        link_coords = []
        node_coords = []
        for i in range(height+1):
            for j in range(0, width+1):
                node_coords.append((i,j))
        for i in range(height+1):
            for j in range(-1, width):
                link_coords.append((i, j+0.5))
        for i in range(height):
            if i%2 == 0:
                for j in range((width+1)//2):
                    link_coords.append((i+0.5, 2*j+1))
            else:
                for j in range(width//2+1):
                    link_coords.append((i+0.5, 2*j))
        super().__init__(node_coords + link_coords)

class SquareLattice(Lattice):
    def __init__(self, nodes_x: int, nodes_y: int) -> None:
        self.nodes_x = nodes_x
        self.nodes_y = nodes_y
        coords = []
        for i in range(nodes_y):
            for j in range(nodes_x):
                coords.append((i,j))
        super().__init__(coords)

class LGTSquareLattice(Lattice):
    def __init__(self, plaquette_width: int, plaquette_height: int) -> None:
        self.plaquette_width = plaquette_width
        self.plaquette_height = plaquette_height
        coords = []
        for x in np.arange(0, plaquette_width+0.5, 0.5):
            for y in np.arange(0, plaquette_height+0.5, 0.5):
                coords.append((y,x))
        super().__init__(coords)