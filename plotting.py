from qiskit.circuit import BoxOp, Clbit, ControlFlowOp, ControlledGate, Measure, Operation, SwitchCaseOp
from hardware.utils.geometry import Lattice, is_node_coords, is_link_coords
# Qiskit's own mpl drawer style and gate labels, so colors and texts match circuit.draw("mpl")
from qiskit.visualization.circuit._utils import get_gate_ctrl_text
from qiskit.visualization.circuit.qcstyle import MPLDefaultStyle
from qiskit.circuit.controlflow import condition_resources
from matplotlib.patheffects import Normal, Stroke
from matplotlib.transforms import Affine2D
from hardware.utils.circs import depth2qb
from qiskit.circuit.classical import expr
from matplotlib.typing import MarkerType
from typing import Literal, NamedTuple
from matplotlib.figure import Figure
from IPython.display import Image
from qiskit import QuantumCircuit
from matplotlib.axes import Axes
from matplotlib.path import Path
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import math
import os
import io

def luminance(*rgb: float) -> float:
    """Calcula la luminancia relativa de un color."""
    rgb = list(rgb)
    for i in range(3):
        if rgb[i] <= 0.03928:
            rgb[i] = rgb[i] / 12.92
        else:
            rgb[i] = ((rgb[i] + 0.055) / 1.055) ** 2.4
    return rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722

def contrast_ratio(l1: float, l2: float) -> float:
    """Calcula la relación de contraste entre dos luminancias."""
    if l1 > l2:
        return (l1 + 0.05) / (l2 + 0.05)
    else:
        return (l2 + 0.05) / (l1 + 0.05)

def best_text_color(*rgb: float) -> Literal["white", "black"]:
    """Determina si el texto blanco o negro se verá mejor sobre un color dado."""
    lum = luminance(*rgb)
    contrast_with_white = contrast_ratio(lum, luminance(1, 1, 1))
    contrast_with_black = contrast_ratio(lum, luminance(0, 0, 0))
    
    if contrast_with_white > contrast_with_black:
        return "white"
    else:
        return "black"

def plot_circuit_arr_depth(circuit_arr, x_ticks=None, x_label=None, filepath=""):
    plt.rc("text", usetex=True)
    plt.rc("font", size=20, family="serif", weight="bold")

    fig, ax = plt.subplots(figsize=[9, 6])
    
    x = np.arange(1, len(circuit_arr)+1)
    y = [depth2qb(circ) for circ in circuit_arr]
    color = plt.get_cmap("Set2")((1 % 8)/8 + 0.01)
    plt.plot(x, y, "o--", markersize=12, markeredgecolor="black", color=color)
    if x_label is None:
        plt.xlabel("Circuit index", labelpad=10)
    else:
        plt.xlabel(x_label, labelpad=10)
    if x_ticks is not None:
        plt.xticks(x_ticks)
    plt.ylabel("Depth")
    plt.grid(color="gray", linestyle="dashdot", linewidth=1.6, zorder=0)
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, dpi=300, facecolor="none")
    plt.show()
    plt.rcdefaults()

def plot_many_lines(x, ys, x_label="", x_ticks=None, y_label="", y_ticks=None, labels=None, filepath=""):
    x = np.array(x)
    ys = np.array(ys)
    if x.ndim != 1:
        raise ValueError("x must be a 1D array-like")
    if x.shape[0] != ys.shape[1]:
        raise ValueError("x and ys must have compatible shapes")
    if ys.ndim > 2:
        raise ValueError("ys must be a 1D or 2D array-like")
    
    plt.rc("text", usetex=True)
    plt.rc("font", size=18, family="serif", weight="bold")

    fig, ax = plt.subplots(figsize=[9, 6])
    
    nlines = len(ys)
    for i, y in enumerate(ys):
        color = plt.get_cmap("Set2")((i % 8)/8)
        plt.plot(x, y, "o--", markersize=12, linewidth=2, markeredgecolor="black", color=color, label=labels[i] if labels is not None else None)
    plt.xlabel(x_label, labelpad=10)
    plt.ylabel(y_label)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    if labels is not None:
        plt.legend()
    plt.grid(color="gray", linestyle="dashdot", linewidth=1.6, zorder=0)
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, dpi=300, facecolor="none")
    plt.show()
    plt.rcdefaults()

def matrix(matrix_to_plot, cmap="inferno", xlabel="Row", ylabel="Column", xticks=None, yticks=None, cbar_range=None, clabel=r"$\left|\left\langle u(t_a) \right| \left. u(t_b) \right\rangle\right|^2$", filepath=""):
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", size=18, weight="bold")
    xs = np.arange(matrix_to_plot.shape[1])
    ys = np.arange(matrix_to_plot.shape[0])
    X, Y = np.meshgrid(xs, ys)
    if cbar_range is None:
        vmin, vmax = None, None
    else:
        vmin, vmax = cbar_range
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.pcolormesh(X, Y, matrix_to_plot, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.gca().invert_yaxis()
    plt.colorbar(label=clabel)
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, dpi=300, facecolor="none")
    plt.show()
    plt.rcdefaults()

def samples_histogram(samples, xlabel="Basis state", ylabel="Counts", filepath=""):
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", size=18)
    
    plt.plot([int(k) for k in samples.keys()], list(samples.values()), "o", markersize=8, markeredgecolor="black", color=plt.get_cmap("Set2")(0))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.rcdefaults()
    if filepath:
        plt.savefig(filepath, dpi=300, facecolor="none")
    plt.show()

# ----------------------------- Circuit sketches on the lattice ----------------------------- #
# Draw the steps of a circuit on a Lattice, as a single step, a grid of steps or a gif

_STYLE = MPLDefaultStyle().style

class _Instruction(NamedTuple):
    operation: Operation
    qubits: tuple[int, ...]
    writes: tuple[int, ...] # Clbits written (e.g. by a measurement)
    reads: tuple[int, ...] # Clbits read by the enclosing control flow
    conditional: bool
    measured: tuple[int, ...] # Qubits whose measurements set the condition of the enclosing control flow

def _condition_clbits(op: ControlFlowOp) -> tuple[Clbit, ...]:
    """Clbits of the condition of an if_test, while_loop or switch, as in MatplotlibDrawer._condition. None for a for_loop"""
    if isinstance(op, SwitchCaseOp):
        condition = op.target if isinstance(op.target, expr.Expr) else (op.target, 0)
    else:
        condition = getattr(op, "condition", None)
    return () if condition is None else condition_resources(condition).clbits

def _flatten(circuit: QuantumCircuit, qubits: tuple[int, ...], clbits: tuple[int, ...], reads: tuple[int, ...] = (), conditional: bool = False, measured: tuple[int, ...] = (), measurer: dict[int, int] | None = None) -> list[_Instruction]:
    """Inlines the first block of every control flow operation, mapping all bits to the top-level circuit indices.
    Boxes are inlined as regular operations, the body of any other control flow is marked as conditional.
    measurer maps each clbit to the qubit of its last measurement so far, and is updated in place
    """
    measurer = {} if measurer is None else measurer
    instructions = []
    for ci in circuit.data:
        op_qubits = tuple(qubits[circuit.find_bit(q).index] for q in ci.qubits)
        op_clbits = tuple(clbits[circuit.find_bit(c).index] for c in ci.clbits)
        if isinstance(ci.operation, ControlFlowOp):
            if isinstance(ci.operation, BoxOp):
                instructions += _flatten(ci.operation.blocks[0], op_qubits, op_clbits, reads, conditional, measured, measurer)
            else:
                condition = [clbits[circuit.find_bit(c).index] for c in _condition_clbits(ci.operation)]
                block_measured = tuple(set(measured) | {measurer[c] for c in condition if c in measurer})
                instructions += _flatten(ci.operation.blocks[0], op_qubits, op_clbits, tuple(set(reads) | set(op_clbits)), True, block_measured, measurer)
        else:
            if isinstance(ci.operation, Measure):
                measurer.update(zip(op_clbits, op_qubits))
            instructions.append(_Instruction(ci.operation, op_qubits, op_clbits, reads, conditional, measured))
    return instructions

def circuit_steps(circuit: QuantumCircuit) -> list[list[_Instruction]]:
    """Splits the circuit into steps of operations acting on disjoint qubits, scheduling every operation as soon as possible.
    Barriers are not drawn, but operations after a barrier start after all the operations before it
    """
    qubit_free = [0]*circuit.num_qubits # First step at which each qubit is free
    write_done = [0]*circuit.num_clbits # First step after the last write of each clbit
    read_done = [0]*circuit.num_clbits # First step after the last read of each clbit
    steps = []
    for instr in _flatten(circuit, tuple(range(circuit.num_qubits)), tuple(range(circuit.num_clbits))):
        if instr.operation.name == "barrier":
            sync = max(qubit_free[q] for q in instr.qubits)
            for q in instr.qubits:
                qubit_free[q] = sync
            continue
        if not instr.qubits:
            continue
        step = max([qubit_free[q] for q in instr.qubits] + [write_done[c] for c in instr.writes + instr.reads] + [read_done[c] for c in instr.writes])
        if step == len(steps):
            steps.append([])
        steps[step].append(instr)
        for q in instr.qubits:
            qubit_free[q] = step + 1
        for c in instr.writes:
            write_done[c] = step + 1
        for c in instr.reads:
            read_done[c] = max(read_done[c], step + 1)
    return steps

def _gate_colors(op: Operation, raw_gate_text: str) -> tuple[str, str]:
    """(face color, text color) of the gate, following MatplotlibDrawer._get_colors for the iqp style"""
    color = _STYLE["dispcol"].get(raw_gate_text, _STYLE["dispcol"].get(op.name))
    base_name = getattr(getattr(op, "base_gate", None), "name", None)
    if color is None and base_name in ["x", "dcx", "swap"]:
        color = _STYLE["dispcol"][base_name]
    if color is None:
        return _STYLE["gc"], _STYLE["gt"]
    if isinstance(color, str):
        return color, _STYLE["gt"]
    return color[0], color[1]

def _meter_marker() -> Path:
    # Arc of the measurement symbol plus its needle, centered at the origin
    arc = Path.arc(0, 180).transformed(Affine2D().translate(0, -0.5))
    needle = Path([(0, -0.5), (0.6, 0.4)])
    return Path.make_compound_path(arc, needle)

class _StepDrawer:
    def __init__(self, ax: Axes, lattice: Lattice, scale: float) -> None:
        self.ax = ax
        self.lattice = lattice
        self.scale = scale

    def xy(self, qubit: int) -> tuple[float, float]:
        coords = self.lattice.coords[qubit]
        return coords[1], self.lattice.max_y - coords[0]

    def marker(self, qubit: int) -> tuple[MarkerType, float]:
        # Same markers as Lattice.plot
        coords = self.lattice.coords[qubit]
        if is_node_coords(coords):
            return "o", 350*self.scale
        if is_link_coords(coords):
            return (4, 0, 45), 400*self.scale
        return "h", 350*self.scale

    def qubit(self, qubit: int, facecolor: str, conditional: bool, zorder: float = 3) -> None:
        marker, size = self.marker(qubit)
        self.ax.scatter(*self.xy(qubit), size, marker=marker, c=facecolor, edgecolors="black", zorder=zorder, clip_on=False)
        if conditional:
            self.ax.scatter(*self.xy(qubit), 1.7*size, marker=marker, facecolors="none", edgecolors="black", linestyles="--", zorder=zorder, clip_on=False)

    def symbol(self, qubit: int, marker: MarkerType, rel_size: float, color: str, **kwargs) -> None:
        self.ax.scatter(*self.xy(qubit), rel_size*self.marker(qubit)[1], marker=marker, c=color, zorder=4, clip_on=False, **kwargs)

    def label(self, qubit: int, text: str, color: str) -> None:
        self.ax.text(*self.xy(qubit), text, horizontalalignment="center", verticalalignment="center", color=color, fontsize=6*self.scale, fontfamily="serif", zorder=5)

    def line(self, qubit1: int, qubit2: int, color: str, conditional: bool) -> None:
        (x1, y1), (x2, y2) = self.xy(qubit1), self.xy(qubit2)
        self.ax.plot([x1, x2], [y1, y2], color=color, linewidth=1.5*self.scale, linestyle="--" if conditional else "-", zorder=2.5)

    def classical_line(self, qubit1: int, qubit2: int) -> None:
        # Double line of Qiskit's classical wires, drawn as a wide line with a white core, below the quantum lines.
        # Not snapped to the pixel grid, which shifts the line and its core differently and makes it look off-center
        (x1, y1), (x2, y2) = self.xy(qubit1), self.xy(qubit2)
        self.ax.plot([x1, x2], [y1, y2], color="white", linewidth=1.5*self.scale, zorder=2.4, snap=False, path_effects=[Stroke(linewidth=3.5*self.scale, foreground="black"), Normal()])

    def instruction(self, instr: _Instruction) -> None:
        op, qubits, cond = instr.operation, instr.qubits, instr.conditional
        gate_text, _, raw_gate_text = get_gate_ctrl_text(op, "mpl", style=_STYLE)
        fc, gt = _gate_colors(op, raw_gate_text)
        # Each measured qubit of the condition is linked to the nearest qubit of the operation
        for m in instr.measured:
            if m not in qubits:
                self.classical_line(m, min(qubits, key=lambda q: math.dist(self.xy(m), self.xy(q))))
        if isinstance(op, Measure):
            self.qubit(qubits[0], fc, cond)
            self.symbol(qubits[0], _meter_marker(), 0.5, "none", edgecolors=gt, linewidths=0.6*self.scale)
        elif isinstance(op, ControlledGate):
            controls, targets = qubits[:op.num_ctrl_qubits], qubits[op.num_ctrl_qubits:]
            for c in controls:
                self.line(c, targets[0], fc, cond)
            for t1, t2 in zip(targets, targets[1:]):
                self.line(t1, t2, fc, cond)
            # Bit k of ctrl_state is the state of the k'th control. Open controls are drawn hollow
            for k, c in enumerate(controls):
                self.qubit(c, "white", cond)
                closed = (op.ctrl_state >> k) & 1
                self.symbol(c, "o", 0.25, fc if closed else "white", edgecolors=fc, linewidths=0.8*self.scale)
            base_name = op.base_gate.name
            kind = base_name if base_name == "swap" or (base_name in ["x", "z"] and len(targets) == 1) else "gate"
            for t in targets:
                self.target(t, kind, gate_text, fc, gt, cond)
        else:
            for q1, q2 in zip(qubits, qubits[1:]):
                self.line(q1, q2, fc, cond)
            for q in qubits:
                self.target(q, "swap" if op.name == "swap" else "gate", gate_text, fc, gt, cond)

    def target(self, qubit: int, kind: str, gate_text: str, fc: str, gt: str, conditional: bool) -> None:
        match kind:
            case "x": # Target of a controlled X
                self.qubit(qubit, fc, conditional)
                self.symbol(qubit, "+", 0.6, _STYLE["dispcol"]["target"][0], linewidths=1.2*self.scale)
            case "z": # Target of a controlled Z, same as a control
                self.qubit(qubit, "white", conditional)
                self.symbol(qubit, "o", 0.25, fc)
            case "swap":
                self.qubit(qubit, "white", conditional)
                self.symbol(qubit, "x", 0.5, fc, linewidths=1.2*self.scale)
            case _:
                self.qubit(qubit, fc, conditional)
                self.label(qubit, gate_text, gt)

def _draw_step(lattice: Lattice, steps: list[list[_Instruction]], i: int, scale: float, number_qubits: bool, title: bool = True) -> Figure:
    if not -len(steps) <= i < len(steps):
        raise IndexError(f"Step {i} out of range for a circuit with {len(steps)} steps")
    lattice.plot(scale=scale, number_qubits=number_qubits)
    fig, ax = plt.gcf(), plt.gca()
    if title:
        # Extra band on top of the figure for the title, keeping the size of the lattice
        band = 0.25*scale
        width, height = fig.get_size_inches()
        pos = ax.get_position()
        fig.set_size_inches(width, height + band)
        ax.set_position([pos.x0, pos.y0*height/(height + band), pos.width, pos.height*height/(height + band)])
    drawer = _StepDrawer(ax, lattice, scale)
    for instr in steps[i]:
        drawer.instruction(instr)
    # Idle qubits are redrawn to hide the lines of long-range gates passing below them (but not their numbers)
    active = {q for instr in steps[i] for q in instr.qubits}
    for q in set(range(len(lattice))) - active:
        drawer.qubit(q, "white", False, zorder=2.75)
    if title:
        ax.set_title(f"Step {i % len(steps)}", fontfamily="serif", fontsize=8*scale)
    return fig

def _check_sizes(lattice: Lattice, circuit: QuantumCircuit) -> None:
    if circuit.num_qubits > len(lattice):
        raise ValueError(f"Circuit has {circuit.num_qubits} qubits but the lattice only {len(lattice)}")

def plot_step(lattice: Lattice, circuit: QuantumCircuit, i: int, scale: float = 1.5, number_qubits: bool = False, filepath: str | os.PathLike | None = None) -> None:
    """Plots the operations of the i'th step of the circuit on the lattice, with circuit qubit j placed at lattice.coords[j]
    Steps are those of circuit_steps. Operations inside control flow (e.g. if_test) are drawn with dashed outlines,
    and joined by a classical double line to the qubits whose measurements set their condition
    """
    _check_sizes(lattice, circuit)
    fig = _draw_step(lattice, circuit_steps(circuit), i, scale, number_qubits)
    if filepath is not None:
        fig.savefig(filepath, dpi=300, facecolor="none")

def _render_frames(lattice: Lattice, circuit: QuantumCircuit, step_range: range | None, scale: float, number_qubits: bool, dpi: int, titles: bool = True) -> list[PIL.Image.Image]:
    _check_sizes(lattice, circuit)
    steps = circuit_steps(circuit)
    indices = range(len(steps)) if step_range is None else step_range
    if not indices:
        raise ValueError("No steps to plot")
    frames = []
    for i in indices:
        fig = _draw_step(lattice, steps, i, scale, number_qubits, titles)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=dpi, facecolor="white")
        plt.close(fig)
        frames.append(PIL.Image.open(buffer).convert("RGB"))
    return frames

def plot_steps_grid(lattice: Lattice, circuit: QuantumCircuit, step_range: range | None = None, ncols: int | None = None, scale: float = 1.5, number_qubits: bool = False, titles: bool = True, dpi: int = 200, filepath: str | os.PathLike | None = None) -> None:
    """Plots the steps of the circuit side by side in a single figure, each one as drawn by plot_step
    Arguments:
        step_range:
            Steps to plot, e.g. range(2, 10). All of them if None
        ncols:
            Number of columns of the grid. If None, the grid is as square as possible
        titles:
            Whether to show the "Step i" title above each step
    """
    frames = _render_frames(lattice, circuit, step_range, scale, number_qubits, dpi, titles)
    ncols = math.ceil(math.sqrt(len(frames))) if ncols is None else ncols
    nrows = math.ceil(len(frames)/ncols)
    width, height = frames[0].size
    # One figure pixel per frame pixel when saved at the same dpi
    fig, axes = plt.subplots(nrows, ncols, figsize=[ncols*width/dpi, nrows*height/dpi], squeeze=False)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    for ax in axes.flat:
        ax.axis("off")
    for ax, frame in zip(axes.flat, frames):
        ax.imshow(frame)
    if filepath is not None:
        fig.savefig(filepath, dpi=dpi, facecolor="white")

def plot_circuit(lattice: Lattice, circuit: QuantumCircuit, step_range: range | None = None, duration: int = 600, scale: float = 1.5, number_qubits: bool = False, dpi: int = 200, filepath: str | os.PathLike | None = None) -> Image:
    """Returns a gif with one frame per step of the circuit, as drawn by plot_step
    Arguments:
        step_range:
            Steps to include, e.g. range(2, 10). All of them if None
        duration:
            Milliseconds each frame is shown
        filepath:
            If given, the gif is also saved there
    """
    frames = _render_frames(lattice, circuit, step_range, scale, number_qubits, dpi)
    gif = io.BytesIO()
    frames[0].save(gif, format="GIF", save_all=True, append_images=frames[1:], duration=duration, loop=0)
    if filepath is not None:
        with open(filepath, "wb") as f:
            f.write(gif.getvalue())
    return Image(data=gif.getvalue(), format="gif")
