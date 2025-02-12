from __future__ import annotations

import functools
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Protocol

from hugr import Wire, Hugr, Node, ops, tys
from hugr.std.float import FLOAT_OPS_EXTENSION, FLOAT_T

from hugr_phase_fold.analysis import Phase, SimplePhase, LoopHoistedPhase, Analysis
from hugr_phase_fold.util import (
    load_float_const,
    remove_gate,
    replace_gate,
    outgoing_qubits,
    insert_gate,
)

from tket2.circuit.build import Rz, CX, PauliX, OneQbGate

S = OneQbGate("S")
Sdg = OneQbGate("Sdg")
T = OneQbGate("T")

FNeg = FLOAT_OPS_EXTENSION.get_op("fneg").instantiate([])
FAdd = FLOAT_OPS_EXTENSION.get_op("fadd").instantiate([])
FSub = FLOAT_OPS_EXTENSION.get_op("fsub").instantiate([])


class PhaseAccumulator(Protocol):
    def add_phase(
        self, angle: Fraction | Wire, hugr: Hugr, parent: Node, negate: bool = False
    ) -> PhaseAccumulator: ...

    def to_wire(self, parent: Node, hugr: Hugr) -> Wire: ...


@dataclass(frozen=True)
class StaticPhaseAccumulator(PhaseAccumulator):
    angle: Fraction = field(default_factory=Fraction)

    def add_phase(
        self, angle: Fraction | Wire, hugr: Hugr, parent: Node, negate: bool = False
    ) -> PhaseAccumulator:
        if isinstance(angle, Fraction):
            if negate:
                angle = -angle
            return StaticPhaseAccumulator(self.angle + angle)
        else:
            if negate:
                neg = hugr.add_node(FNeg, parent, 1)
                hugr.add_link(angle.out_port(), neg.inp(0))
                angle = neg.out(0)
            return DynamicPhaseAccumulator(angle, self.angle)

    def to_wire(self, parent: Node, hugr: Hugr) -> Wire:
        return load_float_const(self.angle, parent, hugr)


@dataclass(frozen=True)
class DynamicPhaseAccumulator(PhaseAccumulator):
    wire: Wire
    static_angle: Fraction = field(default_factory=Fraction)
    force_dynamic: bool = False

    def add_phase(
        self, angle: Fraction | Wire, hugr: Hugr, parent: Node, negate: bool = False
    ) -> PhaseAccumulator:
        if isinstance(angle, Fraction):
            if self.force_dynamic:
                if negate:
                    angle = -angle
                angle_float = angle.numerator / angle.denominator
                angle = load_float_const(angle_float, parent, hugr)
                return self.add_phase(angle, hugr, parent, negate)
            else:
                return DynamicPhaseAccumulator(
                    self.wire, self.static_angle + angle, self.force_dynamic
                )
        else:
            op = FSub if negate else FAdd
            node = hugr.add_node(op, parent, 1)
            hugr.add_link(self.wire.out_port(), node.inp(0))
            hugr.add_link(angle.out_port(), node.inp(1))
            return DynamicPhaseAccumulator(
                node.out(0), self.static_angle, self.force_dynamic
            )

    def to_wire(self, parent: Node, hugr: Hugr) -> Wire:
        if self.static_angle == 0:
            return self.wire
        else:
            angle_float = self.static_angle.numerator / self.static_angle.denominator
            angle = load_float_const(angle_float, parent, hugr)
            add = hugr.add_node(FAdd, parent, 1)
            hugr.add_link(self.wire.out_port(), add.inp(0))
            hugr.add_link(angle.out_port(), add.inp(1))
            return add.out(0)


class PhaseFolder:
    hugr: Hugr

    def __init__(self, hugr: Hugr):
        self.hugr = hugr

    def run(self, parent: Node) -> None:
        a, _ = Analysis.run(self.hugr, parent)
        stack = [a]
        while stack:
            a = stack.pop()
            for phases in a.phases.values():
                self.fold_phases(phases)
            stack += a.nested_analysis.values()

    def fold_phases(self, phases: list[Phase]):
        assert len(phases) > 0
        *to_remove, last = phases
        acc = StaticPhaseAccumulator()
        for phase in to_remove:
            acc = self.remove_and_acc(phase, acc)
        self.replace_with_acc(last, acc)

    @functools.singledispatchmethod
    def remove_and_acc(self, phase: Phase, acc: PhaseAccumulator) -> PhaseAccumulator:
        """Removes a phase gate and updates the accumulator with its angle."""
        assert False, "Unreachable"

    @remove_and_acc.register
    def _remove_and_acc_simple_phase(
        self, phase: SimplePhase, acc: PhaseAccumulator
    ) -> PhaseAccumulator:
        parent = self.hugr[phase.node].parent
        remove_gate(phase.node, self.hugr)
        return acc.add_phase(phase.angle, self.hugr, parent, negate=bool(phase.parity))

    @remove_and_acc.register
    def _remove_and_acc_loop_hoisted_phase(
        self, phase: LoopHoistedPhase, acc: PhaseAccumulator
    ) -> PhaseAccumulator:
        loop = self.hugr[phase.loop_node]

        # Make a wire for the accumulator if we don't already have one
        if isinstance(acc, DynamicPhaseAccumulator):
            acc_wire = acc.wire
        else:
            acc_wire = load_float_const(0, loop.parent, self.hugr)

        # Add it as an extra input to the loop
        assert isinstance(loop.op, ops.TailLoop)
        loop.op.rest.append(FLOAT_T)
        num_vars = len(loop.op.rest)
        self.hugr.add_link(acc_wire.out_port(), phase.loop_node.inp(num_vars - 1))

        # Also add it to the loop input
        inp_node, out_node, *_ = loop.children
        inp, out = self.hugr[inp_node], self.hugr[out_node]
        assert isinstance(inp.op, ops.Input)
        assert isinstance(out.op, ops.Output)
        inp.op.types.append(FLOAT_T)
        nested_acc_wire = inp_node.out(num_vars - 1)

        # Visit the inside of the loop
        nested_acc = DynamicPhaseAccumulator(nested_acc_wire, force_dynamic=True)
        for inner in phase.to_hoist:
            nested_acc = self.remove_and_acc(inner, nested_acc)

        # Add accumulator as an extra loop output
        assert isinstance(nested_acc, DynamicPhaseAccumulator)
        self.hugr.add_link(nested_acc.wire.out_port(), out_node.inp(num_vars))
        acc_wire = phase.loop_node.out(num_vars - 1)

        static_angle = acc.angle if isinstance(acc, StaticPhaseAccumulator) else 0
        return DynamicPhaseAccumulator(acc_wire, static_angle)

    @functools.singledispatchmethod
    def replace_with_acc(self, phase: Phase, acc: PhaseAccumulator) -> None:
        """Replaces a phase gate with the accumulated angle."""
        assert False, "Unreachable"

    @replace_with_acc.register
    def _replace_simple_phase_with_acc(
        self, phase: SimplePhase, acc: PhaseAccumulator
    ) -> None:
        parent = self.hugr[phase.node].parent
        acc = acc.add_phase(phase.angle, self.hugr, parent)
        if isinstance(acc, StaticPhaseAccumulator) and (
            gates := gates_from_angle(acc.angle)
        ):
            # Replace with discrete gates
            first, *rest = gates
            g = replace_gate(phase.node, first, self.hugr)
            for gate in rest:
                insert_gate(gate, [g.out(0)], parent, self.hugr)
        else:
            # Replace with an Rz gate
            rz = replace_gate(phase.node, Rz, self.hugr)
            self.hugr.add_link(acc.to_wire(parent, self.hugr).out_port(), rz.inp(1))

    @replace_with_acc.register
    def _replace_hoisted_loop_phase_with_acc(
        self, phase: LoopHoistedPhase, acc: PhaseAccumulator
    ) -> None:
        parent = self.hugr[phase.loop_node].parent
        acc = self._remove_and_acc_loop_hoisted_phase(phase, acc)
        # We need to insert a CX ladder after the loop to reconstruct the parity for the
        # hoisted phase
        qs = list(outgoing_qubits(phase.loop_node, self.hugr))
        ones = [i for i, x in enumerate(phase.inner_eqn) if x]
        *rest, tgt = ones
        for q in rest:
            qs[q], qs[tgt] = insert_gate(CX, [qs[q], qs[tgt]], parent, self.hugr)
        # Add an Rz gate with the accumulated phase
        (qs[tgt],) = (rz,) = insert_gate(Rz, [qs[tgt]], parent, self.hugr)
        self.hugr.add_link(
            acc.to_wire(parent, self.hugr).out_port(), rz.out_port().node.inp(1)
        )
        # Uncompute the CX ladder
        for q in reversed(rest):
            qs[q], qs[tgt] = insert_gate(CX, [qs[q], qs[tgt]], parent, self.hugr)


def gates_from_angle(angle: Fraction) -> list[ops.Op] | None:
    if angle.denominator == 1:
        if angle.numerator % 2 == 0:
            return [ops.Noop(tys.Qubit)]
        else:
            return [PauliX]
    elif angle.denominator == 2:
        match angle.numerator:
            case 1:
                return [S]
            case 3:
                return [Sdg]
            case _:
                assert False
    elif angle.denominator == 4:
        match angle.numerator % 8:
            case 1:
                return [T]
            case 3:
                return [S, T]
            case 5:
                return [PauliX, T]
            case 7:
                return [Sdg, T]
            case _:
                assert False
    return None
