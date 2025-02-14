from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from typing import NamedTuple, TypeAlias

import numpy as np
from galois import GF2
from hugr import Hugr, Node, Wire, ops
from hugr.ops import Op

from hugr_phase_fold.domain import Domain
from hugr_phase_fold.util import incoming_qubits, outgoing_qubits, toposort


QubitId: TypeAlias = int
HashableGF2Vec = tuple[bool, ...]


Phase: TypeAlias = "SimplePhase | LoopHoistedPhase"


class SimplePhase(NamedTuple):
    node: Node
    parity: bool
    angle: Fraction | Wire


class LoopHoistedPhase(NamedTuple):
    loop_node: Node
    to_hoist: list[Phase]
    inner_eqn: GF2
    parity: bool


@dataclass
class Analysis:
    #: Affine equations describing the current state of each qubit x'_i as a function of
    #: the initial states x_j, intermediate variables y_j, and an offset bit c. E.g.
    #:
    #:    x1' = x1 + x3 + x6 + c1
    #:    x2' = x1 + x2 + c2
    #:    x3' = x2 + x5 + x6 + x7 + c4
    #:
    #: We encode these equations as bitvectors where the lowest bit 0 represents the
    #: offset c_j and the remaining bits toggle the various qubits and intermediates:
    #:
    #:    E = [ X | Y | c ]
    #:
    #: In other words, it's a GF2 matrix of shape [qubits, qubits + intermediates + 1].
    equation: GF2

    #: Mapping from affine equations to phases that act on those states. This map only
    #: includes phases of nodes that act in the current DFG or ones that can be hoisted
    #: into it.
    phases: defaultdict[HashableGF2Vec, list[Phase]]

    nested_analysis: dict[Node, Analysis]

    hugr: Hugr[Op]

    num_qubits: int
    num_tmps: int

    def __init__(self, num_qubits: int, hugr: Hugr[Op]):
        self.num_qubits = num_qubits
        self.num_tmps = 0
        self.hugr = hugr
        # Initialise as [ I | 0 ], i.e. `x_i' = x_i` for all i
        self.equation = np.hstack(
            (
                GF2.Identity(num_qubits),
                GF2.Zeros((num_qubits, 1)),
            )
        )
        self.phases = defaultdict(list)
        self.nested_analysis = {}

    @staticmethod
    def run(hugr: Hugr[Op], parent: Node) -> tuple[Analysis, list[QubitId]]:
        inp, *_ = hugr.children(parent)
        qs = list(range(len(list(outgoing_qubits(inp, hugr)))))
        analysis = Analysis(len(qs), hugr)
        outs = analysis.apply_dfg(qs, parent)
        return analysis, outs

    def to_domain(self) -> Domain:
        """Turns the gathered information into an instance of our domain."""
        # Construct [ X' | X | Y | c ] where X' is just an identity diagonal
        rel = np.hstack((GF2.Identity(self.num_qubits), self.equation))
        return Domain(rel, self.num_qubits)

    def new_qubit(self) -> int:
        # Insert a new column
        columns = (
            self.equation[:, : self.num_qubits],
            GF2.Zeros([self.num_qubits, 1]),
            self.equation[:, self.num_qubits :],
        )
        # Qubit allocated in zero state, so the new row should be all zeroes
        new_row = GF2.Zeros([1, self.num_qubits + self.num_tmps + 2])
        self.equation = np.vstack((np.hstack(columns), new_row))
        self.num_qubits += 1
        return self.num_qubits - 1

    def new_tmp(self) -> int:
        inserted = (
            self.equation[:, :-1],
            GF2.Zeros([self.num_qubits, 1]),
            self.equation[:, -1:],
        )
        self.equation = np.hstack(inserted)
        self.num_tmps += 1
        return self.num_qubits + self.num_tmps - 1

    def add_simple_phase(self, q: QubitId, loc: Node, angle: Fraction | Wire) -> None:
        eqn = self.equation[q, :-1]
        parity = self.equation[q, -1]
        self.phases[as_tuple(eqn)].append(SimplePhase(loc, parity, angle))

    def pad_phases(self):
        """Updates the phases to the current vocabulary of variables by padding with
        zeros."""
        phases_full = defaultdict(list)
        for eqn, phases in self.phases.items():
            eqn_full = GF2.Zeros(self.equation.shape[1] - 1)
            eqn_full[: len(eqn)] = eqn
            phases_full[as_tuple(eqn_full)] = phases
        self.phases = phases_full

    def apply_quantum_op(self, op: str, qs: list[QubitId], loc: Node) -> list[QubitId]:
        match op, qs:
            case "QAlloc" | "QuregExtractIndex", []:
                return [self.new_qubit()]
            case "Measure", [q]:
                return [q]
            case "QFree", [_]:
                return []
            case "Reset", [q]:
                self.equation[q] = 0
            case "X", [q]:
                self.equation[q, -1] += GF2(1)
            case "CX", [q1, q2]:
                self.equation[q2] += self.equation[q1]
            case "T", [q]:
                self.add_simple_phase(q, loc, Fraction(1, 4))
            case _, qs:
                for q in qs:
                    y = self.new_tmp()
                    self.equation[q] = 0
                    self.equation[q, y] = 1
        return qs

    def apply_dfg(self, qs: list[QubitId], parent: Node) -> list[QubitId]:
        hugr = self.hugr
        inp, *_ = hugr.children(parent)
        id_of = dict(zip(outgoing_qubits(inp, hugr), qs, strict=False))
        wire_of = dict(zip(qs, outgoing_qubits(inp, hugr), strict=False))

        for node in toposort(hugr, parent):
            in_ids = [id_of[wire] for wire in incoming_qubits(node, hugr)]
            out_ids = []
            match hugr[node].op:
                case ops.Custom(extension="tket2.quantum" | "QPR", op_name=name):
                    out_ids = self.apply_quantum_op(name, in_ids, node)
                case ops.Conditional():
                    out_ids = self.apply_conditional(node, in_ids)
                case ops.TailLoop():
                    out_ids = self.apply_loop(node, in_ids)
                case ops.Output():
                    outputs = in_ids

            # Update tracked wires
            for out_wire, out_id in zip(
                outgoing_qubits(node, hugr), out_ids, strict=False
            ):
                id_of[out_wire] = out_id
                wire_of[out_id] = out_wire

        return outputs

    def apply_conditional(self, node: Node, qs: list[QubitId]) -> list[QubitId]:
        cases, case_outs = [], []
        summary = None
        for case_node in self.hugr.children(node):
            case, out = Analysis.run(self.hugr, case_node)
            cases.append(case)
            case_outs.append(out)

            # Apply permutation such that the outputted qubits are in the front in order
            # and discarded ones at the end
            discarded = [q for q in range(case.num_qubits) if q not in out]
            case.equation[:] = case.equation[out + discarded]

            # Project the discarded qubits and temporaries
            d = case.to_domain().forget_ancillas(len(discarded))
            d = d.project_tmps()

            # Update summary by joining with this branch
            if summary is not None:
                summary = summary.join(d)
            else:
                summary = d

            # Check for phase gates that only depend on qubits that pass through the
            # case (i.e. no dependency on measured or freshly allocated qubits).
            # Also, we need to make sure that those qubits are passed through *all* of
            # the conditionals to allow safe hoisting!
            # TODO

        # We might need to create more qubit variables on top-level to handle all the
        # stuff that this conditional returns
        outs = []
        for q in case_outs[0]:
            if q in qs:
                outs.append(q)
            else:
                outs.append(self.new_qubit())

        # Expand summary to the full set of qubits
        summary = summary.embed_into(self.num_qubits, qs)

        # Fast-forward the current state with the summary. This may introduce new
        # temporary variables into the current state.
        self.equation = self.to_domain().compose_ff(summary)

        # Update previous stored phase equations to include the new temporaries
        # TODO: We're assume that we only added new temporaries and none were removed.
        #   Is that a safe assumption??
        self.pad_phases()

        # Look for phases that can be hoisted out of the loop. This is only legal for
        # phases that only depend on qubits that we can safely identify after the
        # conditional. This is only the case for qubits that are outputted at the same
        # index for all cases.
        # case_outs = np.array(case_outs)
        # static_outs = set(np.where(np.all(case_outs == case_outs[0, :], axis=0)))
        # for i, case in enumerate(cases):
        #     for eqn_tuple, phases in case.phases.items():
        #         # Find variables that influence this gate (excluding the parity bit)
        #         eqn = GF2(eqn_tuple)
        #         (vs,) = eqn.nonzero()
        #         # Check if it's hoistable
        #         if all(v < case.num_qubits and v in static_outs for v in vs):
        #             # Express the equation in terms of the global vocabulary
        #             eqn_full = GF2.Zeros(self.num_qubits + self.num_tmps)
        #             eqn_full[case_outs[i]] = eqn
        #             # Canonicalize the equation w.r.t the outer relation
        #             eqn_ff, parity = d.canonicalize(eqn_full)
        #             self.phases[as_tuple(eqn_ff)].append(
        #                 LoopHoistedPhase(node, phases, eqn, parity)
        #             )


        return outs

    def apply_loop(self, node: Node, qs: list[QubitId]) -> list[QubitId]:
        loop, continue_vars = Analysis.run(self.hugr, node)

        # Linearity ensures that the loop has the same number of qubits that go around
        # in each iteration and eventually outputted
        assert len(continue_vars) == len(qs)
        # However, there might be allocations and matching deallocations in the loop
        discarded = [q for q in range(loop.num_qubits) if q not in continue_vars]

        # Apply permutation such that the qubits needed for the next iteration are back
        # in the original first rows and discarded ones at the end
        loop.equation[:] = loop.equation[continue_vars + discarded]

        # Check if any phase gates in the loop depend only on variables that remain
        # static between iterations. Those may be hoisted out of the loop later on
        hoistable = []
        for eqn_tuple, phases in loop.phases.items():
            # Find variables that influence this gate (excluding the parity bit)
            eqn = GF2(eqn_tuple)
            (vs,) = eqn.nonzero()
            # We can only hoist if it doesn't depend on ancillas or temporaries.
            if all(v < loop.num_qubits for v in vs) and all(
                # We also need to check if all of those qubits are returned to their
                # original state for the next iteration
                np.array_equal(np.where(loop.equation[v] == 1), np.array([[v]]))
                for v in vs
            ):
                hoistable.append(eqn_tuple)

        # Project the discarded qubits
        d = loop.to_domain().forget_ancillas(len(discarded))

        # Project the temporary variables and compute Kleene closure
        # TODO: Consider the starting state when computing the Kleene closure?
        summary = d.project_tmps().kleene_closure()

        # Expand summary to the full set of qubits
        summary = summary.embed_into(self.num_qubits, qs)

        # Fast-forward the current state with the summary. This may introduce new
        # temporary variables into the current state.
        self.equation = self.to_domain().compose_ff(summary)

        # Update previous stored phase equations to include the new temporaries
        # TODO: We're assume that we only added new temporaries and none were removed.
        #   Is that a safe assumption??
        self.pad_phases()

        # Add hoisted phases
        d = self.to_domain()
        for eqn in hoistable:
            # Remove the equation from the inner loop context
            phases = loop.phases.pop(eqn)
            # Express the equation in terms of the global vocabulary
            eqn_full = GF2.Zeros(self.num_qubits + self.num_tmps)
            eqn_full[qs] = eqn
            # Canonicalize the equation w.r.t the outer relation
            eqn_ff, parity = d.canonicalize(eqn_full)
            self.phases[as_tuple(eqn_ff)].append(
                LoopHoistedPhase(node, phases, eqn, parity)
            )

        # Finally, store the nested analysis with the remaining unhoisted phases
        self.nested_analysis[node] = loop
        return qs


def as_tuple(xs: GF2) -> tuple[bool, ...]:
    return tuple(x.item() for x in xs)
