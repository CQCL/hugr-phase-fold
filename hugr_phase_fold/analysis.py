from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from typing import NamedTuple, TypeAlias

import numpy as np
from galois import GF2
from hugr import Hugr, Node, Wire, ops
from hugr.ops import Op

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

    def apply_quantum_op(self, op: str, qs: list[QubitId], loc: Node) -> list[QubitId]:
        match op, qs:
            case "QAlloc", []:
                return [self.new_qubit()]
            case "Measure" | "QFree", [_]:
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
                case ops.Custom(extension="tket2.quantum", op_name=name):
                    out_ids = self.apply_quantum_op(name, in_ids, node)
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

        # Expand vocabulary to the full set of qubits
        # TODO: Replace with numpy magic
        missing_qubits = [q for q in range(self.num_qubits) if q not in qs]
        num_rows = summary.rel.shape[0]
        summary_full = GF2.Zeros(
            (
                num_rows + len(missing_qubits),
                2 * self.num_qubits + 1,
            )
        )
        for i in range(num_rows):
            cols = qs + [self.num_qubits + q for q in qs] + [-1]
            summary_full[i, cols] = summary.rel[i]
        for i, q in enumerate(missing_qubits):
            summary_full[num_rows + i, q] = summary_full[
                num_rows + i, self.num_qubits + q
            ] = 1

        # Fast-forward the current state with the summary. This may introduce new
        # temporary variables into the current state.
        self.equation = self.to_domain().compose_ff(
            Domain(summary_full, self.num_qubits)
        )

        # Update previous stored phase equations to include the new temporaries
        # TODO: We're assume that we only added new temporaries and none were removed.
        #   Is that a safe assumption??
        phases_full = defaultdict(list)
        for eqn, phases in self.phases.items():
            eqn_full = GF2.Zeros(self.equation.shape[1] - 1)
            eqn_full[: len(eqn)] = eqn
            phases_full[as_tuple(eqn_full)] = phases
        self.phases = phases_full

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


@dataclass
class Domain:
    """The affine relation domain `KS[X', X, Y]`.

    An element A of this domain corresponds to a relation represented by a GF(2) matrix
    of with layout A := [ X' | X | Y | c ].
    """

    rel: GF2
    num_qubits: int

    @property
    def dim(self) -> int:
        return self.rel.shape[1]

    @property
    def pre(self) -> GF2:
        return self.rel[:, self.num_qubits : 2 * self.num_qubits]

    @property
    def post(self) -> GF2:
        return self.rel[:, : self.num_qubits]

    @property
    def tmp(self) -> GF2:
        return self.rel[:, 2 * self.num_qubits : -1]

    @property
    def c(self) -> GF2:
        return self.rel[:, -1:]

    def forget_ancillas(self, count: int) -> Domain:
        """Stops tracking of the last `count` qubits by turning them into temporary
        variables.
        """
        rel = project(self.rel, self.num_qubits - count, self.num_qubits)
        return Domain(rel, self.num_qubits - count)

    def project_tmps(self) -> Domain:
        rel = project(self.rel, 2 * self.num_qubits, self.dim - 1)
        return Domain(rel, self.num_qubits)

    def join(self, other: Domain) -> Domain:
        # We have:  A v B := project | A | A |
        #                            | B | 0 |
        #
        # Where the projection is over the left block. See bottom of p. 14.
        assert self.dim == other.dim
        a, b = self.rel, other.rel
        block = np.hstack(
            (
                np.vstack((a, b)),
                np.vstack((a, GF2.Zeros((b.shape[0], a.shape[1])))),
            )
        )
        return Domain(project(block, 0, self.dim), self.num_qubits)

    def compose(self, other: Domain) -> Domain:
        """Sequential composition of relations."""
        # We have:  A ; B := project | A_Post |      0 | A_Pre | A_Tmp |     0 | c |
        #                            | B_Pre  | B_Post |     0 |     0 | B_Tmp | d |
        #
        # Where the projection is over the left-most block. See bottom of p. 14.
        assert self.dim == other.dim
        a, b = self, other
        top = (
            a.post,
            GF2.Zeros([a.post.shape[0], b.post.shape[1]]),
            a.pre,
            a.tmp,
            GF2.Zeros([a.post.shape[0], b.tmp.shape[1]]),
            a.c,
        )
        bot = (
            b.pre,
            b.post,
            GF2.Zeros([b.post.shape[0], a.pre.shape[1]]),
            GF2.Zeros([b.post.shape[0], a.tmp.shape[1]]),
            b.tmp,
            b.c,
        )
        block = np.vstack((np.hstack(top), np.hstack(bot)))
        return Domain(project(block, 0, self.num_qubits), self.num_qubits)

    def kleene_closure(self) -> Domain:
        a = self
        while True:
            b = a.join(a.compose(self))
            if np.array_equal(a.rel, b.rel):
                return b
            a = b

    def compose_ff(self, other: Domain) -> GF2:
        """Fast-forward sequential composition of relations where all temporaries have
        been projected out of `other`.

        Thus, `other` may not have full rank. However, this method ensures that the
        resulting fast-forwarded relation has full rank again (i.e. every variable in
        X' has a representation) by introducing more temporaries.

        The resulting matrix has shape [ X | Y | c ], leaving the X' identity matrix
        at the start implicit.
        """
        # According to the paper (p. 16) we have
        #
        #     A ;ff B := project | A_Post |      0 | A_Pre | A_Tmp | 0 | c |
        #                        |  B_Pre | B_Post |     0 |     0 | 0 | d |
        #                        |      0 |      I |     0 |     0 | I | 0 |
        #
        # However, since we want to go back into a representation [ X' | X | Y | c ]
        # where X' is the identity, it's more convenient to use the following system:
        #
        #    | I |      I |      0 |     0 |     0 | 0
        #    | 0 | B_Post |  B_Pre |     0 |     0 | d
        #    | 0 |      0 | A_Post | A_Tmp | A_Pre | c
        #
        # The first column represents the new variables describing the post state of the
        # whole transition. Row-reducing ensures that we solve B_Post and can just read
        # of the solution [ X' | Y | X | c ] from the first row. Note that the new
        # temporaries Y are made up of  the reduced B_Post, B_Pre, and A_Tmp variables.
        assert self.num_qubits == other.num_qubits
        assert other.dim == 2 * other.num_qubits + 1
        assert self.rel.shape[0] == self.num_qubits, "must have full rank"
        a, b = self, other
        n = self.num_qubits
        top = (
            GF2.Identity(n),
            GF2.Identity(n),
            GF2.Zeros([n, n]),
            GF2.Zeros([n, n]),
            GF2.Zeros(a.tmp.shape),
            GF2.Zeros(a.c.shape),
        )
        mid = (
            GF2.Zeros([b.rel.shape[0], n]),
            b.post,
            b.pre,
            GF2.Zeros([b.rel.shape[0], a.tmp.shape[1]]),
            GF2.Zeros([b.rel.shape[0], a.pre.shape[1]]),
            b.c,
        )
        bot = (
            GF2.Zeros([n, n]),
            GF2.Zeros([n, b.post.shape[1]]),
            a.post,
            a.tmp,
            a.pre,
            a.c,
        )
        block = np.vstack((np.hstack(top), np.hstack(mid), np.hstack(bot)))
        block = block.row_reduce()
        # We can read the solution off from the first rows of the reduced matrix.
        # We get a new system [ X' | Y | X | c ] where X' is the identity
        res = block[:n]
        # Reshape it into [ X | Y | c ] to be compatible with the context and try to
        # purge zero columns from Y (those correspond to unused temporaries)
        x, y, c = res[:, -n - 1 : -1], res[:, n : -n - 1], res[:, -1:]
        y = y[:, np.any(y != 0, axis=0)]
        return np.hstack((x, y, c))

    def canonicalize(self, eqn: GF2) -> tuple[GF2, bool]:
        """Canonicalizes a functional represented by a vector with respect to the
        current relation.
        """
        # This works by performing the row reduction
        #
        #    [ A_Post | A_Tmp | A_Pre | c ]  ->  [ A_Post | A_Tmp    | A_Pre    | c ]
        #    [    Eqn |     0 |     0 | 0 ]      [      0 | Eqn'_Tmp | Eqn'_Pre | p ]
        #
        # and returning [ Eqn'_Pre | Eqn'_Tmp ] together with the new parity p. In the
        # paper, thisoperations is referred to as `reduce` (see p. 18).
        assert self.rel.shape[0] == self.num_qubits
        top = np.hstack((self.post, self.tmp, self.pre, self.c))
        bot = GF2.Zeros((1, self.dim))
        bot[:, : self.num_qubits] = eqn
        block = np.vstack((top, bot)).row_reduce()
        new_eqn = (
            block[-1, -self.num_qubits - 1 : -1],
            block[-1, self.num_qubits : -self.num_qubits - 1],
        )
        return np.concatenate(new_eqn), block[-1, -1]


def project(a: GF2, start: int, stop: int) -> GF2:
    """Projects a chunk of variables (i.e. columns) out of the relation."""
    # See bottom of p. 14.
    assert stop >= start
    if start != 0:
        # Permute matrix to move the columns that should be projected out to the front
        a = np.hstack((a[:, start:stop], a[:, :start], a[:, stop:]))
    # Computing the reduced row echelon form
    a = a.row_reduce()
    # Any rows where one of the projection columns is non-zero can be removed (since
    # those rows can be solved by fixing one of the projected values)
    return a[np.all(a[:, : stop - start] == 0, axis=1), stop - start :]


def project_columns(a: GF2, cols: list[int]) -> GF2:
    """Projects discontinuous columns out of a relation."""
    # Permute matrix to move the columns that should be projected out to the front
    col_set = set(cols)
    assert len(col_set) == len(cols)
    other_cols = [i for i in range(a.shape[1]) if i not in col_set]
    a = np.hstack((a[:, cols], a[:, other_cols]))
    # Computing the reduced row echelon form
    a = a.row_reduce()
    # Any rows where one of the projection columns is non-zero can be removed (since
    # those rows can be solved by fixing one of the projected values)
    return a[np.all(a[:, : len(cols)] == 0, axis=1), len(cols) :]


def as_tuple(xs: GF2) -> tuple[bool, ...]:
    return tuple(x.item() for x in xs)
