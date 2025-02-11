from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TypeAlias

from galois import GF2

import numpy as np

from hugr import Node, Hugr, Wire, OutPort
from hugr import ops

from hugr.build.dfg import DfBase

from tket2_exts import quantum as quantum_ext

quantum = quantum_ext()



QubitId: TypeAlias = int


@dataclass
class Term:
    equation: GF2
    phase_gates: set[tuple[Node, bool]]


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

    # Mapping from
    terms: dict[bytes, Term]

    hugr: Hugr

    num_qubits: int
    num_tmps: int

    def __init__(self, num_qubits: int, hugr: Hugr):
        self.num_qubits = num_qubits
        self.num_tmps = 0
        self.hugr = hugr
        # Initialise as [ I | 0 ], i.e. `x_i' = x_i` for all i
        self.equation = np.hstack(
            (GF2.Identity(num_qubits), GF2.Zeros((num_qubits, 1)))
        )
        self.terms = {}

    @staticmethod
    def run(hugr: Hugr, parent: Node) -> Analysis:
        inp, *_ = hugr.children(parent)
        qs = {wire: i for i, (wire, _) in enumerate(hugr.outgoing_links(inp))}
        analysis = Analysis(len(qs), hugr)
        for node in toposort(hugr, parent):
            match hugr[node].op:
                case ops.Custom(extension="tket2.quantum", op_name=name):
                    analysis.apply_gate(
                        name, [qs[wire] for _, [wire] in hugr.incoming_links(node)], node
                    )
                case ops.TailLoop():
                    pass

            # Update tracked wires
            for (_, [in_wire]), (out_wire, _) in zip(hugr.incoming_links(node), hugr.outgoing_links(node)):
                qs[out_wire] = qs[in_wire]

        return analysis

    def to_domain(self) -> Domain:
        """Turns the gathered information into an instance of our domain."""
        # Construct [ X' | X | Y | c ] where X' is just an identity diagonal
        rel = np.hstack((GF2.Identity(self.num_qubits), self.equation))
        return Domain(rel, self.num_qubits)

    def new_var(self) -> int:
        inserted = (
            self.equation[:, :-1],
            GF2.Zeros([self.num_qubits, 1]),
            self.equation[:, -1:],
        )
        self.equation = np.hstack(inserted)
        self.num_tmps += 1
        return self.num_qubits + self.num_tmps - 1

    def add_tern(self, q: QubitId, loc: Node) -> None:
        eq = self.equation[q, 1:]
        parity = self.equation[q, 0]
        if eq.tobytes() not in self.terms:
            self.terms[eq.tobytes()] = Term(eq, {(loc, parity)})
        else:
            self.terms[eq.tobytes()].phase_gates.add((loc, parity))

    def apply_gate(self, op: str, qs: list[QubitId], loc: Node) -> None:
        match op, qs:
            case "Reset", [q]:
                self.equation[q] = 0
            case "X", [q]:
                self.equation[q, -1] += 1
            case "CX", [q1, q2]:
                self.equation[q2] += self.equation[q1]
            case "T", [q]:
                self.add_tern(q, loc)
            case _, qs:
                for q in qs:
                    y = self.new_var()
                    self.equation[q] = 0
                    self.equation[q, y] = 1

    def apply_loop(self, node: Node) -> None:
        loop = Analysis.run(self.hugr, node)

        # Check if any phase gates depend only on variables that remain static between
        # loop input and output
        # for term in loop.terms.values():
        #     for v in np.where(term.equation == 1):
        #         is_tmp =


        # Project out the temporary variables and compute Kleene closure
        summary = loop_ctx.to_domain().project_tmps().kleene_closure()
        # Reduce the loop terms w.r.t. to the new closure relation
        # Fast-forward the current state with the summary
        rel = self.to_domain().compose_ff(summary)






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
        return self.rel[:, self.num_qubits:2*self.num_qubits]

    @property
    def post(self) -> GF2:
        return self.rel[:, :self.num_qubits]

    @property
    def tmp(self) -> GF2:
        return self.rel[:, 2*self.num_qubits:-1]

    @property
    def c(self) -> GF2:
        return self.rel[:, -1:]

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
        block = np.hstack((np.vstack((a, b)), np.vstack((a, GF2.Zeros(a.shape)))))
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
        x, y, c = res[:, -n-1:-1], res[:, n:-n-1], res[:, -1:]
        y = y[:, np.any(y != 0, axis=0)]
        return np.hstack((x, y, c))

    def kleene_closure(self) -> Domain:
        a = self
        while True:
            b = a.join(a.compose(self))
            if np.array_equal(a.rel, b.rel):
                return b
            a = b



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
    return a[np.all(a[:, :stop-start] == 0, axis=1), stop-start:]


def toposort(hugr: Hugr, parent: Node) -> Iterator[Node]:
    inp, *_ = hugr.children(parent)
    queue = {inp}
    visited = set()
    while queue:
        node = queue.pop()
        yield node
        visited.add(node)
        for _, succs in hugr.outgoing_links(node):
            for succ in succs:
                if all(pred.node in visited for pred, _ in hugr.incoming_links(node)):
                    queue.add(succ.node)




