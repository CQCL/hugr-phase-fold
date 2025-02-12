from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from galois import GF2


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
