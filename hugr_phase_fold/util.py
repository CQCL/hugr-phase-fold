from __future__ import annotations

from collections.abc import Iterator
from typing import TypeVar

from hugr import Hugr, InPort, Node, Wire, ops, tys
from hugr.ops import Op
from hugr.std.float import FLOAT_T, FloatVal


OpVar = TypeVar("OpVar", bound=Op)


def toposort(hugr: Hugr[OpVar], parent: Node) -> Iterator[Node]:
    queue = {node for node in hugr.children(parent) if hugr.num_incoming(node) == 0}
    visited = set()
    while queue:
        node = queue.pop()
        yield node
        visited.add(node)
        for _, succs in hugr.outgoing_links(node):
            for succ in succs:
                if all(
                    pred.node in visited or hugr[pred.node].parent != hugr[succ.node].parent
                    for _, [pred] in hugr.incoming_links(succ.node)
                ):
                    queue.add(succ.node)


def incoming_qubits(node: Node, hugr: Hugr[OpVar]) -> Iterator[Wire]:
    return (
        wire
        for _, [wire] in hugr.incoming_links(node)
        if hugr.port_type(wire) == tys.Qubit
    )


def outgoing_qubits(node: Node, hugr: Hugr[OpVar]) -> Iterator[Wire]:
    return (
        wire
        for (wire, _) in hugr.outgoing_links(node)
        if hugr.port_type(wire) == tys.Qubit
    )


def outgoing_qubit_targets(node: Node, hugr: Hugr[OpVar]) -> Iterator[InPort]:
    return (
        tgt
        for (_, [tgt]) in hugr.outgoing_links(node)
        if hugr.port_type(tgt) == tys.Qubit
    )


def load_float_const(v: float, parent: Node, hugr: Hugr[OpVar]) -> Wire:
    c = hugr.add_const(FloatVal(v), parent)
    load = hugr.add_node(ops.LoadConst(FLOAT_T, 1), parent, 1)
    hugr.add_link(c.out(0), load.inp(0))
    return load.out(0)


def remove_gate(node: Node, hugr: Hugr[OpVar]) -> None:
    in_srcs = list(incoming_qubits(node, hugr))
    out_tgts = list(outgoing_qubit_targets(node, hugr))
    hugr.delete_node(node)
    for src, tgt in zip(in_srcs, out_tgts, strict=True):
        hugr.add_link(src.out_port(), tgt)


def replace_gate(node: Node, replacement: ops.Op, hugr: Hugr[OpVar]) -> Node:
    in_srcs = list(incoming_qubits(node, hugr))
    out_tgts = list(outgoing_qubit_targets(node, hugr))
    parent = hugr[node].parent
    hugr.delete_node(node)
    repl = hugr.add_node(replacement, parent, len(out_tgts))
    for i, src in enumerate(in_srcs):
        hugr.add_link(src.out_port(), repl.inp(i))
    for i, tgt in enumerate(out_tgts):
        hugr.add_link(repl.out(i), tgt)
    return repl


def insert_gate(
    op: ops.Op, qubits: list[Wire], parent: Node, hugr: Hugr[OpVar]
) -> list[Wire]:
    tgts = [tgt for q in qubits for tgt in hugr.linked_ports(q.out_port())]
    gate = hugr.add_node(op, parent, len(tgts))
    for i, (src, tgt) in enumerate(zip(qubits, tgts, strict=True)):
        hugr.delete_link(src.out_port(), tgt)
        hugr.add_link(src.out_port(), gate.inp(i))
        hugr.add_link(gate.out(i), tgt)
    return list(gate.outputs())
