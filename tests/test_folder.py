
from hugr import Hugr, ops, tys, val
from hugr.build.dfg import Dfg
from hugr.hugr.render import DotRenderer
from tket2.circuit.build import CX, PauliX, QAlloc, QFree, Rz

from hugr_phase_fold.folder import FAdd, FSub, PhaseFolder, S, T


def run(hugr: Hugr, request, export: bool):
    if export:
        renderer = DotRenderer()
        name = request.node.name[5:]
        renderer.store(hugr, f"examples/{name}")
    PhaseFolder(hugr).run(hugr.root)
    if export:
        renderer.store(hugr, f"examples/{name}.after")


def gate_count(hugr: Hugr, gate: ops.Custom | ops.ExtOp):
    if isinstance(gate, ops.ExtOp):
        gate = gate.to_custom_op()
    count = 0
    for n in hugr:
        op = hugr[n].op
        if isinstance(op, ops.ExtOp):
            op = op.to_custom_op()
        if isinstance(op, ops.Custom) and op.op_name == gate.op_name:
            count += 1
    return count


def test_noop_loop(request):
    circ = Dfg(tys.Qubit)
    (q,) = circ.inputs()
    q = circ.add_op(T, q)
    with circ.add_tail_loop([], [q]) as loop:
        loop.set_outputs(loop.load(val.TRUE), *loop.inputs())
    (q,) = loop.outputs()
    q = circ.add_op(T, q)
    circ.set_outputs(q)

    run(circ.hugr, request, export=True)
    assert gate_count(circ.hugr, T) == 0
    assert gate_count(circ.hugr, S) == 1


def test_cx_loop(request):
    circ = Dfg(tys.Qubit, tys.Qubit)
    q1, q2 = circ.inputs()
    q1 = circ.add_op(T, q1)
    q2 = circ.add_op(T, q2)
    with circ.add_tail_loop([], [q1, q2]) as loop:
        q1, q2 = loop.inputs()
        q1, q2 = loop.add_op(CX, q1, q2)
        loop.set_outputs(loop.load(val.TRUE), q1, q2)
    q1, q2 = loop.outputs()
    q1 = circ.add_op(T, q1)
    q2 = circ.add_op(T, q2)
    circ.set_outputs(q1, q2)

    # Only one T-pair can be cancelled
    run(circ.hugr, request, export=True)
    assert gate_count(circ.hugr, T) == 2
    assert gate_count(circ.hugr, S) == 1


def test_swap_loop(request):
    circ = Dfg(tys.Qubit, tys.Qubit)
    q1, q2 = circ.inputs()
    q1, q2 = circ.add_op(CX, q1, q2)
    q1 = circ.add_op(T, q1)
    q2 = circ.add_op(T, q2)
    q1, q2 = circ.add_op(CX, q1, q2)
    with circ.add_tail_loop([], [q1, q2]) as loop:
        q1, q2 = loop.inputs()
        loop.set_outputs(loop.load(val.TRUE), q2, q1)
    q1, q2 = loop.outputs()
    q1, q2 = circ.add_op(CX, q1, q2)
    q1 = circ.add_op(T, q1)
    q2 = circ.add_op(T, q2)
    circ.set_outputs(q1, q2)

    run(circ.hugr, request, export=False)
    assert gate_count(circ.hugr, T) == 2
    assert gate_count(circ.hugr, S) == 1


def test_ancilla_loop(request):
    circ = Dfg(tys.Qubit)
    (q,) = circ.inputs()
    q = circ.add_op(T, q)
    with circ.add_tail_loop([], [q]) as loop:
        (q,) = loop.inputs()
        loop.add_op(QFree, q)
        q = loop.add_op(QAlloc)
        loop.set_outputs(loop.load(val.TRUE), q)
    (q,) = loop.outputs()
    q = circ.add_op(T, q)
    circ.set_outputs(q)

    # No Ts can be cancelled
    run(circ.hugr, request, export=False)
    assert gate_count(circ.hugr, T) == 2


def test_ancilla_zero(request):
    circ = Dfg(tys.Qubit)
    (q,) = circ.inputs()
    q = circ.add_op(T, q)
    with circ.add_tail_loop([], [q]) as loop:
        (q,) = loop.inputs()
        tmp = loop.add_op(QAlloc)
        tmp, q = loop.add_op(CX, tmp, q)
        loop.add_op(QFree, tmp)
        loop.set_outputs(loop.load(val.TRUE), q)
    (q,) = loop.outputs()
    q = circ.add_op(T, q)
    circ.set_outputs(q)

    run(circ.hugr, request, export=True)
    assert gate_count(circ.hugr, T) == 0


def test_hoist_loop_basic(request):
    circ = Dfg(tys.Qubit)
    (q,) = circ.inputs()
    q = circ.add_op(T, q)
    with circ.add_tail_loop([], [q]) as loop:
        (q,) = loop.inputs()
        q = loop.add_op(T, q)
        loop.set_outputs(loop.load(val.TRUE), q)
    (q,) = loop.outputs()
    circ.set_outputs(q)

    run(circ.hugr, request, export=True)
    assert gate_count(circ.hugr, T) == 0
    assert gate_count(circ.hugr, Rz) == 1
    assert gate_count(circ.hugr, FAdd) == 1


def test_hoist_loop_gadget(request):
    circ = Dfg(tys.Qubit, tys.Qubit, tys.Qubit)
    q1, q2, q3 = circ.inputs()
    with circ.add_tail_loop([], [q1, q2, q3]) as loop:
        q1, q2, q3 = loop.inputs()
        q1, q2 = loop.add_op(CX, q1, q2)
        q2, q3 = loop.add_op(CX, q2, q3)
        q2 = loop.add_op(T, q2)
        q1, q2 = loop.add_op(CX, q1, q2)
        loop.set_outputs(loop.load(val.TRUE), q1, q2, q3)
    q1, q2, q3 = loop.outputs()
    circ.set_outputs(q1, q2, q3)

    run(circ.hugr, request, export=True)
    assert gate_count(circ.hugr, T) == 0
    assert gate_count(circ.hugr, Rz) == 1
    assert gate_count(circ.hugr, FAdd) == 1


def test_hoist_loop_no_repeat(request):
    circ = Dfg(tys.Qubit, tys.Qubit, tys.Qubit)
    q1, q2, q3 = circ.inputs()
    q1, q2 = circ.add_op(CX, q1, q2)
    q2 = circ.add_op(T, q2)
    q1, q2 = circ.add_op(CX, q1, q2)
    with circ.add_tail_loop([], [q1, q2, q3]) as loop:
        q1, q2, q3 = loop.inputs()
        q1, q2 = loop.add_op(CX, q1, q2)
        q2, q3 = loop.add_op(CX, q2, q3)
        q2 = loop.add_op(T, q2)
        q1, q2 = loop.add_op(CX, q1, q2)
        loop.set_outputs(loop.load(val.TRUE), q1, q2, q3)
    q1, q2, q3 = loop.outputs()
    q1, q2 = circ.add_op(CX, q1, q2)
    q2 = circ.add_op(T, q2)
    q1, q2 = circ.add_op(CX, q1, q2)
    circ.set_outputs(q1, q2, q3)

    run(circ.hugr, request, export=True)
    assert gate_count(circ.hugr, T) == 0
    assert gate_count(circ.hugr, Rz) == 1
    assert gate_count(circ.hugr, FAdd) == 2


def test_hoist_loop_nested(request):
    circ = Dfg(tys.Qubit)
    (q,) = circ.inputs()
    q = circ.add_op(T, q)
    with circ.add_tail_loop([], [q]) as loop:
        (q,) = loop.inputs()
        q = loop.add_op(T, q)
        with loop.add_tail_loop([], [q]) as inner_loop:
            (q,) = inner_loop.inputs()
            q = inner_loop.add_op(T, q)
            inner_loop.set_outputs(inner_loop.load(val.TRUE), q)
        loop.set_outputs(loop.load(val.TRUE), *inner_loop.outputs())
    (q,) = loop.outputs()
    circ.set_outputs(q)

    run(circ.hugr, request, export=True)
    assert gate_count(circ.hugr, T) == 0
    assert gate_count(circ.hugr, Rz) == 1
    assert gate_count(circ.hugr, FAdd) == 2


def test_hoist_loop_nested_cx(request):
    circ = Dfg(tys.Qubit, tys.Qubit)
    q1, q2 = circ.inputs()
    q2 = circ.add_op(PauliX, q2)
    with circ.add_tail_loop([], [q1, q2]) as loop:
        q1, q2 = loop.inputs()
        q1, q2 = loop.add_op(CX, q1, q2)
        with loop.add_tail_loop([], [q1, q2]) as inner_loop:
            q1, q2 = inner_loop.inputs()
            q2 = inner_loop.add_op(T, q2)
            inner_loop.set_outputs(inner_loop.load(val.TRUE), q1, q2)
        q1, q2 = inner_loop.outputs()
        q1, q2 = loop.add_op(CX, q1, q2)
        loop.set_outputs(loop.load(val.TRUE), q1, q2)
    q1, q2 = loop.outputs()
    circ.set_outputs(q1, q2)

    run(circ.hugr, request, export=True)
    assert gate_count(circ.hugr, T) == 0
    assert gate_count(circ.hugr, Rz) == 1
    assert gate_count(circ.hugr, FAdd) == 0
    assert gate_count(circ.hugr, FSub) == 1


def test_hoist_loop_partial(request):
    circ = Dfg(tys.Qubit, tys.Qubit)
    q1, q2 = circ.inputs()
    q1 = circ.add_op(T, q1)
    q2 = circ.add_op(T, q2)
    with circ.add_tail_loop([], [q1]) as loop:
        (q1,) = loop.inputs()
        q1 = loop.add_op(T, q1)
        loop.set_outputs(loop.load(val.TRUE), q1)
    (q1,) = loop.outputs()
    circ.set_outputs(q1, q2)

    run(circ.hugr, request, export=True)
    assert gate_count(circ.hugr, T) == 1
    assert gate_count(circ.hugr, Rz) == 1
    assert gate_count(circ.hugr, FAdd) == 1


def test_hoist_loop_fastforward(request):
    circ = Dfg(tys.Qubit, tys.Qubit)
    q1, q2 = circ.inputs()
    q1, q2 = circ.add_op(CX, q1, q2)
    q1 = circ.add_op(T, q1)
    q2 = circ.add_op(T, q2)
    with circ.add_tail_loop([], [q1, q2]) as loop:
        q1, q2 = loop.inputs()
        q2 = loop.add_op(T, q2)
        loop.set_outputs(loop.load(val.TRUE), q1, q2)
    q1, q2 = loop.outputs()
    circ.set_outputs(q1, q2)

    run(circ.hugr, request, export=True)
    assert gate_count(circ.hugr, T) == 1
    assert gate_count(circ.hugr, Rz) == 1
    assert gate_count(circ.hugr, FAdd) == 1


def test_hoist_loop_parity(request):
    circ = Dfg(tys.Qubit)
    (q,) = circ.inputs()
    q = circ.add_op(PauliX, q)
    q = circ.add_op(T, q)
    with circ.add_tail_loop([], [q]) as loop:
        (q,) = loop.inputs()
        q = loop.add_op(T, q)
        loop.set_outputs(loop.load(val.TRUE), q)
    (q,) = loop.outputs()
    circ.set_outputs(q)

    run(circ.hugr, request, export=True)
    assert gate_count(circ.hugr, T) == 0
    assert gate_count(circ.hugr, Rz) == 1
    assert gate_count(circ.hugr, FSub) == 1


def test_conditional_id(request):
    circ = Dfg(tys.Qubit, tys.Qubit)
    q1, q2 = circ.inputs()
    q1 = circ.add_op(T, q1)
    q2 = circ.add_op(T, q2)
    with circ.add_conditional(circ.load(val.TRUE), q1, q2) as cond:
        with cond.add_case(0) as case:
            q1, q2 = case.inputs()
            case.set_outputs(q1, q2)
        with cond.add_case(1) as case:
            q1, q2 = case.inputs()
            case.set_outputs(q1, q2)
    q1, q2 = cond.outputs()
    q1 = circ.add_op(T, q1)
    q2 = circ.add_op(T, q2)
    circ.set_outputs(q1, q2)

    run(circ.hugr, request, export=True)
    assert gate_count(circ.hugr, T) == 0
    assert gate_count(circ.hugr, S) == 2


def test_conditional_swap(request):
    circ = Dfg(tys.Qubit, tys.Qubit)
    q1, q2 = circ.inputs()
    q1, q2 = circ.add_op(CX, q1, q2)
    q2 = circ.add_op(T, q2)
    q1, q2 = circ.add_op(CX, q1, q2)
    with circ.add_conditional(circ.load(val.TRUE), q1, q2) as cond:
        with cond.add_case(0) as case:
            q1, q2 = case.inputs()
            case.set_outputs(q2, q1)
        with cond.add_case(1) as case:
            q1, q2 = case.inputs()
            case.set_outputs(q1, q2)
    q1, q2 = cond.outputs()
    q1, q2 = circ.add_op(CX, q1, q2)
    q1 = circ.add_op(T, q1)
    q2 = circ.add_op(T, q2)
    circ.set_outputs(q1, q2)

    run(circ.hugr, request, export=True)
    assert gate_count(circ.hugr, T) == 1
    assert gate_count(circ.hugr, S) == 1
