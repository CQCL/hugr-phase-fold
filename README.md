# Phase Folding Optimizer for Hugr

This is a prototype implementation of the paper [*Linear and non-linear relational analyses for Qantum
Program Optimization*](https://doi.org/10.1145/3704873) by Amy and Lunderville in Hugr.


## Features

### Linear Phase Folding

Section 4 of the paper is fully implemented, allowing phase folding using linear
relational analysis over Hugr's `Conditional` and `TailLoop` control flow.
Here are some examples of optimisations that are performed by the pass
(using C syntax instead of Hugr for readability):


<table>
<tr>
<td> Before </td> <td> After </td>
</tr>
<tr>
<td>

```c
t(q1);
if (...) {
    cx(q1, q2);
}
t(q1);
```

</td>
<td>
    
```c

if (...) {
    cx(q1, q2);
}
s(q1);
```
</td>
</tr>

<tr>
<td>

```c
t(q1);
do {
    cx(q1, q2);
} while (...);
t(q1);
```

</td>
<td>
    
```c

do {
    cx(q1, q2);
} while (...);
s(q1);
```
</td>
</tr>

<tr>
<td>

```c
cx(q1, q2);
t(q2)
cx(q1, q2);
do {
    swap(q1, q2);
} while (...);
cx(q1, q2)
t(q2)
```

</td>
<td>
    
```c



do {
    swap(q1, q2);
} while (...);
cx(q1, q2)
s(q2)
```
</td>
</tr>
</table>


### Loop Hoisting

The original paper does not attempt to merge phase gates inside control flow blocks with ones on the outside.
However, due to the expressiveness of Hugr, it is actually feasible for us to perform those merges by hoisting them out of the loop and replacing them with an angle accumulator:


<table>
<tr>
<td> Before </td> <td> After </td>
</tr>
<tr>
<td>

```c
t(q);
do {
    t(q);
} while (...);

```

</td>
<td>
    
```c
float acc = 0.0f;
do {
    acc += 0.25f;
} while (...);
rz(q, acc + 0.25f);
```
</td>
</tr>

<tr>
<td>

```c
cx(q1, q2);
rz(q2, angle);
cx(q1, q2);
do {
    cx(q1, q2);
    t(q2);
    cx(q2, q3);
    cx(q1, q2);
} while (...);



```

</td>
<td>
    
```c


float acc = 0.0f;
do {
    cx(q1, q2);
    acc += 0.25f;
    cx(q2, q3);
    cx(q1, q2);
} while (...);
cx(q1, q2);
rz(q, acc + angle);
cx(q1, q2);
```
</td>
</tr>

<tr>
<td>

```c
x(q2);
do {
    cx(q1, q2);
    do {
        t(q2);
    } while (...);
    cx(q1, q2);
} while (...);



```

</td>
<td>
    
```c
float acc = 0.0f;
do {

    do {
        acc -= 0.25f;
    } while (...);

} while (...);
cx(q1, q2);
rz(q, acc + 0.25f);
cx(q1, q2);
```
</td>
</tr>
</table>


### Local Qubit Scopes for Improved Performance

The original paper and implementation assumes that all qubits are globally allocated and uses this whole register everywhere.
This means that the relational analysis for loop acting only on a small subset of qubits still scales with the total qubit number.
Here, I tried to improve the performance by running the analysis of blocks only on locally used qubits and only take the whole relation into account during the fast-forwarding step.

This also allows us to express things like ancillas that are locally created and again deallocated in the same loop.


## Usage

To get a local development version, just run

```sh
uv sync
```

The pass can be run as follows:

```python
from hugr_phase_fold.folder import PhaseFolder

hugr = ...  # Define your Hugr

folder = PhaseFolder(hugr)
# Run pass on a data-flow region rooted at a given node, mutating the Hugr in-place
folder.run(hugr.root)
```


## Todos and Ideas for Improvement

* Implement hoisting out of `Conditional` nodes
* Implement non-linear analysis pass (i.e. Section 5 of the paper)
* Take measurement into account: E.g. can we use the information that a loop is only repeated if a certain qubit is in state 0?
  ```c
  qubit q = qubit();
  do {
      qubit tmp = qubit();
      h(tmp);
      cx(tmp, q);
      t(q);
  } while (!measure(tmp));
  # Loop is only repeated if tmp was |0>, so the t only has effect a single time,
  # so it could be hoisted out
  ```
* Take intitial state into account when computing Kleene closure instead of only doing it during fast-forwarding:
  ```c
  q2 = qubit();
  cx(q1, q2);  # q1 and q2 have same parity
  do {
      # Those two Ts could be merged, but we don't since we don't know this
      # while computing the summary...
      t(q1);
      t(q2);
  } while (...);
  ```
* Extend to arbitrary control-flow graphs? This would make the loop hoisting step more difficult.

