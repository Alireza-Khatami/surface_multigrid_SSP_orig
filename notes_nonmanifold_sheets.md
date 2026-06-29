# One-Ring Sheet Partitioning — Non-Manifold Case

## The question

In a non-manifold mesh, how can one endpoint of an edge have faces in a sheet
while the other endpoint has none? If two vertices share an edge, they must share
at least one face — so shouldn't both vertices appear in the same sheet?

## The answer

Both vertices **do** share at least one face (the face containing the edge).
But the one-ring of each vertex extends **beyond** the shared faces to include
faces in other parts of the mesh where only that vertex appears.

## Concrete example

```
Sheet A:  {vi, a, b}  {vi, b, c}   ← vi has faces here, vj does NOT
Sheet B:  {vi, vj, d} {vi, vj, e}  ← BOTH vi and vj have faces here
```

- The edge (vi, vj) **only exists in Sheet B**
- vi's full one-ring = Sheet A faces + Sheet B faces
- vj's full one-ring = Sheet B faces only

When `partition_onering_by_sheet` sees vi's one-ring, it creates an entry for
Sheet A. For that sheet:
- `sheets_Nsf[A]` = vi's Sheet A faces — **non-empty**
- `sheets_Ndf[A]` = vj's Sheet A faces — **empty** (vj is not in Sheet A at all)

## Why this happens in the SSP pipeline

Sheet A is a **topologically separate component** connected to vi through other
edges — not through the edge (vi, vj). The edge (vi, vj) only participates in
the sheets where both endpoints actually meet (Sheet B).

## Consequence for UV computation

UV correspondence for a collapse of edge (vi, vj) requires **both** endpoints
in the one-ring. Sheet A has only vi, so:
- `get_collapse_onering_faces` returns only vi's faces for Sheet A
- vj never appears in `F_ring_pre`
- `subsetVIdx_si` does not contain vj
- `b_si(1)` is never set → assert `b_si(0) < b_si(1)` fires

## Fix

Skip any sheet where either endpoint has no faces.
The edge (vi, vj) does not exist on that sheet, so UV correspondence is
impossible and unnecessary there.

```cpp
if (sheets_Nsf[si].empty() || sheets_Ndf[si].empty()) continue;
```
