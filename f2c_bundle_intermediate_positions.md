# F2C Bundle: Intermediate Positions and What Gets Stored

## What the bundle stores vs. what it doesn't

The bundle stores only the **UV topology and UV geometry** of each collapse's one-ring:

- `UV_pre` / `UV_post` — 2D geometry of the one-ring before/after
- `FUV_pre` / `FUV_post` — UV face connectivity
- `FIdx_pre` / `FIdx_post` — which 3D face indices were in the ring
- `subsetVIdx` — which global vertex indices the ring involves
- `b` — absorbed/survivor local indices

The **3D world-space positions** of the ring vertices at each step are NOT stored. When a collapse happens, the survivor vertex `v_s` moves — its position in `gV` is overwritten with the new collapse target. That transient position is never written to disk.

---

## Topology vs. position — the key distinction

**What IS preserved**: face and vertex *indices*. During decimation, faces are marked dead but their index slots remain. Vertex indices (`global_d`, `global_s`) are never freed. The bundle records exactly which face indices were in the one-ring at each collapse (`FIdx_pre`, `FIdx_post`, `subsetVIdx`). The full topological history is recoverable.

**What is NOT preserved**: the 3D *position* of each vertex at each intermediate step.

So after full decimation:
- `gV[global_s]` = final coarse position (overwritten many times during decimation)
- `gVO[vi]` = original fine position (never touched)
- Bundle step `ci` = `UV_pre`/`UV_post` (2D geometry of the ring at that step, not 3D world)

`decIM` is just a reverse index — "for face F, which collapse indices touched it." It's a lookup table used to find the next relevant collapse in the F2C walk, not a data store.

---

## What each F2C mapping step actually produces

Each step produces exactly two things:
- **face** — which global face index the point lands on in the post-collapse topology
- **BC** — the three barycentric coordinates on that face

The 3D position is never stored — it's computed on demand by interpolating `_v(BF[k])` with BC, and `_v` itself is an approximation since the true intermediate positions aren't in the bundle.

---

## Why the changing face geometry isn't a problem for the mapping

The face **index** is stable and permanent throughout decimation — face 1092 is always face 1092, never renumbered. Faces only get marked dead, never reassigned.

The face's **geometric realization** does change (when `global_s` moves, any face with `global_s` as a corner changes shape in 3D). But the F2C walk doesn't care — it operates entirely in UV space:

- **Before**: embed BC into `UV_pre` → get a 2D point `uv_q`
- **After**: find where `uv_q` lands in `UV_post` → get new face + new BC

`UV_pre` and `UV_post` are freshly computed 2D parameterizations of the ring **at that exact collapse moment** — they already reflect where the ring vertices were in 3D at step `ci`, baked into the 2D layout. The bijection between them is geometrically correct for that step.

The global face index is just a **label** — which triangle in the connectivity the point is on. The actual geometry used during the cast is `UV_pre`/`UV_post` from the bundle, not the live 3D positions of `gV`.

The 3D position problem only appears when you try to **visualize** the intermediate result by calling `_v()` — that's where the approximation happens, because `gV` has moved on. The mapping itself (face + BC) is exact with respect to the UV parameterization stored at that step.

---

## Options to recover true intermediate 3D positions

1. **Extend the bundle during decimation** — at each collapse, before moving `v_s`, record the 3D positions of the ring vertices (`subsetVIdx`). Then `_v(idx)` at step `ci` looks up the last stored position of `idx` before `ci`.

2. **Store just the survivor's new position per collapse** — lighter version of (1). At each collapse, record where `v_s` moved to. This is enough because only `v_s` moves; all other ring vertices stay put. One 3D vector per collapse — minimal bundle change.

3. **Replay from scratch** — start from `gVO`, re-apply each collapse's position update in sequence. No bundle changes needed, but requires storing or recomputing the collapse target position sequence.

4. **Approximate** — what `_v()` does now: use `fineV` for non-surviving vertices and `coarseV` for surviving ones. Geometrically wrong at intermediate steps but costs nothing extra.

Option 2 is the minimal bundle change and gives exact intermediate positions.
