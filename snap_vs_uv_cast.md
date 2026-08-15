# SNAP vs UV Cast — Why SNAP exists in the F2C walk

## The question

In the F2C (fine-to-coarse) walk, every step is logically:
> "Before this collapse, where was the tracked point? After the collapse, where does it map to?"

That is exactly what the UV cast does — embed the current BC into UV_pre, cast to UV_post, pick the best containing face and BC. So why do we need a special SNAP operation at all? When the absorbed vertex `v_d` is gone, other triangles should expand to cover its domain. All we have to do is find which post-collapse face the UV coordinate of `v_d` falls into and continue from there.

---

## First answer (partially wrong)

Initial reasoning: the UV parameterization is degenerate *at* `v_d` because the two collapse triangles — the ones sharing the edge `(v_d, v_s)` — are deleted in UV_post. `v_d` sits at their shared apex. So its UV_pre coordinate falls inside a region that has no face in UV_post, and the UV cast fails there.

---

## The correction

The user's pushback was correct: when the two collapse triangles are deleted from UV_post, the remaining faces reorganize to cover that region. If the pre/post parameterizations share the same 2D domain, the UV coordinate of `v_d` in UV_pre *would* land inside a valid face in UV_post. The cast wouldn't be undefined — it would find a face.

The code already handles any gap gracefully: `argmax(B.min(axis=1))` always picks the "best" face even if the query point is outside all faces, so it never crashes.

---

## What SNAP is actually doing

SNAP is correct for two distinct reasons:

### 1. Independent parameterizations may not cover the same 2D domain

SSP parameterizes UV_pre and UV_post **independently** — both rings are freshly flattened from 3D geometry into a shared 2D coordinate space, but there is no guarantee the UV_post faces cover exactly the same 2D area as the UV_pre faces. The two parameterizations are different flat maps of topologically distinct surfaces (pre-collapse ring vs post-collapse ring). So `v_d`'s UV_pre position *may* fall in a gap or poorly-covered region of UV_post, making the cast give an imprecise answer.

### 2. BC drift accumulation

Even if the UV cast would technically land somewhere valid near `v_d`, the tracked point may have already drifted slightly away from `v_d` across prior steps (each UV cast introduces a small numerical error). By the time the collapse that absorbs `v_d` is reached, the BC might be `(0.798, 0.202, ~0)` instead of `(1.0, 0.0, 0.0)`. The UV cast would then embed a drifted position and cast it to an imprecise location.

SNAP resets the BC to exact one-hot on `v_s`, stopping drift accumulation cold.

---

## Summary

| Case | Mechanism | Why |
|------|-----------|-----|
| Interior point | UV cast | Bijection is well-defined; cast is accurate |
| Absorbed vertex `v_d` | SNAP | Semantic answer is exact (`v_d → v_s`); avoids both potential UV domain gaps and accumulated BC drift |

SNAP is not correcting a hard failure of the UV cast — the cast would find *something* even at `v_d`. SNAP is the **numerically exact shortcut** for the one case where the semantic answer is known without any parameterization arithmetic.

The `vid` state variable is what makes SNAP reliable across multi-hop collapses: it tracks which coarse vertex the sample currently represents (starts at `vi`, updates to `global_s` on each SNAP), so the condition `vid == global_d` fires on vertex identity regardless of how much BC has drifted.
