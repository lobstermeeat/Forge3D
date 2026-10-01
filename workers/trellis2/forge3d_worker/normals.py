"""
Shading normals for exported meshes: smooth where the surface is meant to be smooth, crisp at real edges.

The remesh is faithful to what the generator drew, and the generator often draws curved surfaces as
wide flat facets. A potion's bulb came out as a 32-sided polygon around and along it: facets some 70
voxels wide, meeting at about 11 degrees. Vertex normals averaged over one ring of triangles keep each
facet flat, so a glossy material mirrors the room once per facet, as a grid of blocks.

So the face normals are filtered first, with a bilateral filter iterated over each face's ring of
neighbours until it has spread SMOOTH_VOXELS: neighbours whose normals differ by less than about
FACET_ANGLE are averaged, larger differences are left alone. Shallow faceting melts into a smooth
surface, while low-poly facets, panel lines and bevels keep their shape. Vertex normals are then the
area-weighted average of the filtered normals around each vertex, except across edges sharper than
HARD_ANGLE (a box's corners, the two sides of a thin sheet), where the vertex is split so that each side
keeps its own normal. Vertices split at UV seams are welded first (they are exact copies), so both sides
of a seam get the same normal and seams never show.

Positions are never moved. The filter is numpy only: well under a second for 100k faces on one core.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

# How far the smoothing reaches (the diffusion length of the filter), in voxels of the mesh's grid
SMOOTH_VOXELS = 32.0
# Normal differences (degrees) up to about this are smoothed, larger ones kept: neighbours this far apart
# count 0.6 as much as alike ones, 1.4 times as far apart 0.15, twice as far apart nothing
FACET_ANGLE = 10.0
# Edges whose faces meet at more than this (degrees) stay crisp: each side gets its own vertex normal
HARD_ANGLE = 60.0
# Upper bound on the filter's steps, whatever the tessellation (about 30 ms each at 100k faces)
MAX_ITERATIONS = 40

# A triangle this thin (4·sqrt(3)·area / sum of squared edge lengths; 1 = equilateral) or this small
# (in square voxels) has an unreliable normal: it never makes an edge hard and takes no part in the filter
_SLIVER_QUALITY = 0.05
_TINY_AREA_VOXELS = 1e-3


@dataclass(frozen=True)
class ShadingNormals:
    """A vertex list of copies of the input's: output vertex i copies input vertex source[i]."""

    source: np.ndarray  # (K,) int64; the first len(vertices) entries are 0, 1, 2, ...
    faces: np.ndarray  # (F, 3) int64: the input faces over the output vertices
    normals: np.ndarray  # (K, 3) float32 unit vectors


def shading_normals(
    vertices: Any,
    faces: Any,
    voxel_size: float,
    smooth_voxels: float = SMOOTH_VOXELS,
    facet_angle: float = FACET_ANGLE,
    hard_angle: float = HARD_ANGLE,
    max_iterations: int = MAX_ITERATIONS,
) -> ShadingNormals:
    """
    Smooth, feature-preserving vertex normals for a triangle mesh whose vertices may be split at UV seams.

    Input vertices keep their indices. A vertex with corners on both sides of a hard edge gets a copy per
    extra side, appended after the input vertices; faces are re-indexed to use them.
    """
    positions = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    tris = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    count = len(positions)
    if len(tris) == 0:
        return ShadingNormals(np.arange(count), tris, _fallback(count).astype(np.float32))

    # Vertices split at UV seams are exact copies of one point: weld them (+ 0.0 makes -0.0 equal 0.0)
    _, weld = np.unique(positions + 0.0, axis=0, return_inverse=True)
    welded = weld.reshape(-1)[tris]

    corners = positions[tris]
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    double_area = np.sqrt(np.einsum("ij,ij->i", cross, cross))
    area = double_area / 2
    normal = cross / np.maximum(double_area, 1e-300)[:, None]
    centroid = corners.mean(axis=1)
    edges = corners[:, [1, 2, 0]] - corners
    quality = 2 * math.sqrt(3) * double_area / np.maximum(np.einsum("ijk,ijk->i", edges, edges), 1e-300)
    weak = (quality < _SLIVER_QUALITY) | (area < _TINY_AREA_VOXELS * voxel_size**2)

    group, groups = _corner_groups(welded, normal, weak, math.cos(math.radians(hard_angle)))
    first, second = _neighbour_pairs(group, groups, len(tris))

    # Every neighbour counts the same in the filter, not by its area: decimation leaves small triangles
    # along creases and large ones inside facets, and with area weights facets would hear each other
    # only through the small ones, keeping half the faceting however long the filter ran
    filtered = _bilateral(
        normal,
        np.where(weak, 0.0, 1.0),
        area,
        centroid,
        first,
        second,
        reach=smooth_voxels * voxel_size,
        facet_cos=math.cos(math.radians(facet_angle)),
        max_iterations=max_iterations,
    )

    # Each corner group's normal: the area-weighted average of its faces' filtered normals
    corner_face = np.repeat(np.arange(len(tris)), 3)
    corner_weight = area[corner_face]
    group_normal = _normalize(_sum_rows(group, corner_weight[:, None] * filtered[corner_face], groups))
    # A group of zero-area faces only takes the average normal of its vertex
    lost = ~np.any(group_normal != 0, axis=1)
    if lost.any():
        flat_welded = welded.reshape(-1)
        weighted = corner_weight[:, None] * normal[corner_face]
        vertex_normal = _normalize(_sum_rows(flat_welded, weighted, int(weld.max()) + 1))
        group_vertex = np.empty(groups, dtype=np.int64)
        group_vertex[group] = flat_welded
        group_normal[lost] = vertex_normal[group_vertex[lost]]
        # and a vertex of zero-area faces only (never drawn) any unit vector
        group_normal[~np.any(group_normal != 0, axis=1)] = _fallback(1)
    return _split(tris.reshape(-1), group, group_normal, count)


def with_shading_normals(mesh: Any, voxel_size: float) -> Any:
    """
    A textured trimesh mesh (what o_voxel's to_glb returns) as a new one with shading_normals' vertex
    normals, which its GLB export writes as NORMAL. Positions, UVs and the material are kept; vertices
    split at hard edges are copies.
    """
    import trimesh

    result = shading_normals(mesh.vertices, mesh.faces, voxel_size)
    visual = mesh.visual
    uv = getattr(visual, "uv", None)
    return trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices)[result.source],
        faces=result.faces,
        vertex_normals=result.normals,
        visual=trimesh.visual.TextureVisuals(
            uv=None if uv is None else np.asarray(uv)[result.source],
            material=getattr(visual, "material", None),
        ),
        process=False,
    )


def _corner_groups(welded: np.ndarray, normal: np.ndarray, weak: np.ndarray, hard_cos: float):
    """
    Corner (face, vertex) groups: corners around a welded vertex that are connected across smooth edges.
    An edge is smooth when exactly two consistently oriented faces share it and they meet at less than
    the hard angle (or one of them is too thin to tell). Corner 3·f + k is slot k of face f.
    """
    start = welded.reshape(-1)  # half-edge 3·f + k runs from corner k of face f to corner k + 1
    end = welded[:, [1, 2, 0]].reshape(-1)
    key = np.minimum(start, end) * (int(welded.max()) + 1) + np.maximum(start, end)
    order = np.argsort(key, kind="stable")
    begins = np.flatnonzero(_starts(key[order]))
    sizes = np.diff(np.r_[begins, len(key)])
    pair = begins[sizes == 2]
    one, two = order[pair], order[pair + 1]
    # Two sheets back to back (the remesher's take on thin parts such as ears) give their edges four
    # half-edges. Pair each a→b with the b→a whose face points the same way, so that each sheet is
    # smooth on its own: unpaired, every triangle would be flat; paired the other way, they would cancel
    quad = order[begins[sizes == 4][:, None] + np.arange(4)]
    forward = start[quad] < end[quad]
    quad, forward = quad[forward.sum(axis=1) == 2], forward[forward.sum(axis=1) == 2]
    quad = np.take_along_axis(quad, np.argsort(~forward, axis=1, kind="stable"), axis=1)  # forward ones first
    a1, a2, b1, b2 = (normal[quad[:, k] // 3] for k in range(4))
    straight = np.einsum("ij,ij->i", a1, b1) + np.einsum("ij,ij->i", a2, b2)
    crossed = np.einsum("ij,ij->i", a1, b2) + np.einsum("ij,ij->i", a2, b1)
    swap = crossed > straight
    one = np.r_[one, quad[:, 0], quad[:, 1]]
    two = np.r_[two, np.where(swap, quad[:, 3], quad[:, 2]), np.where(swap, quad[:, 2], quad[:, 3])]
    f1, f2 = one // 3, two // 3
    alike = np.einsum("ij,ij->i", normal[f1], normal[f2]) >= hard_cos
    smooth = (start[one] == end[two]) & (alike | weak[f1] | weak[f2])
    one, two = one[smooth], two[smooth]
    # Half-edge one runs a→b and two runs b→a: join their corners at a, and their corners at b
    after_one = one - one % 3 + (one % 3 + 1) % 3
    after_two = two - two % 3 + (two % 3 + 1) % 3
    label = _components(3 * len(welded), np.r_[one, after_one], np.r_[after_two, two])
    root = label == np.arange(len(label))
    dense = np.cumsum(root) - 1
    return dense[label], int(root.sum())


def _components(size: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Connected components of the graph with edges a–b: each node's label is its component's least node."""
    label = np.arange(size)
    while True:
        la, lb = label[a], label[b]
        if np.array_equal(la, lb):
            return label
        # Hook the larger root onto the smaller, then point every node straight at its root
        np.minimum.at(label, np.maximum(la, lb), np.minimum(la, lb))
        while True:
            jumped = label[label]
            if np.array_equal(jumped, label):
                break
            label = jumped


def _neighbour_pairs(group: np.ndarray, groups: int, tri_count: int):
    """Unordered pairs of distinct faces sharing a corner group (a vertex, on one side of its hard edges)."""
    order = np.argsort(group, kind="stable")
    member = order // 3  # the face of each corner, in group order
    sorted_group = group[order]
    sizes = np.bincount(group, minlength=groups)
    begins = np.cumsum(sizes) - sizes
    # Every corner paired with every corner of its group
    repeat = sizes[sorted_group]
    left = np.repeat(np.arange(len(order)), repeat)
    offset = np.arange(len(left)) - np.repeat(np.cumsum(repeat) - repeat, repeat)
    right = begins[sorted_group[left]] + offset
    a, b = member[left], member[right]
    keep = a < b
    # Faces sharing an edge share two groups: keep each pair once (a sort; np.unique is far slower)
    key = np.sort(a[keep] * tri_count + b[keep])
    key = key[_starts(key)]
    return key // tri_count, key % tri_count


def _bilateral(
    normal: np.ndarray,
    weight: np.ndarray,
    area: np.ndarray,
    centroid: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    reach: float,
    facet_cos: float,
    max_iterations: int,
) -> np.ndarray:
    """
    Iterated bilateral filter of face normals over each face's ring of neighbours: each step averages a
    face with its neighbours (each counting `weight`, times how alike their current normals are), and
    steps repeat until the averaging has spread `reach`, as the standard deviation of a random walk.
    """
    tri_count = len(normal)
    usable = (weight > 0) & (area > 0)
    if reach <= 0 or len(first) == 0 or not usable.any():
        return normal
    # One step's spread at a face: its neighbours' mean squared distance per surface axis (so halved),
    # diluted by the face's own weight; the typical step is the average over the surface's area
    distance_sq = np.square(centroid[first] - centroid[second]).sum(axis=1)
    w_first, w_second = weight[first], weight[second]
    total = weight + np.bincount(first, w_second, tri_count) + np.bincount(second, w_first, tri_count)
    moved = np.bincount(first, w_second * distance_sq, tri_count)
    moved += np.bincount(second, w_first * distance_sq, tri_count)
    typical = np.average(moved[usable] / total[usable] / 2, weights=area[usable])
    if typical <= 0:
        return normal
    steps = int(min(max_iterations, max(1, math.ceil(reach**2 / typical))))

    inverse_scale = 1 / (1 - facet_cos)  # (1 - cos θ) relative to its value at the facet angle
    x, y, z = (np.ascontiguousarray(normal[:, k]) for k in range(3))
    for _ in range(steps):
        x1, y1, z1, x2, y2, z2 = x[first], y[first], z[first], x[second], y[second], z[second]
        t = (1 - (x1 * x2 + y1 * y2 + z1 * z2)) * inverse_scale
        # Flat near no difference, then falling steeply: 0.6 at the facet angle, 0.0003 at twice it
        alike = np.exp(-0.5 * t * t)
        to_first, to_second = alike * w_second, alike * w_first
        sx, sy, sz = (
            own * weight + np.bincount(first, to_first * theirs, tri_count)
            + np.bincount(second, to_second * mine, tri_count)
            for own, mine, theirs in ((x, x1, x2), (y, y1, y2), (z, z1, z2))
        )
        length = np.sqrt(sx * sx + sy * sy + sz * sz)
        # A face with nothing to average (no weight, no usable neighbours) keeps its normal
        keep = length > 0
        safe = np.where(keep, length, 1.0)
        x, y, z = np.where(keep, sx / safe, x), np.where(keep, sy / safe, y), np.where(keep, sz / safe, z)
    return np.stack([x, y, z], axis=1)


def _split(corner_vertex: np.ndarray, group: np.ndarray, group_normal: np.ndarray, count: int):
    """One output vertex per (input vertex, corner group); a vertex keeps its index for its first group."""
    groups = len(group_normal)
    key = corner_vertex * groups + group
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    new = _starts(sorted_key)
    unique = sorted_key[new]
    inverse = np.empty(len(key), dtype=np.int64)
    inverse[order] = np.cumsum(new) - 1
    vertex, vertex_group = unique // groups, unique % groups
    primary = _starts(vertex)
    extra = ~primary
    index = np.empty(len(unique), dtype=np.int64)
    index[primary] = vertex[primary]
    index[extra] = count + np.arange(int(extra.sum()))
    normals = _fallback(count + int(extra.sum()))
    normals[vertex[primary]] = group_normal[vertex_group[primary]]
    normals[count:] = group_normal[vertex_group[extra]]
    source = np.r_[np.arange(count), vertex[extra]]
    return ShadingNormals(source, index[inverse].reshape(-1, 3), normals.astype(np.float32))


def _starts(sorted_values: np.ndarray) -> np.ndarray:
    """Where each run of equal values begins in a sorted array."""
    starts = np.ones(len(sorted_values), dtype=bool)
    starts[1:] = sorted_values[1:] != sorted_values[:-1]
    return starts


def _sum_rows(index: np.ndarray, rows: np.ndarray, size: int) -> np.ndarray:
    """Sums of `rows` (n, 3) per index value, as a (size, 3) array."""
    return np.stack([np.bincount(index, rows[:, k], size) for k in range(3)], axis=1)


def _normalize(v: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(v, axis=1, keepdims=True)
    return np.where(length > 0, v / np.maximum(length, 1e-300), 0.0)


def _fallback(count: int) -> np.ndarray:
    """Normals for vertices that no face uses (any unit vector will do: they are never drawn)."""
    out = np.zeros((count, 3))
    out[:, 1] = 1
    return out
