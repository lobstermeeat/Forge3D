"""
Drops small pieces that float apart from a model: the export's geometry cleanup.

TRELLIS.2 sometimes leaves a bit of a model hanging in the air beside it: the tip of a dragon's tail on
its own, specks under a cabin's floor. drop_floaters removes a piece of the mesh (a connected component)
only when it is both

- small: under MAX_AREA_SHARE of the model's surface area, and
- apart: farther than MIN_GAP_SHARE of the bounding box's diagonal from every piece that stays.

Pieces within that gap of each other are judged together, as one group: a group that reaches a big
piece stays, and so does a group whose pieces add up to a big share of the surface. So sprinkles sitting
on a donut, chopsticks resting in a bowl, a cup's handle and a balloon's basket on its ropes all stay; a
ball floating beside a donut, a tail tip in the air and specks under a cabin go. The largest piece always
stays, and the floaters together never take more than MAX_AREA_SHARE of the surface: when they would,
the model is several separate things (a pile of coins), not one with floaters, and nothing is dropped.

Distances are between surfaces (each piece's vertices against the other's triangles, both ways), so a
piece resting on one large flat triangle counts as touching it. Vertices split at UV seams are exact
copies and are welded first, so a seam never splits a piece. The faces that stay keep their vertices,
UVs and texture. numpy only.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .normals import _components

# Tuned on Phase 5's 39 finals (see the workers README). Real parts that come loose sit close to the
# model: a fox's tail tip 0.9 % of the diagonal away, a chair's armrests 0.75 %, a balloon's pilot and
# burner up to 1.4 %; sprinkles, pearls and chopsticks touch it (within 0.35 %). The one floater worth
# dropping, a dragon's tail tip, hung 4.9 % away, as two back-to-back sheets of 5.7 % of the surface
MAX_AREA_SHARE = 0.08
MIN_GAP_SHARE = 0.03
# Bounds on the distance search's working memory: entries of a block of points against the triangles
# near it, and point-triangle pairs measured at once
_MATRIX = 1 << 20
_PAIRS = 1 << 16
# Points per block when looking for the pieces near one
_BLOCK = 512
# How many dropped pieces the report describes, largest first
_REPORTED = 8


@dataclass(frozen=True)
class Pieces:
    """A mesh's connected pieces and the verdict on each."""

    face_piece: np.ndarray  # (F,) the piece of each face
    faces: np.ndarray  # (P,) face count per piece
    area_share: np.ndarray  # (P,) share of the whole surface area
    # (P,) distance over the bounding-box diagonal: for a small piece that stays, to the nearest piece it
    # touches (within the gap); for a dropped one, to the nearest piece that stays; 0 for big pieces
    gap_share: np.ndarray
    center: np.ndarray  # (P, 3) bounding-box centre
    keep: np.ndarray  # (P,) bool
    # Per piece: "big", "near" (its group reaches a big piece), "group" (its group is big), "many" (kept
    # because the floaters would add up to too much of the surface), "floater"
    why: tuple

    @property
    def drop_faces(self) -> np.ndarray:
        return ~self.keep[self.face_piece]


def find_pieces(
    vertices: Any,
    faces: Any,
    *,
    max_area_share: float = MAX_AREA_SHARE,
    min_gap_share: float = MIN_GAP_SHARE,
) -> Pieces:
    """The pieces of a triangle mesh, and which of them are floaters (see the module's docstring)."""
    positions = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    tris = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(tris) == 0:
        none = np.zeros(0, dtype=np.int64)
        return Pieces(none, none, none.astype(float), none.astype(float), np.zeros((0, 3)), none.astype(bool), ())

    # Vertices split at UV seams are exact copies of one point: weld them (+ 0.0 makes -0.0 equal 0.0)
    points, weld = np.unique(positions + 0.0, axis=0, return_inverse=True)
    welded = weld.reshape(-1)[tris]
    label = _components(len(points), np.r_[welded[:, 0], welded[:, 1]], np.r_[welded[:, 1], welded[:, 2]])
    _, face_piece = np.unique(label[welded[:, 0]], return_inverse=True)
    count = int(face_piece.max()) + 1
    # Pieces numbered in the order of their first face
    first_face = np.full(count, len(tris))
    np.minimum.at(first_face, face_piece.reshape(-1), np.arange(len(tris)))
    rank = np.empty(count, dtype=np.int64)
    rank[np.argsort(first_face, kind="stable")] = np.arange(count)
    face_piece = rank[face_piece.reshape(-1)]

    corners = points[welded]
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    area = np.sqrt(np.einsum("ij,ij->i", cross, cross)) / 2
    piece_area = np.bincount(face_piece, weights=area, minlength=count)
    area_share = piece_area / max(float(piece_area.sum()), 1e-300)
    faces_per_piece = np.bincount(face_piece, minlength=count)

    # Each piece's bounding box, and the whole mesh's diagonal
    point_piece = np.full(len(points), -1, dtype=np.int64)  # -1: no face uses the point
    point_piece[welded] = face_piece[:, None]
    used = point_piece >= 0
    lo = np.full((count, 3), np.inf)
    hi = np.full((count, 3), -np.inf)
    np.minimum.at(lo, point_piece[used], points[used])
    np.maximum.at(hi, point_piece[used], points[used])
    diagonal = float(np.linalg.norm(points[used].max(axis=0) - points[used].min(axis=0)))
    gap = min_gap_share * diagonal

    big = area_share >= max_area_share
    big[int(np.argmax(piece_area))] = True
    gap_share = np.where(big, 0.0, np.inf)
    why = np.where(big, "big", "floater").astype(object)
    if big.all() or diagonal <= 0:
        return Pieces(face_piece, faces_per_piece, area_share, gap_share, (lo + hi) / 2, big, tuple(why))

    search = _Search(points, welded, face_piece)
    # Pairs of pieces within the gap of each other, found from each small piece
    first, second = [np.zeros(0, np.int64)], [np.zeros(0, np.int64)]
    for piece in np.flatnonzero(~big):
        near, distance = search.near(piece, lo[piece], hi[piece], gap)
        if len(near):
            first.append(np.full(len(near), piece))
            second.append(near)
            gap_share[piece] = distance.min() / diagonal
    group = _components(count, np.concatenate(first), np.concatenate(second))
    group_area = np.bincount(group, weights=area_share, minlength=count)
    group_big = np.zeros(count, dtype=bool)
    np.logical_or.at(group_big, group, big)
    keep = big | group_big[group] | (group_area[group] >= max_area_share)
    why[~big & group_big[group]] = "near"
    why[~big & ~group_big[group] & keep] = "group"
    if area_share[~keep].sum() > max_area_share:
        # Floaters adding up to this much are separate things (a pile of coins), not bits that came loose
        why[~keep] = "many"
        keep[:] = True

    # How far each dropped piece is from what stays, for the report
    stays = keep[face_piece]
    for piece in np.flatnonzero(~keep):
        gap_share[piece] = search.distance(piece, stays, lo[piece], hi[piece], gap, diagonal) / diagonal
    return Pieces(face_piece, faces_per_piece, area_share, gap_share, (lo + hi) / 2, keep, tuple(why))


def drop_floaters(
    mesh: Any,
    *,
    max_area_share: float = MAX_AREA_SHARE,
    min_gap_share: float = MIN_GAP_SHARE,
) -> dict:
    """
    Removes the floating pieces of ``mesh`` (a trimesh mesh, as o_voxel's to_glb returns it) in place and
    says what went: how many pieces the mesh had and how many were dropped, their faces and share of the
    surface area, and the largest of them (area and gap shares, faces, centre).
    """
    started = time.perf_counter()
    pieces = find_pieces(mesh.vertices, mesh.faces, max_area_share=max_area_share, min_gap_share=min_gap_share)
    drop = pieces.drop_faces
    if drop.any():
        mesh.update_faces(~drop)
        mesh.remove_unreferenced_vertices()
    dropped = np.flatnonzero(~pieces.keep)
    largest = dropped[np.argsort(-pieces.area_share[dropped], kind="stable")][:_REPORTED]
    return {
        "pieces": int(len(pieces.keep)),
        "dropped": int(len(dropped)),
        "faces_dropped": int(drop.sum()),
        "area_dropped": round(float(pieces.area_share[dropped].sum()), 5),
        "floaters": [
            {
                "area_share": round(float(pieces.area_share[i]), 5),
                "gap_share": round(float(pieces.gap_share[i]), 4),
                "faces": int(pieces.faces[i]),
                "center": [round(float(c), 4) for c in pieces.center[i]],
            }
            for i in largest
        ],
        "seconds": round(time.perf_counter() - started, 3),
    }


class _Search:
    """Distances between one piece and the rest of the mesh, looking only at what lies near the piece."""

    def __init__(self, points: np.ndarray, welded: np.ndarray, face_piece: np.ndarray) -> None:
        self.points = points
        self.welded = welded
        self.face_piece = face_piece
        # Pieces are joined by shared points, so every point used belongs to exactly one piece
        self.point_piece = np.zeros(len(points), dtype=np.int64)
        self.point_piece[welded] = face_piece[:, None]
        self.count = int(face_piece.max()) + 1
        corners = points[welded]
        self.face_lo = corners.min(axis=1)
        self.face_hi = corners.max(axis=1)
        order = np.argsort(face_piece, kind="stable")
        bounds = np.searchsorted(face_piece[order], np.arange(int(face_piece.max()) + 2))
        self.piece_faces = [order[bounds[i] : bounds[i + 1]] for i in range(len(bounds) - 1)]

    def _around(self, piece: int, others: np.ndarray, lo: np.ndarray, hi: np.ndarray, reach: float):
        """
        The piece's points and faces, and the faces of ``others`` (a per-face mask) and their points
        within ``reach`` of the piece's box lo..hi.
        """
        faces = np.flatnonzero(
            others & np.all(self.face_hi >= lo - reach, axis=1) & np.all(self.face_lo <= hi + reach, axis=1)
        )
        theirs = np.unique(self.welded[faces])
        box = self.points[theirs]
        theirs = theirs[np.all(box >= lo - reach, axis=1) & np.all(box <= hi + reach, axis=1)]
        own = self.piece_faces[piece]
        return np.unique(self.welded[own]), own, theirs, faces

    def _triangles(self, faces: np.ndarray) -> np.ndarray:
        return self.points[self.welded[faces]]

    def near(self, piece: int, lo: np.ndarray, hi: np.ndarray, reach: float):
        """
        The other pieces within ``reach`` of ``piece`` (its box lo..hi), each with a distance within reach
        (the least found, not always the least there is). Vertices first: a vertex of the other piece that
        close to one of this piece's settles it cheaply. Only the pieces left are measured exactly, a block
        of points at a time both ways, each piece found leaving the blocks after it.
        """
        mine, own, theirs, faces = self._around(piece, self.face_piece != piece, lo, hi, reach)
        best = np.full(self.count, np.inf)
        own_points = self.points[mine]
        squares = (own_points**2).sum(axis=1)
        block = max(1, _MATRIX // max(len(mine), 1))
        for start in range(0, len(theirs), block):
            chunk = theirs[start : start + block]
            theirs_points = self.points[chunk]
            span = (theirs_points**2).sum(axis=1)[:, None] + squares[None] - 2 * theirs_points @ own_points.T
            closest = np.sqrt(np.maximum(span.min(axis=1), 0.0))
            within = closest <= reach
            np.minimum.at(best, self.point_piece[chunk[within]], closest[within])
        own_triangles = self._triangles(own)
        for points, mine_first in ((mine, True), (theirs, False)):
            for start in range(0, len(points), _BLOCK):
                chunk = points[start : start + _BLOCK]
                if mine_first:  # this piece's points against the faces of pieces not found yet
                    live = faces[~np.isfinite(best[self.face_piece[faces]])]
                    (_, face), distance = _point_triangle(self.points[chunk], self._triangles(live), reach)
                    met = self.face_piece[live[face]]
                else:  # the points of pieces not found yet against this piece's faces
                    chunk = chunk[~np.isfinite(best[self.point_piece[chunk]])]
                    (point, _), distance = _point_triangle(self.points[chunk], own_triangles, reach)
                    met = self.point_piece[chunk[point]]
                np.minimum.at(best, met, distance)
        pieces = np.flatnonzero(np.isfinite(best))
        return pieces, best[pieces]

    def distance(
        self, piece: int, stays: np.ndarray, lo: np.ndarray, hi: np.ndarray, reach: float, limit: float
    ) -> float:
        """The least distance from ``piece`` to the faces in ``stays``, searching ever wider from ``reach``."""
        others = stays & (self.face_piece != piece)
        while reach <= 2 * limit:
            mine, own, theirs, faces = self._around(piece, others, lo, hi, reach)
            found = _nearest(self.points[mine], self._triangles(faces), reach)
            found = _nearest(self.points[theirs], self._triangles(own), min(reach, found), found)
            if found <= reach:
                return found
            reach *= 2
        return math.inf


def _candidates(points: np.ndarray, triangles: np.ndarray, reach: float):
    """
    Blocks of (point, triangle) index pairs that may lie within ``reach`` of each other, by the distance
    to each triangle's centre, in bounded memory.
    """
    if not len(points) or not len(triangles):
        return
    centers = triangles.mean(axis=1)
    radii = np.sqrt(((triangles - centers[:, None]) ** 2).sum(axis=2)).max(axis=1)
    squares = (centers**2).sum(axis=1)
    block = max(1, min(len(points), _MATRIX // len(triangles)))
    for start in range(0, len(points), block):
        chunk = points[start : start + block]
        span = (chunk**2).sum(axis=1)[:, None] + squares[None] - 2 * chunk @ centers.T
        row, col = np.nonzero(span <= (reach + radii[None]) ** 2)
        for begin in range(0, len(row), _PAIRS):
            yield row[begin : begin + _PAIRS] + start, col[begin : begin + _PAIRS]


def _point_triangle(points: np.ndarray, triangles: np.ndarray, reach: float):
    """The (point, triangle) index pairs within ``reach`` of each other, and their distances."""
    rows, cols, found = [np.zeros(0, np.int64)], [np.zeros(0, np.int64)], [np.zeros(0)]
    for row, col in _candidates(points, triangles, reach):
        distance = _distances(triangles[col], points[row])
        within = distance <= reach
        rows.append(row[within])
        cols.append(col[within])
        found.append(distance[within])
    return (np.concatenate(rows), np.concatenate(cols)), np.concatenate(found)


def _nearest(points: np.ndarray, triangles: np.ndarray, reach: float, best: float = math.inf) -> float:
    """The least point-triangle distance if it is within ``reach`` (or ``best``, if that is less), else inf."""
    for row, col in _candidates(points, triangles, min(reach, best)):
        best = min(best, float(_distances(triangles[col], points[row]).min()))
    return best if best <= reach else math.inf


def _distances(triangles: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Each point's distance to its triangle; a degenerate one (a line or a point) is measured at its corners."""
    from trimesh.triangles import closest_point

    with np.errstate(invalid="ignore", divide="ignore"):
        distance = np.linalg.norm(closest_point(triangles, points) - points, axis=1)
    lost = ~np.isfinite(distance)
    if lost.any():
        distance[lost] = np.linalg.norm(triangles[lost] - points[lost][:, None], axis=2).min(axis=1)
    return distance
