#!/usr/bin/env python

"""Core class types in phyloshape

TODO
----
- general transform methods should take arrays and return arrays
- transform methods of class objects should return class objects.

Classes
-------
Vertex:
    stores coordinates and color data
Face:
    stores collection of Vertex objects composing a face.
Vector:
    stores an edge connecting 2 Vertex objects w/ a reference face
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Sequence
import numpy as np
import transform


@dataclass(frozen=True)
class Vertex:
    """Stores coordinate and optional color data for a vertex."""
    id: int
    coords: np.ndarray
    color: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0], dtype=np.uint8))

    def __post_init__(self):
        if self.coords.shape != (3,):
            raise ValueError("coords must be a 3-element array.")
        if self.color.shape != (3,):
            raise ValueError("color must be a 3-element RGB array.")

    def __repr__(self) -> str:
        return f"Vertex(id={self.id}, coords={tuple(self.coords.round(3))})"


@dataclass(frozen=True)
class Face:
    """Stores three Vertex objects to construct a face from their coordinates
    
    Faces are used for visualization, data storage, and to create new
    reference coordinate systems (see class methods).
    """
    vertices: Tuple[Vertex, Vertex, Vertex]

    def __post_init__(self):
        if len(self.vertices) != 3:
            raise ValueError("Face must contain exactly 3 vertices.")

    def __getitem__(self, idx):
        return self.vertices[idx]

    def __len__(self):
        return 3

    def __repr__(self):
        return f"Face({tuple(v.id for v in self.vertices)})"

    def coords(self) -> List[np.ndarray]:
        return [v.coords for v in self.vertices]

    @property
    def edges(self) -> Tuple[Tuple[Vertex, Vertex]]:
        v = self.vertices
        return ((v[0], v[1]), (v[1], v[2]), (v[0], v[2]))

    @property
    def normal(self) -> np.ndarray:
        a, b, c = self.coords()
        u = b - a
        v = c - a
        cross = np.cross(u, v)
        norm = np.linalg.norm(cross)
        if np.isclose(norm, 0.0):
            raise ValueError("Degenerate face: normal vector magnitude is zero.")
        return cross / norm

    def to_relative(self, point):
        """
        Transform a 3D point to this face's local coordinate system.
        """
        return transform.transform_point_to_relative(self.coords, point)

    def to_global(self, relative_point):
        """
        Transform a point from this face's local system to global coordinates.
        """
        return transform.transform_point_to_global(self.coords, relative_point)


@dataclass
class Vector:
    """Vector stores an edge connecting 2 Vertices and a Face.
    
    A vector can be transformed to a Face-relative coordinate system.
    """
    start: Vertex
    end: Vertex
    face: Optional[Face] = None

    _unit: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _absolute: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _relative: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _dist: Optional[float] = field(default=None, init=False, repr=False)

    @property
    def unit(self) -> np.ndarray:
        if self._unit is None:
            if self.face is None:
                raise ValueError("Vector must have a face to compute a unit vector")
            self._unit = self.face.normal
        return self._unit

    @property
    def dist(self) -> float:
        if self._dist is None:
            diff = (self.end.coords - self.start.coords) ** 2
            self._dist = np.sqrt(np.sum(diff))
        return self._dist

    @property
    def absolute(self) -> np.ndarray:
        if self._absolute is None:
            self._absolute = self.end.coords - self.start.coords
        return self._absolute

    def to_relative(self):
        """
        Transform this vector to the face-relative coordinate system.
        """
        if self.face is None:
            raise ValueError("Vector must have a reference face to compute relative coordinates.")
        return transform.transform_vector_to_relative(self.face.coords, self.coords)

    def to_global(self, relative_vec):
        """
        Transform a vector from face-relative to global coordinates using this vector's face.
        """
        if self.face is None:
            raise ValueError("Vector must have a reference face to compute global coordinates.")
        return transform.transform_vector_to_global(self.face.coords, relative_vec)        

    def __repr__(self):
        return f"Vector({self.start.id}, {self.end.id})"


def transform_vector_to_relative(vector: np.ndarray, face: List[np.ndarray]) -> np.ndarray:
    """
    Transform a vector from global coordinates to a local coordinate system defined by a face.
    The face defines a local coordinate system with origin at face[0], x-axis in the direction
    of face[1] - face[0], and normal to the face as z-axis.

    Parameters
    ----------
    vector : np.ndarray
        The vector to transform (shape (3,))
    face : List[np.ndarray]
        List of 3 vertex coordinate arrays defining a triangle.

    Returns
    -------
    np.ndarray
        The vector in the face-relative coordinate system.
    """
    p0, p1, p2 = face
    x_axis = p1 - p0
    x_axis /= np.linalg.norm(x_axis)
    z_axis = np.cross(x_axis, p2 - p0)
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    R = np.vstack([x_axis, y_axis, z_axis]).T
    return R.T @ vector




# --- Unit Tests and Main ---
if __name__ == "__main__":
    def test_transform_point_to_relative():
        v0 = Vertex(id=0, coords=np.array([0.0, 0.0, 0.0]))
        v1 = Vertex(id=1, coords=np.array([1.0, 0.0, 0.0]))
        v2 = Vertex(id=2, coords=np.array([0.0, 1.0, 0.0]))
        face = Face((v0, v1, v2))
        p = np.array([0.0, 0.0, 1.0])
        rel = face.transform_point_to_relative(p)
        expected = np.array([0.0, 0.0, 1.0])
        assert np.allclose(rel, expected), f"Expected {expected}, got {rel}"
        global_p = face.transform_point_to_global(rel)
        assert np.allclose(global_p, p), f"Expected global {p}, got {global_p}"

    test_transform_point_to_relative()

    def main():
        v0 = Vertex(id=0, coords=np.array([0, 0, 1], dtype=np.float32))
        v1 = Vertex(id=1, coords=np.array([0, 1, 0], dtype=np.float32))
        v2 = Vertex(id=2, coords=np.array([1, 0, 0], dtype=np.float32))
        v3 = Vertex(id=3, coords=np.array([0, 1, 1], dtype=np.float32))

        f0 = Face((v0, v1, v2))
        f1 = Face((v2, v3, v0))

        print(f0)

        V = Vector(v0, v1, f0)
        print(V)
        print(V.dist)
        print(V.absolute)
        print(V.relative)
        print(v0)
        print(Vertex(id=5, coords=np.array([100.2341, 22.0, 0], dtype=np.float32)))

    main()
