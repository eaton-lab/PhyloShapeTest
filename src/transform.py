#!/usr/bin/env python

"""Simple transformation functions from/to relative/global coordinates.
"""

import numpy as np
from typing import Tuple

def transform_point_to_relative(face_coords: np.ndarray, point: np.ndarray) -> np.ndarray:
    """
    Transform a global 3D point into the coordinate system relative to the triangle defined by face_coords.

    Parameters:
        face_coords: np.ndarray of shape (3, 3)
        point: np.ndarray of shape (3,)
    
    Returns:
        np.ndarray of shape (3,)
    """
    origin = face_coords[0]
    basis = _get_face_basis(face_coords)
    return basis.T @ (point - origin)

def transform_point_to_global(face_coords: np.ndarray, relative_point: np.ndarray) -> np.ndarray:
    """
    Transform a point from face-relative to global coordinates.
    
    Parameters:
        face_coords: np.ndarray of shape (3, 3)
        relative_point: np.ndarray of shape (3,)
    
    Returns:
        np.ndarray of shape (3,)
    """
    origin = face_coords[0]
    basis = _get_face_basis(face_coords)
    return origin + basis @ relative_point

def transform_vector_to_relative(face_coords: np.ndarray, vector_coords: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """
    Transform a vector into face-relative coordinates.
    
    Parameters:
        face_coords: np.ndarray of shape (3, 3)
        vector_coords: Tuple of (start, end) np.ndarrays
    
    Returns:
        np.ndarray of shape (3,)
    """
    vec = vector_coords[1] - vector_coords[0]
    basis = _get_face_basis(face_coords)
    return basis.T @ vec

def transform_vector_to_global(face_coords: np.ndarray, relative_vec: np.ndarray) -> np.ndarray:
    """
    Transform a vector from face-relative to global coordinates.
    
    Parameters:
        face_coords: np.ndarray of shape (3, 3)
        relative_vec: np.ndarray of shape (3,)
    
    Returns:
        np.ndarray of shape (3,)
    """
    basis = _get_face_basis(face_coords)
    return basis @ relative_vec

def _get_face_basis(face_coords: np.ndarray) -> np.ndarray:
    """
    Compute a local orthonormal basis (3x3 matrix) for a triangle.

    Parameters:
        face_coords: np.ndarray of shape (3, 3)

    Returns:
        np.ndarray of shape (3, 3) with orthonormal basis vectors as columns
    """
    a, b, c = face_coords
    v1 = b - a
    v2 = c - a
    x_axis = v1 / np.linalg.norm(v1)
    normal = np.cross(v1, v2)
    z_axis = normal / np.linalg.norm(normal)
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack((x_axis, y_axis, z_axis))

# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Demonstrate basic usage
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    c = np.array([0.0, 1.0, 0.0])
    face_coords = np.array([a, b, c])

    point = np.array([0.5, 0.5, 0.0])
    rel = transform_point_to_relative(face_coords, point)
    back = transform_point_to_global(face_coords, rel)

    print("Original point:", point)
    print("Relative coords:", rel)
    print("Back to global :", back)

    vstart = np.array([0.0, 0.0, 0.0])
    vend = np.array([1.0, 0.0, 0.0])
    rel_vec = transform_vector_to_relative(face_coords, (vstart, vend))
    print("Relative vector:", rel_vec)
