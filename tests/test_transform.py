#!/usr/bin/env python

import unittest
import numpy as np
from phyloshapetest.core import Face, Vector
from phyloshapetest import transform

class TestTransformFunctions(unittest.TestCase):

    def setUp(self):
        # 2D triangle in the XY plane
        self.a = np.array([0.0, 0.0, 0.0])
        self.b = np.array([1.0, 0.0, 0.0])
        self.c = np.array([0.0, 1.0, 0.0])
        self.face_xy = Face(coords=[self.a, self.b, self.c])
        self.vector_xy = Vector(coords=(self.a, self.b), face=self.face_xy)

        # 3D triangle not in the XY plane
        a3 = np.array([1.0, 1.0, 1.0])
        b3 = np.array([2.0, 1.0, 2.0])
        c3 = np.array([1.0, 2.0, 2.0])
        self.face_3d = Face(coords=[a3, b3, c3])
        self.vector_3d = Vector(coords=(a3, b3), face=self.face_3d)

    def test_point_to_relative_and_back_xy(self):
        point = np.array([0.5, 0.5, 0.0])
        rel = self.face_xy.to_relative(point)
        global_point = self.face_xy.to_global(rel)
        np.testing.assert_allclose(global_point, point, rtol=1e-6)

    def test_vector_to_relative_xy(self):
        rel_vec = self.vector_xy.to_relative()
        expected = np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(rel_vec, expected, rtol=1e-6)

    def test_vector_to_global_xy(self):
        rel_vec = np.array([1.0, 0.0, 0.0])
        global_vec = self.vector_xy.to_global(rel_vec)
        expected = np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(global_vec, expected, rtol=1e-6)

    def test_basis_is_orthonormal_xy(self):
        basis = transform._get_face_basis(self.face_xy.coords)
        identity = np.dot(basis.T, basis)
        np.testing.assert_allclose(identity, np.eye(3), rtol=1e-6)

    def test_point_to_relative_and_back_3d(self):
        point = np.array([1.5, 1.5, 2.0])
        rel = self.face_3d.to_relative(point)
        global_point = self.face_3d.to_global(rel)
        np.testing.assert_allclose(global_point, point, rtol=1e-6)

    def test_vector_to_relative_3d(self):
        rel_vec = self.vector_3d.to_relative()
        expected_length = np.linalg.norm(self.vector_3d.coords[1] - self.vector_3d.coords[0])
        np.testing.assert_allclose(np.linalg.norm(rel_vec), expected_length, rtol=1e-6)

    def test_basis_is_orthonormal_3d(self):
        basis = transform._get_face_basis(self.face_3d.coords)
        identity = np.dot(basis.T, basis)
        np.testing.assert_allclose(identity, np.eye(3), rtol=1e-6)

    def test_vector_to_relative_without_face_raises(self):
        v = Vector(coords=(self.a, self.b), face=None)
        with self.assertRaises(ValueError):
            _ = v.to_relative()

    def test_vector_to_global_without_face_raises(self):
        v = Vector(coords=(self.a, self.b), face=None)
        with self.assertRaises(ValueError):
            _ = v.to_global(np.array([1.0, 0.0, 0.0]))


if __name__ == '__main__':
    unittest.main()
