import numpy as np
from typing import Tuple


def _rotation_matrix_from_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return rotation matrix that aligns vector a to vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)

    if s == 0:
        return np.eye(3)

    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)


def _generate_horn_path(
    length: float, twist: float, curve_x: float, curve_y: float, num_discs: int
) -> np.ndarray:
    """Generate 3D path of the horn's centerline."""
    t = np.linspace(0, 1, num_discs)
    z = t * length
    angle = twist * t
    x = curve_x * (1 - np.cos(angle))
    y = curve_y * np.sin(angle)
    return np.column_stack([x, y, z])


def _generate_disc_points(
    origin: np.ndarray,
    normal: np.ndarray,
    radius: float,
    num_points: int,
) -> np.ndarray:
    """
    Generate a ring of 3D points lying on a plane orthogonal to normal.

    Parameters:
        origin: (3,) center of the disc
        normal: (3,) vector perpendicular to the disc
        radius: float, radius of the disc
        num_points: number of points on the disc

    Returns:
        np.ndarray of shape (num_points, 3)
    """
    normal = normal / np.linalg.norm(normal)
    if np.allclose(normal, [0, 1, 1]):
        u = np.array([1.0, 0.0, 0.0])
    else:
        u = np.cross(normal, [0, 1, 0])
        u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)

    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    circle = origin + radius * (np.outer(np.cos(theta), u) + np.outer(np.sin(theta), v))
    return circle.astype(np.float32)


def get_beak_landmarks(
    start_radius: float,
    end_radius: float,
    length: float,
    twist: float,
    curve_x: float,
    curve_y: float,
    num_discs: int = 50,
    num_points: int = 30,
    reorient_base: bool = False,
) -> np.ndarray:
    """
    Generate a 3D curved horn-like structure as a (num_discs, num_points, 3) array.

    Parameters:
        start_radius: radius at the base of the horn
        end_radius: radius at the tip of the horn
        length: total length of the horn along the curved path
        twist: number of radians to twist along the horn
        curve_x: amplitude of curvature in X
        curve_y: amplitude of curvature in Y
        num_discs: number of cross-sections (rings)
        num_points: number of points per disc
        reorient_base: if True, rotate shape so base is horizontal and centered

    Returns:
        np.ndarray of shape (num_discs, num_points, 3)
    """
    # expand twist
    twist = np.pi * twist

    # fill arrays
    path = _generate_horn_path(length, twist, curve_x, curve_y, num_discs)
    radii = np.linspace(start_radius, end_radius, num_discs)
    cloud = np.empty((num_discs, num_points, 3), dtype=np.float32)

    # get disk rings at each point, treat end disks slightly different
    for i in range(num_discs):
        origin = path[i]
        if i == 0:
            normal = path[1] - path[0]
        elif i == num_discs - 1:
            normal = path[-1] - path[-2]
        else:
            normal = path[i + 1] - path[i - 1]
        normal /= np.linalg.norm(normal)
        disc = _generate_disc_points(origin, normal, radii[i], num_points)
        cloud[i] = disc

    # ensure the first ring is flat on z plane
    if reorient_base:
        first_normal = path[1] - path[0]
        first_normal /= np.linalg.norm(first_normal)
        R = _rotation_matrix_from_vectors(first_normal, np.array([0.0, 0.0, 1.0]))
        cloud = cloud @ R.T

    # Move first ring center to origin
    center0 = cloud[0].mean(axis=0)
    cloud -= center0

    return cloud


def landmarks_to_mesh(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a (num_discs, num_points, 3) point cloud into a triangle mesh.

    Returns:
        vertices: (N, 3)
        faces: (M, 3)
    """
    num_discs, num_points, _ = points.shape
    vertices = points.reshape(-1, 3)
    faces = []

    for i in range(num_discs - 1):
        for j in range(num_points):
            a = i * num_points + j
            b = i * num_points + (j + 1) % num_points
            c = (i + 1) * num_points + j
            d = (i + 1) * num_points + (j + 1) % num_points
            faces.append((a, b, d))
            faces.append((a, d, c))

    return vertices, np.array(faces, dtype=np.int32)


if __name__ == "__main__":
    # Generate the landmark cloud
    cloud = get_beak_landmarks(
        start_radius=1.0,
        end_radius=0.3,
        length=6.0,
        twist=3,
        curve_x=0.0,
        curve_y=1.0,
        num_discs=50,
        num_points=40,
        reorient_base=True,
    )
    print(cloud)
