"""
Visualization utilities for 3D landmark clouds and meshes.
Includes functions to render point clouds, export meshes, and preview geometries.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from landmarks import get_beak_landmarks, landmarks_to_mesh


def set_3d_axes_equal(ax, x: np.ndarray, y: np.ndarray, z: np.ndarray, scale: bool = False):
    """
    Set 3D axes to have equal aspect ratio.
    If scale is True, forces a cubic bounding box (equal units).
    """
    if scale:
        ax.set_box_aspect([1, 1, 1])
    else:
        ax.set_box_aspect([x.ptp(), y.ptp(), z.ptp()])


def plot_pointcloud_3d(
    cloud: np.ndarray,
    color1: str = "royalblue",
    color2: str = "orangered",
    alpha: float = 0.8,
    figsize=(8, 6),
    scale: bool = False,
):
    """
    Plot a 3D point cloud with diverging colors per ring.
    First and last quarters of each ring get color1, middle gets color2.
    """
    num_discs, num_points, _ = cloud.shape
    q = num_points // 2
    color_array = np.zeros((num_discs, num_points), dtype=object)
    color_array[:] = color2
    color_array[:, :q] = color1
    color_array = color_array.flatten()

    # color_array = np.concatenate(color_array)
    flat = cloud.reshape(-1, 3)
    x, y, z = flat[:, 0], flat[:, 1], flat[:, 2]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=color_array, s=6, alpha=alpha)

    set_3d_axes_equal(ax, x, y, z, scale)
    ax.set_title("3D Point Cloud")
    plt.tight_layout()
    return fig, ax


def plot_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    color: str = "lightcoral",
    alpha: float = 0.8,
    figsize=(8, 6),
    scale: bool = False,
):
    """
    Plot a 3D triangle mesh using matplotlib.
    """
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    mesh = Poly3DCollection(vertices[faces], alpha=alpha)
    mesh.set_facecolor(color)
    mesh.set_edgecolor("k")
    ax.add_collection3d(mesh)

    ax.set_xlim([x.min(), x.max()])
    ax.set_ylim([y.min(), y.max()])
    ax.set_zlim([z.min(), z.max()])

    set_3d_axes_equal(ax, x, y, z, scale)
    ax.set_title("3D Mesh")
    plt.tight_layout()
    return fig, ax


if __name__ == "__main__":
    # Generate landmark cloud

    kwargs = dict(
        start_radius=3.0,
        end_radius=0.3,
        length=10.0,
        twist=1. * np.pi,
        curve_x=3,
        curve_y=5,
    )

    cloud = get_beak_landmarks(
        num_discs=50,
        num_points=40,
        reorient_base=True,
        **kwargs,
    )

    # Plot point cloud
    plot_pointcloud_3d(cloud)

    # Convert to mesh and plot
    # verts, faces = landmarks_to_mesh(cloud)
    # plot_mesh(verts, faces)
    plt.show()
