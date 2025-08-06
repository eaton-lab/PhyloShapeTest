"""
Visualization utilities for 3D landmark clouds and meshes.
Includes functions to render point clouds, export meshes, and preview geometries.
"""

from typing import Tuple
import math
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
    ax=None,
    scale: bool = False,
    title: str = "3D Point Cloud"
):
    """
    Plot a 3D point cloud with diverging colors per ring.
    If ax is None, creates a new figure. Otherwise, uses the given ax.
    """
    if cloud.ndim > 3:
        raise ValueError("cloud must be ndim=2 or ndim=3")

    # disc-colored cloud
    elif cloud.ndim == 3:
        num_discs, num_points, _ = cloud.shape
        q = num_points // 2
        color_array = np.full((num_discs, num_points), color2, dtype=object)
        color_array[:, :q] = color1
        color_array = color_array.flatten()
        flat = cloud.reshape(-1, 3)
        x, y, z = flat[:, 0], flat[:, 1], flat[:, 2]

    # single array
    elif cloud.ndim == 2:
        x, y, z = cloud.T
        color_array = np.full(cloud.shape[0], color1, dtype=object)

    # draw
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = None

    ax.scatter(x, y, z, c=color_array, s=6, alpha=alpha)
    set_3d_axes_equal(ax, x, y, z, scale)
    ax.set_title(title)

    return fig, ax


def plot_pointcloud_grid(
    clouds: list,
    ncols: int = 4,
    figsize_per_plot: Tuple[int, int] = (4, 4),
    color1: str = "royalblue",
    color2: str = "orangered",
    alpha: float = 0.8,
    scale: bool = False,
):
    """
    Plot a grid of 3D point clouds.

    Parameters:
        clouds: list of np.ndarray (each shape [N,3] or [D,N,3])
        ncols: number of columns in the grid
        figsize_per_plot: (width, height) for each subplot
        color1/color2: colors for diverging scheme
        alpha: transparency
        scale: whether to set axes to equal scale
    """
    n = len(clouds)
    nrows = math.ceil(n / ncols)
    fig = plt.figure(figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows))

    for i, cloud in enumerate(clouds):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection='3d')
        plot_pointcloud_3d(
            cloud,
            color1=color1,
            color2=color2,
            alpha=alpha,
            ax=ax,
            scale=scale,
            title=f"Shape {i + 1}",
        )

    plt.tight_layout()
    return fig


def plot_pointcloud_grid(
    clouds: list,
    ncols: int = 4,
    figsize_per_plot: Tuple[int, int] = (4, 4),
    color1: str = "royalblue",
    color2: str = "orangered",
    alpha: float = 0.8,
    scale: bool = False,
    uniform_axes: bool = False,
):
    """
    Plot a grid of 3D point clouds.

    Parameters:
        clouds: list of np.ndarray (each shape [N,3] or [D,N,3])
        ncols: number of columns in the grid
        figsize_per_plot: (width, height) for each subplot
        color1/color2: colors for diverging scheme
        alpha: transparency
        scale: if True, apply set_3d_axes_equal (unless overridden by uniform_axes)
        uniform_axes: if True, apply same x/y/z limits to all subplots
    """
    n = len(clouds)
    nrows = math.ceil(n / ncols)
    fig = plt.figure(figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows))

    # If uniform_axes is True, compute global min/max
    if uniform_axes:
        all_points = []
        for cloud in clouds:
            flat = cloud.reshape(-1, 3) if cloud.ndim == 3 else cloud
            all_points.append(flat)
        all_points = np.vstack(all_points)
        xlim = (np.min(all_points[:, 0]), np.max(all_points[:, 0]))
        ylim = (np.min(all_points[:, 1]), np.max(all_points[:, 1]))
        zlim = (np.min(all_points[:, 2]), np.max(all_points[:, 2]))
    else:
        xlim = ylim = zlim = None

    for i, cloud in enumerate(clouds):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection='3d')
        plot_pointcloud_3d(
            cloud,
            color1=color1,
            color2=color2,
            alpha=alpha,
            ax=ax,
            scale=(scale and not uniform_axes),
            title=f"Shape {i + 1}",
        )

        if uniform_axes:
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_zlim(*zlim)

    plt.tight_layout(h_pad=3.0, w_pad=3.0)
    return fig





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

    # integrifolia
    kwargs = dict(
        start_radius=3.0,
        end_radius=0.3,
        length=20.0,
        twist=-3. * np.pi,
        curve_x=0,
        curve_y=3,
    )

    # fetisowii
    kwargs = dict(
        start_radius=3.0,
        end_radius=0.3,
        length=20.0,
        twist=-0. * np.pi,
        curve_x=-3,
        curve_y=3,
    )

    # draw
    cloud = get_beak_landmarks(
        num_discs=30,
        num_points=20,
        reorient_base=True,
        **kwargs,
    )

    # Plot point cloud
    plot_pointcloud_3d(cloud)

    # clouds = [cloud, cloud, cloud, cloud, cloud]
    # fig = plot_pointcloud_grid(clouds, ncols=3)
    # fig.savefig("/tmp/pointcloud_grid.pdf", bbox_inches="tight")

    # Convert to mesh and plot
    # verts, faces = landmarks_to_mesh(cloud)
    # plot_mesh(verts, faces)
    plt.show()
