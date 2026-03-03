#!/usr/bin/env python

"""Methods to simulate horn shapes

1. Sample an ancestral shape parameter set.
2. Simulate Brownian states along the tree nodes.
3. Generate landmarks for each node.
4. Write to TSV file.
"""

import toytree
import pandas as pd
import numpy as np
from landmarks import get_beak_landmarks
from visualize import plot_pointcloud_3d, plot_pointcloud_grid


LABELS = ["start_radius", "end_radius", "length", "twist", "curve_x", "curve_y"]


def simulate_phylo_beaks(
    tree: toytree.ToyTree, 
    seed: np.random.Generator,
    num_discs: int = 30,
    num_points: int = 20,
    reorient_base: bool = True,
):
    """
    
    """
    # seed random generator
    rng = np.random.default_rng(seed)

    # sample the root state of sim params
    means = [3, 0.3, 15, 0, 0, 0]
    stdevs = [0.3, 0.1, 3, 1, 1, 1]
    root_states = rng.normal(loc=means, scale=stdevs)

    # simulate Brownian evolution from root
    data = tree.pcm.simulate_continuous_brownian(
        rates=stdevs,
        root_states=root_states,
        seed=rng,
        tips_only=False,
    )
    data.columns = LABELS
    return data


if __name__ == "__main__":

    from pathlib import Path
    import matplotlib.pyplot as plt

    SEED = 123
    RNG = np.random.default_rng(SEED)

    for i in range(10):
        DATADIR = Path(f"../data/phylo_{i}")
        DATADIR.mkdir(exist_ok=True)

        seed = RNG.integers(0, 1e9)

        # most trait SD=1 so tree-height=2 expects ~2 SD of change from 
        # root to tip.
        tree = toytree.rtree.unittree(ntips=10, treeheight=2, seed=seed)

        # set internal node names
        tree.set_node_data("nidx", {i: f"f{i.idx}" for i in tree}, inplace=True)

        # write seed and tree to file
        with open(DATADIR / "setup.txt", "w") as out:
            out.write(f"SEED={seed}\nTREE={tree.write(internal_labels='nidx')}")

        # simulate parameters
        data = simulate_phylo_beaks(tree, seed=seed)
        data.to_csv(DATADIR / "traits.tsv", sep="\t")
        print(data)

        # get landmarks
        clouds = []
        for idx in data.index:
            cloud = get_beak_landmarks(**data.loc[idx].to_dict(), num_discs=30, num_points=20, reorient_base=True)
            clouds.append(cloud.copy())

            cloud = cloud.reshape(-1, 3)
            cloud = pd.DataFrame(cloud, columns=list("xyz"))
            cloud.to_csv(DATADIR / f"landmarks_{idx}.tsv", sep="\t")

        # save drawing
        fig = plot_pointcloud_grid(clouds, ncols=5, uniform_axes=True)
        fig.savefig(DATADIR / f"visualized.pdf", bbox_inches="tight")

        # debug drawing
        # plt.show()



