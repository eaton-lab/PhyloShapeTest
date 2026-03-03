#!/usr/bin/env python

"""Phyloshape weighted vectors method.

1. Find conserved shapes among models as homologous point clouds 
that retain minimal global vector differences among models in the
data set.
2. Ensure a conserved face on each 'module' of the model, and ...
3. Select conserved faces from among the conserved shapes.
4. For each landmark point project into face-coordinate space.
5. 
6. 

A. Convert landmark vertices to a graph of vectors.
B. Convert vectors to 'face' coordinate space.
C. Compute weights for each vector...
D. Infer PCs from Vectors...
E. Reconstruct ancestral PCs under BM.
F. Convert PCs back into relative vectors.
G. Convert vectors back into absolute coordinates?


1. Load a TSV of landmarks from one or more models. The landmark
file must include ID, X, Y, and Z. Extra 'module' label column is
optional. 
```
ID       X      Y      Z    module
1      0.1    0.2    0.3    beak
2      0.1    0.3    0.4    beak
3      0.1    0.3    0.5    beak
...
21     0.8    0.1    0.5    tube
22     0.8    0.2    0.5    tube
23     0.8    0.3    0.5    tube
``


"""
