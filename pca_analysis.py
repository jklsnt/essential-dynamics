import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import MDAnalysis as mda
from MDAnalysis.tests.datafiles import PSF, DCD, PDB
from MDAnalysis.analysis import pca, align

import warnings
# suppress some MDAnalysis warnings about writing PDB files
warnings.filterwarnings('ignore')


u = mda.Universe("./1mh1-processed.pdb", "./trajectory.dcd")
aligner = align.AlignTraj(u, u, select='backbone',
                          in_memory=True).run()

pc = pca.PCA(u, select='backbone',
             align=True, mean=None,
             n_components=None).run()

# check the percentage variance explained by the 0-th component
variance_percentage = (pc.variance[0]**2)/np.sum(pc.variance**2)

backbone = u.select_atoms('backbone')
n_bb = len(backbone)
print('There are {} backbone atoms in the analysis'.format(n_bb))
print(pc.p_components.shape)

# we will extract the first n principle components
transformed = pc.transform(backbone, n_components=10)
transformed.shape

# cosine content measurement
pca.cosine_content(transformed, 0)
pca.cosine_content(transformed, 1)
pca.cosine_content(transformed, 2)
pca.cosine_content(transformed, 3)
pca.cosine_content(transformed, 4)
pca.cosine_content(transformed, 9)

pc.mean


pc.p_components[:,0].shape
transformed.shape

# first principle component
pc1 = pc.p_components[:, 0]
trans1 = transformed[:, 0]
projected = np.outer(trans1, pc1) + pc.mean.flatten()
coordinates = projected.reshape(len(trans1), -1, 3)
coordinates.shape


#projection
proj1 = mda.Merge(backbone)
proj1.load_new(coordinates, order="fac")

proj1.select_atoms("protein").write("./trajectory-backbone.dcd", frames="all")
proj1.select_atoms("protein").write("./1mh1-backbone.pdb")


