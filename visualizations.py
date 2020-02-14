import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx


def plot(locations, energies, fig, title):


    interesting_indexes = np.where((np.abs(energies) > 0.000))
    locations = locations[interesting_indexes].astype(np.float)
    energies = energies[interesting_indexes].astype(np.float)
    ax = Axes3D(fig)
    ax.set_title(title)
    cmap = cmx.hsv

    x = locations[:, 0]
    y = locations[:, 1]
    z = locations[:, 2]
    e = energies*0.10

    ax.scatter(x,y,z,s=e,cmap=cmap)
    #
    # ax.set_xbound(-5, +5)
    # ax.set_ybound(-5, +5)
    # ax.set_zbound(-5, +5)

def visualize(item):
    calojet, pfjet = item
    calo_locations = calojet[:, 0:3]
    pf_locations = pfjet[:, 0:3]


    calo_energy = calojet[:, 3]
    pf_energy = pfjet[:, 3]

    f = plt.figure()
    plot(calo_locations, calo_energy, f, 'CaloJet')
    f = plt.figure()
    plot(pf_locations, pf_energy, f, 'PFJet')
    plt.show()



