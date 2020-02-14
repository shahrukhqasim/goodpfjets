import numpy as np
import h5py


class JetDataset:
    def __init__(self, filename):
        self.infile = h5py.File(filename, "r")
        self.pfjets_px = self.infile["pfjets_px"]
        self.pfjets_py = self.infile["pfjets_py"]
        self.pfjets_pz = self.infile["pfjets_pz"]
        self.pfjets_eta = self.infile["pfjets_eta"]
        self.pfjets_phi = self.infile["pfjets_phi"]
        self.pfjets_energy = self.infile["pfjets_energy"]
        self.pfjets_pt = self.infile["pfjets_pt"]

        self.calojets_px = self.infile["calojets_px"]
        self.calojets_py = self.infile["calojets_py"]
        self.calojets_pz = self.infile["calojets_pz"]
        self.calojets_eta = self.infile["calojets_eta"]
        self.calojets_phi = self.infile["calojets_phi"]
        self.calojets_energy = self.infile["calojets_energy"]
        self.calojets_pt = self.infile["calojets_pt"]


        self.tracks_px = self.infile["tracks_px"] # Max 100
        self.tracks_py = self.infile["tracks_py"]
        self.tracks_pz = self.infile["tracks_pz"]
        self.tracks_eta = self.infile["tracks_eta"]
        self.tracks_phi = self.infile["tracks_phi"]
        self.tracks_pt = self.infile["tracks_pt"]

        self.ebhit_energy = self.infile["ebhit_energy"] # Max 10000
        self.ebhit_eta = self.infile["ebhit_eta"]
        self.ebhit_phi = self.infile["ebhit_phi"]

        self.hcalhit_energy = self.infile["hcalhit_energy"] # Max 20000
        self.hcalhit_eta = self.infile["hcalhit_eta"]
        self.hcalhit_phi = self.infile["hcalhit_phi"]

    def len(self):
        return len(self.pfjets_px)


    def getitem(self, idx):

        thecalojet = np.concatenate((self.calojets_px[idx][..., np.newaxis],self. calojets_py[idx][..., np.newaxis], self.calojets_pz[idx][..., np.newaxis], self.calojets_energy[idx][..., np.newaxis]), axis=1)
        thepfjet = np.concatenate((self.pfjets_px[idx][..., np.newaxis],self. pfjets_py[idx][..., np.newaxis], self.pfjets_pz[idx][..., np.newaxis], self.pfjets_energy[idx][..., np.newaxis]), axis=1)

        return thecalojet, thepfjet

    def getitemetaphi(self, idx):

        thecalojet = np.concatenate((self.calojets_eta[idx][..., np.newaxis],self. calojets_phi[idx][..., np.newaxis], self.calojets_pz[idx][..., np.newaxis], self.calojets_energy[idx][..., np.newaxis]), axis=1)
        thepfjet = np.concatenate((self.pfjets_eta[idx][..., np.newaxis],self. pfjets_phi[idx][..., np.newaxis], self.pfjets_pz[idx][..., np.newaxis], self.pfjets_energy[idx][..., np.newaxis]), axis=1)

        return thecalojet, thepfjet


    def getitemex(self, idx):
        return self.pfjets_px[idx], self.pfjets_py[idx], self.pfjets_pz[idx], self.pfjets_eta[idx],\
               self.pfjets_phi[idx], self.pfjets_energy[idx], self.pfjets_pt[idx], self.calojets_px[idx],\
               self.calojets_py[idx], self.calojets_pz[idx], self.calojets_eta[idx], self.calojets_phi[idx],\
               self.calojets_energy[idx], self.calojets_pt[idx], self.tracks_px[idx], self.tracks_py[idx],\
               self.tracks_pz[idx], self.tracks_eta[idx], self.tracks_phi[idx], self.tracks_pt[idx], \
               self.ebhit_energy[idx], self.ebhit_eta[idx], self.ebhit_phi[idx], self.hcalhit_energy[idx], \
               self.hcalhit_eta[idx], self.hcalhit_phi[idx]

