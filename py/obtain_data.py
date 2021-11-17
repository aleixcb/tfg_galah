# Authors: F. Anders, A. Cuevas, J. Dolcet (ICCUB 2021)

"""
GALAH EDR3 RC data for tSNE
"""

import numpy as np
from astropy.io import fits as pyfits   #acces als files .fits (Flexible Image Transport System)
from astropy.table import Table   #manipular taules
import scipy    #biblioteca per mates

def sqadd(a, b):   #sqrt(a^2 + b^2)
    "Add 2 values in quadrature"
    return np.sqrt(a*a + b*b)

class galah_dr3_rc(object):       #agrupem les funcions dins la classe galah_dr3_rc
    def __init__(self, abundances=True): 
        """
        Open the GALAH DR3 red clump catalogue (and applies some cuts on the 
        flags.
        
        Based on the flag statistics in the catalogue:
        * ../data/GALAH_DR3_joint_allstar_v2_redclump_flagstatistics.txt  (descarregat)
        we exclude the elements Li, C, Rb, Sr, Mo, Ru, Sm
        
        
        
        La funció obre el catàleg de GALAH DR3 RC i aplica condicions a les flags (que siguin bones mesures).
        Excloem els elements Li, C,Rb, Sr, Mo, Ru, Sm per ser males mesures
        
        """
        hdu = pyfits.open('../data/GALAH_DR3_joint_allstar_v2_redclump.fits', names=True) #HDU (Header Data Unit)
        data = hdu[1].data  #extensió 1
        if abundances:  # (=True)
            data = data[ (data['flag_sp']==0) * (data['flag_fe_h']==0) * 
                         (data['flag_o_fe']==0) * (data['flag_na_fe']==0) * 
                         (data['flag_mg_fe']==0) * (data['flag_al_fe']==0) * 
                         (data['flag_si_fe']==0) * (data['flag_k_fe']==0) * 
                         (data['flag_ca_fe']==0) * (data['flag_sc_fe']==0) * 
                         (data['flag_ti_fe']==0) * (data['flag_v_fe']==0) * 
                         (data['flag_cr_fe']==0) * (data['flag_mn_fe']==0) *
                         (data['flag_co_fe']==0) * (data['flag_ni_fe']==0) * 
                         (data['flag_cu_fe']==0) * (data['flag_zn_fe']==0) * 
                         (data['flag_y_fe']==0) * (data['flag_zr_fe']==0) *
                         (data['flag_ba_fe']==0) * (data['flag_la_fe']==0) *
                         (data['flag_ce_fe']==0) * (data['flag_nd_fe']==0) *
                         (data['flag_eu_fe']==0)
                         ] #(data['e_ba_fe]<1)
        self.data = data
        return None

    def get_ndimspace(self, feh=True, norm="stdev"):  #espai ndimensional d'abundàncies
        """
        We select the dimensions of the GALAH DR3 chemical abundance space:
        * Fe, O, Na, Mg, Al, Si, K, Ca, Sc, Ti, V, Cr, Mn, Co, Ni, Cu, Zn, Y, Zr, Ba, La, Ce, Nd, Eu
        (24 elements)
        
        """
        data = self.data
        
        #MATRIU ABUNDANCIES (dim = 37417 estrelles x 24 elements = files x columnes)
""" np.c_ concatena matrius en columns (cap a la dreta), així com np.r_ concatena matrius en files(cap avall) """
        X        = np.c_[data['fe_h'],data['o_fe'],data['na_fe'],data['mg_fe'],  
                         data['al_fe'],data['si_fe'],data['k_fe'],data['ca_fe'],
                         data['sc_fe'],data['ti_fe'],data['v_fe'],data['cr_fe'],
                         data['mn_fe'],data['co_fe'],data['ni_fe'],data['cu_fe'],
                         data['zn_fe'],data['y_fe'],data['zr_fe'],data['ba_fe'],
                         data['la_fe'],data['ce_fe'],data['nd_fe'],data['eu_fe'] ]
        
        
        #MATRIU ERRORS
        Xerr1    = np.c_[data['e_fe_h'],data['e_o_fe'],data['e_na_fe'],data['e_mg_fe'],
                         data['e_al_fe'],data['e_si_fe'],data['e_k_fe'],data['e_ca_fe'],
                         data['e_sc_fe'],data['e_ti_fe'],data['e_v_fe'],data['e_cr_fe'],
                         data['e_mn_fe'],data['e_co_fe'],data['e_ni_fe'],data['e_cu_fe'],
                         data['e_zn_fe'],data['e_y_fe'],data['e_zr_fe'],data['e_ba_fe'],
                         data['e_la_fe'],data['e_ce_fe'],data['e_nd_fe'],data['e_eu_fe'] ]

        # Take care of the 0.0 uncertainties: forced minimum to 0.03 
        Xerr1[:, :] = np.maximum(Xerr1, 0.03*np.ones(Xerr1.shape))
        """ compara la matriu d'errors amb una matriu plena de 0,03's element a element, per agafar el valor més
        gran dels dos (força un mínim error de 0,03) """
        Xerr     = np.mean( Xerr1, axis=0) #dim=1x24 # fa la mitjana dels errors de l'abundància d'un element per 
#                                                      totes les estrelles
        """ calcula la mitjana dels elements de la matriu seguint un dels eixos
        (axis=0 seguint la columna i axis=1 seguint la fila) """
        

        if not feh:
            X = X[:,1:]; Xerr = Xerr[1:]  #exclou [Fe/H] de les matrius (1a columna de X i 1r element de Xerr)

        if norm == "hogg2016":
            # Normalise everything by the typical range as in Hogg+2016: 
            Xnorm    = (X/Xerr[np.newaxis,:])
        elif norm == "stdev":
            # Normalise everything by the typical range defined by the std deviation: 
            Xnorm    = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        elif norm == None:
            # Do not normalise at all:
            Xnorm    = X
        else:
            raise ValueError("Please pass a valid 'norm' keyword.")
        self.X, self.Xerr, self.Xnorm = X, Xerr, Xnorm
        return

    def get_ndimspace_H(self, cn=True, age=False, norm="stdev"):
        """
        Cut out missing data and prepare t-SNE input array  (separem les dades que faltin)
        Optional:
            age: Bool  - include age in the analysis
            cn:  Bool  - include [C/H] & [N/H], default: True
            norm: str  - normalisation method, default: "stdev"
        """
        # For giants, everything up to Cu is okay (agafem fins al Cu)
        X        = self.data['X_H'][:, :20]
        Xerr     = np.mean(self.data['X_H_ERR'][:, :20], axis=0)
        if not cn:
            X = X[:,2:]; Xerr = Xerr[2:]  #excloem [C/H] & [N/H]
        if norm == "hogg2016":
            # Normalise everything by the typical range as in Hogg+2016: 
            Xnorm    = (X/Xerr[np.newaxis,:])
        elif norm == "stdev":
            # Normalise everything by the typical range defined by the std deviation: 
            Xnorm    = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        elif norm == None:
            # Do not normalise at all:
            self.Xnorm    = X
        else:
            raise ValueError("Please pass a valid 'norm' keyword.")
        self.X, self.Xerr, self.Xnorm = X, Xerr, Xnorm
        return

    def get_umap_tsne_colours(self, p=None, lr=None, nn=None, md=None, 
                              metric="euclidean", version=""):
        """
        Get the umap & t-SNE results for the optimal hyperparameters,
        and also the arrays used to colour the maps.
        """
        #Read umap/t-SNE results table
        results = Table.read("../data/galah_rc"+version+"_dimred_hyperparametertest.fits")
        # Get the relevant columns
        self.Xp = results["X_PCA"]
        self.Yp = results["Y_PCA"]
        self.Xt = results["X_tSNE_"+metric+"_p"+str(p) + "_lr"+str(lr)]
        self.Yt = results["Y_tSNE_"+metric+"_p"+str(p) + "_lr"+str(lr)]
        self.Xu = results["X_umap_"+metric+"_nn"+str(nn) + "_md"+str(md)]
        self.Yu = results["Y_umap_"+metric+"_nn"+str(nn) + "_md"+str(md)]
        # And now the colours
        data  = self.data
        self.colors   = [data['fe_h'], data['mg_fe'], 
                         data['mg_fe']-data['si_fe'], data['mg_fe']-(data['fe_h']+data['o_fe']),
                         data['al_fe']-data['mg_fe'], data['sc_fe'],
                         data['mn_fe']-data['cr_fe'],data['cu_fe']-data['ni_fe'],
                         data['zn_fe'], data['y_fe']-data['ba_fe'], 
                         data['zr_fe'], data['la_fe'], data['eu_fe'],
                         data['R_Rzphi'], data['z_Rzphi'], 
                         data['e_rv_galah'], 
                         np.log10(data['chi2_sp']), data['vR_Rzphi'],
                         data['vT_Rzphi'],data['vz_Rzphi']]
        self.titles   = [r'$\rm [Fe/H]$', r'$\rm [Mg/Fe]$', r'$\rm [Mg/Si]$', r'$\rm [Mg/O]$',
                      r'$\rm [Al/Mg]$', r'$\rm [Sc/Fe]$', r'$\rm [Mn/Cr]$', r'$\rm [Cu/Ni]$',
                      r'$\rm [Zn/Fe]$', r'$\rm [Y/Ba]$', r'$\rm [Zr/Fe]$', r'$\rm [La/Fe]$',
                      r'$\rm [Eu/Fe]$', r'$R$', r'$Z$', r'log $\sigma_{RV}$',#r'log $S/N$',
                      r'log $\chi^2$', r'$v_R$', r'$v_T$', r'$v_Z$']
        return

    def get_umap_subsets(self, nn=100, md=0.1, **kwargs):
        """
        Get the names and indices of the t-sne-defined subsets
        """
        # First get umap results:
        results = Table.read("../data/galah_rc_dimred_hyperparametertest.fits")
        self.Xu = results["X_umap_euclidean_nn"+str(nn) + "_md"+str(md)]
        self.Yu = results["Y_umap_euclidean_nn"+str(nn) + "_md"+str(md)]
        
        # Now run HDBSCAN to define the subsets
        import hdbscan
        clusterer = hdbscan.HDBSCAN(**kwargs)
        clusterer.fit( np.vstack((self.Xu, self.Yu)).T )
        self.classcol = clusterer.labels_
        self.classprob= clusterer.probabilities_
        self.subsets  = np.unique(clusterer.labels_)
        #self.classcol= np.char.rstrip(self.data["tsne_class_teffcut40"],b' ')#.decode('utf8').strip()
        #self.subsets = ["thin", "thick1", "thick2", "thick3", "thick4",
        #           "mpthin", "mpthintrans", "smr", "t4trans", "youngthin",
        #           "debris1", "debris2", "debris3", "debris4", "debris5", 
        #           "smr2", "t2trans1", "highTi","lowMg","highAlMg?"]
        self.names   = ["", "", "", "",
                   "", "", "Transition group", "", "",
                   "Young local disc", "", "", "[s/Fe]-enhanced", "", "", r"", "Debris candidate", 
                   r"Extreme-Ti star", r"Low-[Mg/Fe] star", "High-[Al/Mg] star"]
        self.Xcoords = [10, 11, 4.5, -12,  18, -31, 22, 26,-22.5, -14, -2, -25]
        self.Ycoords = [5.5,.5,  -2, -4,   6,  0,   1.5, -.5, -7, -2, -6, 14]
        self.fsize   = [20 , 16,  12, 12,  15,  13, 11, 11, 11, 11, 11, 11]
        self.sym = ["o", "v", "^", ">", "<", "s", "o", "*", "<", "o",
                    "h", "d", "D", "v", "p", "*", "D", "p", "s", "8"]
        self.al  = [.6, .8, .8, .8, .8, .8, .8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        self.lw  = [0,.5,.5,.5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5]
        self.size= [7,12,12,12,12,15,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18]
        self.col = ["grey", "m", "hotpink", "crimson", "r",
                    "g", "brown", "orange", "gold", "k",
                    "yellow", 
                    "gold", "lime", "k", "royalblue"]

