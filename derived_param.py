"""
derived_param.py

Adding new derived parameters to MCMC chain

"""
import os, sys
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import math, copy

import getdist as gd
import getdist.plots as gdplots
from getdist.mcsamples import MCSamples

#from anesthetic import NestedSamples
from scipy.stats import chi2
from scipy.special import erfinv, erfcinv
from scipy.interpolate import interp1d

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
from matplotlib import rc
import colorsys
rc('text',usetex=True)
from matplotlib.ticker import FormatStrFormatter

# Importing Dr. Muir's helper functions
import getdist_chainplot_utils as gdu

# TO DO, add TATT params
# tex labels
DEFAULT_PLABELS = {\
                   'cosmological_parameters--omega_m':r'$\Omega_{\rm m}$',\
                   'cosmological_parameters--w':r'$w_0$',\
                   'cosmological_parameters--wa':r'$w_a$',\
                   'cosmological_parameters--wp':r'$w_{\rm p}$',\
                   'a_pivot':r'$a_{\rm p}$',\
                   'z_pivot':r'$z_{\rm p}$',\
                   'cosmological_parameters--omega_k':r'$\Omega_k$',\
                   'cosmological_parameters--omega_c':r'$\Omega_{\rm c}$',\
                   'COSMOLOGICAL_PARAMETERS--SIGMA_8':r'$\sigma_8$',\
                   'cosmological_parameters--sigma_8':r'$\sigma_8$',\
                   'S8':r'$S_8$',\
                   'cosmological_parameters--s_8':r'$S_8$',\
                   'cosmological_parameters--h0':r'$h$',\
                   'cosmological_parameters--omega_b':r'$\Omega_b$',\
                   'cosmological_parameters--n_s':r'$n_s$',\
                   'cosmological_parameters--a_s':r'$A_s$',\
                   'cosmological_parameters--a_s_1e9':r'$A_s*10^9$',\
                   'cosmological_parameters--omnuh2':r'$\Omega_{\nu}h^2$',\
                   'cosmological_parameters--ommh2':r'$\Omega_{\rm m}h^2$',\
                   'cosmological_parameters--ombh2':r'$\Omega_{\rm b}h^2$',\
                   'cosmological_parameters--omch2':r'$\Omega_{\rm c}h^2$',\
                   'cosmological_parameters--alens':r'$A_{\rm L}$',\
                   'cosmological_parameters--tau':r'$\tau$',\
                   'summnu':r'$\sum m_{\nu}\,[{\rm eV}]$',\
                   'cosmological_parameters--mnu':r'$\sum m_{\nu}\,[{\rm eV}]$',\
                   'supernova_params--m':r'$M_{\rm SN}$',\
                   'bin_bias--b1':r'$b^{[1]}$',\
                   'bin_bias--b2':r'$b^{[2]}$',\
                   'bin_bias--b3':r'$b^{[3]}$',\
                   'bin_bias--b4':r'$b^{[4]}$',\
                   'bin_bias--b5':r'$b^{[5]}$',\
                   'bin_bias--b6':r'$b^{[6]}$',\
                   'bias_lens--b1':r'$b^{[1]}$',\
                   'bias_lens--b2':r'$b^{[2]}$',\
                   'bias_lens--b3':r'$b^{[3]}$',\
                   'bias_lens--b4':r'$b^{[4]}$',\
                   'bias_lens--b5':r'$b^{[5]}$',\
                   'bias_lens--b6':r'$b^{[6]}$',\
                   'bias_lens--b1wt_bin1':r'$b^{[1]}}$',\
                   'bias_lens--b1wt_bin2':r'$b^{[2]}$',\
                   'bias_lens--b1wt_bin3':r'$b^{[3]}$',\
                   'bias_lens--b1wt_bin4':r'$b^{[4]}$',\
                   'bias_lens--b1wt_bin5':r'$b^{[5]}$',\
                   'bias_lens--b1wt_bin6':r'$b^{[6]}$',\
                   'bias_lens--b1e_bin1':r'$b_1^{[1]}$',\
                   'bias_lens--b1e_bin2':r'$b_1^{[2]}$',\
                   'bias_lens--b1e_bin3':r'$b_1^{[3]}$',\
                   'bias_lens--b1e_bin4':r'$b_1^{[4]}$',\
                   'bias_lens--b1e_bin5':r'$b_1^{[5]}$',\
                   'bias_lens--b1e_bin6':r'$b_1^{[6]}$',\
                   'bias_lens--b2e_bin1':r'$b_2^{[1]}$',\
                   'bias_lens--b2e_bin2':r'$b_2^{[2]}$',\
                   'bias_lens--b2e_bin3':r'$b_2^{[3]}$',\
                   'bias_lens--b2e_bin4':r'$b_2^{[4]}$',\
                   'bias_lens--b2e_bin5':r'$b_2^{[5]}$',\
                   'bias_lens--b2e_bin6':r'$b_2^{[6]}$',\
                   'bias_lens--b1e_sig8_bin1':r'$b_1^{[1]}\sigma_8$',\
                   'bias_lens--b1e_sig8_bin2':r'$b_1^{[2]}\sigma_8$',\
                   'bias_lens--b1e_sig8_bin3':r'$b_1^{[3]}\sigma_8$',\
                   'bias_lens--b1e_sig8_bin4':r'$b_1^{[4]}\sigma_8$',\
                   'bias_lens--b1e_sig8_bin5':r'$b_1^{[5]}\sigma_8$',\
                   'bias_lens--b1e_sig8_bin6':r'$b_1^{[6]}\sigma_8$',\
                   'bias_lens--b2e_sig8sq_bin1':r'$b_2^{[1]}\sigma_8^2$',\
                   'bias_lens--b2e_sig8sq_bin2':r'$b_2^{[2]}\sigma_8^2$',\
                   'bias_lens--b2e_sig8sq_bin3':r'$b_2^{[3]}\sigma_8^2$',\
                   'bias_lens--b2e_sig8sq_bin4':r'$b_2^{[4]}\sigma_8^2$',\
                   'bias_lens--b2e_sig8sq_bin5':r'$b_2^{[5]}\sigma_8^2$',\
                   'bias_lens--b2e_sig8sq_bin6':r'$b_2^{[6]}\sigma_8^2$',\
                   'bias_lens--rmean_bin':r'$X_{\rm lens}$',\
                   'shear_calibration_parameters--m1':R'$m_1$',\
                   'shear_calibration_parameters--m2':r'$m_2$',\
                   'shear_calibration_parameters--m3':r'$m_3$',\
                   'shear_calibration_parameters--m4':r'$m_4$',\
                   'shear_calibration_parameters--m5':r'$m_5$',\
                   'intrinsic_alignment_parameters--a':R'$A_{\rm IA}$',\
                   'intrinsic_alignment_parameters--alpha':R'$\alpha_{\rm IA}$',\
                   'wl_photoz_errors--bias_1':r'$\Delta z_{s}^1$',\
                   'wl_photoz_errors--bias_2':r'$\Delta z_{s}^2$',\
                   'wl_photoz_errors--bias_3':r'$\Delta z_{s}^3$',\
                   'wl_photoz_errors--bias_4':r'$\Delta z_{s}^4$',\
                   'wl_photoz_errors--bias_5':r'$\Delta z_{s}^5$',\
                   'wl_photoz_errors--bias_6':r'$\Delta z_{s}^6$',\
                   'lens_photoz_errors--bias_1':r'$\Delta z_{l}^1$',\
                   'lens_photoz_errors--bias_2':r'$\Delta z_{l}^2$',\
                   'lens_photoz_errors--bias_3':r'$\Delta z_{l}^3$',\
                   'lens_photoz_errors--bias_4':r'$\Delta z_{l}^4$',\
                   'lens_photoz_errors--bias_5':r'$\Delta z_{l}^5$',\
                   'lens_photoz_errors--bias_6':r'$\Delta z_{l}^6$',\
                   'lens_photoz_errors--width_1':r'$W_{l}^1$',\
                   'lens_photoz_errors--width_2':r'$W_{l}^2$',\
                   'lens_photoz_errors--width_3':r'$W_{l}^3$',\
                   'lens_photoz_errors--width_4':r'$W_{l}^4$',\
                   'lens_photoz_errors--width_5':r'$W_{l}^5$',\
                   'lens_photoz_errors--width_6':r'$W_{l}^6$',\
                   'intrinsic_alignment_parameters--a':r'$a_{\rm TATT}$',\
                   'intrinsic_alignment_parameters--alpha':r'$\alpha_{\rm TATT}$',\
                   'intrinsic_alignment_parameters--z_piv':r'$z_{\rm piv}^{\rm TATT}$',\
                   'intrinsic_alignment_parameters--a1':r'$A_1^{\rm TATT}$',\
                   'intrinsic_alignment_parameters--a2':r'$A_2^{\rm TATT}$',\
                   'intrinsic_alignment_parameters--adel':r'$A_{\rm del}^{\rm TATT}$',\
                   'intrinsic_alignment_parameters--alpha1':r'$\alpha_1^{\rm TATT}$',\
                   'intrinsic_alignment_parameters--alpha2':r'$\alpha_2^{\rm TATT}$',\
                   'intrinsic_alignment_parameters--alphadel':r'$\alpha_{\rm del}^{\rm TATT}$',\
                   'intrinsic_alignment_parameters--bias_ta':r'$b_{\rm ta}$',\
                   'intrinsic_alignment_parameters--bias_tt':r'$b_{\rm tt}$',\
                   'shear_calibration_parameters--m1_uncorr':'$m^{1}_{uncorr.}$',
                   'shear_calibration_parameters--m2_uncorr':'$m^{2}_{uncorr.}$',
                   'shear_calibration_parameters--m3_uncorr':'$m^{3}_{uncorr.}$',
                   'shear_calibration_parameters--m4_uncorr':'$m^{4}_{uncorr.}$',
                   'source_photoz_u--u_0_uncorr':'$u^{1}_{s, uncorr.}$',
                   'source_photoz_u--u_1_uncorr':'$u^{2}_{s, uncorr.}$',
                   'source_photoz_u--u_2_uncorr':'$u^{3}_{s, uncorr.}$',
                   'source_photoz_u--u_3_uncorr':'$u^{4}_{s, uncorr.}$',
                   'source_photoz_u--u_4_uncorr':'$u^{5}_{s, uncorr.}$',
                   'source_photoz_u--u_5_uncorr':'$u^{6}_{s, uncorr.}$',
                   'source_photoz_u--u_6_uncorr':'$u^{7}_{s, uncorr.}$',
                   'source_photoz_u--u_7_uncorr':'$u^{8}_{s, uncorr.}$',
                   'source_photoz_u--u_8_uncorr':'$u^{9}_{s, uncorr.}$',
                   'lens_photoz_u--u_0_0':'$u^{0,0}_{l}$',
                   'lens_photoz_u--u_0_1':'$u^{0,1}_{l}$',
                   'lens_photoz_u--u_0_2':'$u^{0,2}_{l}$',
                   'lens_photoz_u--u_0_3':'$u^{0,3}_{l}$',
                   'lens_photoz_u--u_1_0':'$u^{1,0}_{l}$',
                   'lens_photoz_u--u_1_1':'$u^{1,1}_{l}$',
                   'lens_photoz_u--u_1_2':'$u^{1,2}_{l}$',
                   'lens_photoz_u--u_1_3':'$u^{1,3}_{l}$',
                   'lens_photoz_u--u_2_0':'$u^{2,0}_{l}$',
                   'lens_photoz_u--u_2_1':'$u^{2,1}_{l}$',
                   'lens_photoz_u--u_2_2':'$u^{2,2}_{l}$',
                   'lens_photoz_u--u_2_3':'$u^{2,3}_{l}$',
                   'lens_photoz_u--u_3_0':'$u^{3,0}_{l}$',
                   'lens_photoz_u--u_3_1':'$u^{3,1}_{l}$',
                   'lens_photoz_u--u_3_2':'$u^{3,2}_{l}$',
                   'lens_photoz_u--u_3_3':'$u^{3,3}_{l}$',
                   'lens_photoz_u--u_4_0':'$u^{4,0}_{l}$',
                   'lens_photoz_u--u_4_1':'$u^{4,1}_{l}$',
                   'lens_photoz_u--u_4_2':'$u^{4,2}_{l}$',
                   'lens_photoz_u--u_4_3':'$u^{4,3}_{l}$',
                   'lens_photoz_u--u_5_0':'$u^{5,0}_{l}$',
                   'lens_photoz_u--u_5_1':'$u^{5,1}_{l}$',
                   'lens_photoz_u--u_5_2':'$u^{5,2}_{l}$',
                   'lens_photoz_u--u_5_3':'$u^{5,3}_{l}$',
                   'SHEAR_CALIBRATION_PARAMETERS--M1':'$m^{1}$',
                   'SHEAR_CALIBRATION_PARAMETERS--M2':'$m^{2}$',
                   'SHEAR_CALIBRATION_PARAMETERS--M3':'$m^{3}$',
                   'SHEAR_CALIBRATION_PARAMETERS--M4':'$m^{4}$',
                   'SOURCE_PHOTOZ_U--U_0':'$u^{1}_{s}$',
                   'SOURCE_PHOTOZ_U--U_1':'$u^{2}_{s}$',
                   'SOURCE_PHOTOZ_U--U_2':'$u^{3}_{s}$',
                   'SOURCE_PHOTOZ_U--U_3':'$u^{4}_{s}$',
                   'SOURCE_PHOTOZ_U--U_4':'$u^{5}_{s}$',
                   'SOURCE_PHOTOZ_U--U_5':'$u^{6}_{s}$',
                   'SOURCE_PHOTOZ_U--U_6':'$u^{7}_{s}$',
                   'SOURCE_PHOTOZ_U--U_7':'$u^{8}_{s}$',
                   'SOURCE_PHOTOZ_U--U_8':'$u^{9}_{s}$',
                   'source_photoz_u--u_0':'$u^{1}_{s}$',
                   'source_photoz_u--u_1':'$u^{2}_{s}$',
                   'source_photoz_u--u_2':'$u^{3}_{s}$',
                   'source_photoz_u--u_3':'$u^{4}_{s}$',
                   'source_photoz_u--u_4':'$u^{5}_{s}$',
                   'source_photoz_u--u_5':'$u^{6}_{s}$',
                   'source_photoz_u--u_6':'$u^{7}_{s}$',
                   'source_photoz_u--u_7':'$u^{8}_{s}$',
                   'source_photoz_u--u_8':'$u^{9}_{s}$',
                   'mag_alpha_lens--alpha_1':r'$\alpha_1$',\
                   'mag_alpha_lens--alpha_2':r'$\alpha_2$',\
                   'mag_alpha_lens--alpha_3':r'$\alpha_3$',\
                   'mag_alpha_lens--alpha_4':r'$\alpha_4$',\
                   'mag_alpha_lens--alpha_5':r'$\alpha_5$',\
                   'mag_alpha_lens--alpha_6':r'$\alpha_6$',\

                   'delta_neff':r'$\Delta N_{\rm eff}$',\
                   'ranks--rank_hyperparm_1':r'$\mathcal{H}_1$',\
                   'ranks--rank_hyperparm_2':r'$\mathcal{H}_2$',\
                   'ranks--rank_hyperparm_3':r'$\mathcal{H}_3$',\
                   'RANKS--REALISATION_ID':"\mathrm{ HR~rlzn \#}",\
                   'ranks--realisation_id':"\mathrm{ HR~rlzn \#}",\
                   'cosmological_parameters--tau':r'$\tau$',\
                   'planck--a_planck':r'$A_{\rm planck}$',\
                   'cosmological_parameters--yhe':r'$Y_{\rm He}$',
                   'xlens--xlens_all':r'$X_{\rm lens}$',
}

def prep_chain2(chainfname,chainlabel,num,kdesmooth=.5, paramlabels = DEFAULT_PLABELS, rangedict=None, chaindir=''):
    """
    Read in chain, add some derived parameters that we're likely to want
    May want to add/remove things here
    Note that these add_x() functions will just do nothing if necessary
        sampled parameters aren't in the chain

    If rangedict is None, will find hard prior boundaries in chain header
    and automatically pass them to getdist (recommended)
    """
    if rangedict is None:
         rangedict = gdu.get_ranges_from_chainheader(chainfname,chaindir)
    gdchain = gdu.get_gdchain(chainfname,flabel=chainlabel,\
                          kdesmooth=kdesmooth ,indatdir = chaindir,\
                          paramlabels = paramlabels, rangedict = rangedict)
    print('>>>',chainfname,gdchain)
    if gdchain is not None:
        add_S8_mod(gdchain,num)
        gdu.add_S8(gdchain)
        gdu.add_As_scaled(gdchain)
        gdu.add_physical_densities(gdchain)
        gdu.add_wp(gdchain) #if we have w0 and wa, will compute wp
        gdu.add_omega_c(gdchain)
        gdu.add_mnu(gdchain)
        gdu.add_bias_marg_params(gdchain) 

    #print(chainfname)
    #print([n.name for n in gdchain.getParamNames().names])
    return gdchain

def add_S8_mod(gdchains,num=None):
    """
    Takes in a single MCSample object or a list of them, goes through and
    adds a modified derived parameter S8. 

    Asks for what power to add in 
    """
    if type(gdchains)!=list:
        gdchainlist = [gdchains]
    else:
        gdchainlist = gdchains
    for ch in gdchainlist:
        if ch is None:
            continue
        #don't do anything if derived param is already there
        S8ind_mod = ch.paramNames.numberOfName('S8mod')
        if S8ind_mod!=-1:
            continue
        newkey = 'S8mod'
        newlabel = "S_{8mod}"
        omind = ch.paramNames.numberOfName('cosmological_parameters--omega_m')
        sig8ind = ch.paramNames.numberOfName('COSMOLOGICAL_PARAMETERS--SIGMA_8')
        if sig8ind==-1:# wasn't found
            # try lowercase version
            sig8ind = ch.paramNames.numberOfName('COSMOLOGICAL_PARAMETERS--SIGMA_8'.lower())
        om = ch.samples[:,omind]
        sig8 = ch.samples[:,sig8ind]
        #num = float(input("Enter power "))
        if not np.all(np.isnan(sig8)):
            if num == None:
                S8mod = sig8*((om/0.3)**(0.5))
            else:
                S8mod = sig8*((om/0.3)**(num))
            S8modrange = [None,None] #no input range for sigma8
            ch.addDerived(S8mod,name=newkey,label = newlabel+str(num),range=S8modrange)

def weighted_std(x,weights=None):
    """## Weighted standard deviation:

    * given arr x and weights w obtain weighted standaed deviation.
    
    Parameters:
    x: Array of the posterior distribution
    weights: weights associated with the values in x
    
    Returns:
    float: weighted standard deviation
    """
    avg = np.average(x , weights=weights)
    var = np.average((x-avg)**2, weights=weights)
    return np.sqrt(var)

def LogLike(d,mu,sig):
    ll = -0.5 * (((d - mu)/sig)**2 + np.log(2*np.pi)+np.log(sig**2))
    #ll = -0.5 * (((d - mu)/sig)**2)
    return ll

def importance_sampling(samples,param,mu,sig):
    """## Importance Sampling:

    * Given samples obtain LogLikes(gaussian) and reweight the to the currently stored likelihoods,
    and re-weighting accordingly, e.g. for adding a new data constraint. 
    * Note that LogLikes in reweightAddingLogLikes(LogLikes) is an array of -log(likelihood) for each sample to adjust.

    
    Parameters:

    samples: MCsamples
    param: array_like
    mu: mean of param
    sig: standard deviation of 

    Returns:
    out : ndarray
    
    """

    ## IDEA:
    #importance_sampling(sample,mu=None,sig=NOne):
    #w=sample.weights
    #S8m=sample.samples[:,sample.paramNames.numberOfName('S8mod')]
    #if mu=None:
    #mu = np.average(S8m, weights=w)
    #if ig=NOne:
    #sig =weighted_std(S8m,w)


    new_samples = MCSamples.copy(samples)
    LogLikes = -LogLike(param,mu,sig)
    new_samples.reweightAddingLogLikes(LogLikes)
    return new_samples

def ESS(samples):
    """## Effective Sample Size(ESS):
    Given Samples obtain the ESS
    
    Parameters:

    samples: MCsamples

    Returns:
    out: float
    
    """
    w = samples.weights
    return (np.sum(w))**2 / np.sum(w**2)