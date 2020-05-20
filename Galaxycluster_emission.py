#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pylab
import matplotlib.pyplot as plt
import matplotlib
import astropy
from astropy.io import fits
from astropy.convolution import convolve, Gaussian2DKernel
from scipy import stats
import sys
import csv
import os

################################################################################
#
# Constants in CGS units -- will add unit conversion later
#
class CgsConst:
    
    def __init__(self):
        self.c = 2.99792458e10    # cm s**-1
        self.k = 1.3806504e-16    # erg K**-1
        self.h = 6.62606896e-27   # erg s**-1
        self.AMU = 1.660538e-24   # g
        self.sigma = 5.670400e-5  # erg cm**-2 s**-1 K**-4
        self.hck = self.h*self.c/self.k
        self.yr = 365.2422*24.*3600.
        self.pcau = 206265.

const=CgsConst()


################################################################################
#
# Observational fits
#
class ObsFit:
	
	def __init__(self):
		pass

	def lin_test(self, x, y):
		df = len(x) - 2

		b = np.sum((x-np.mean(x))*(y-np.mean(y))) / np.sum((x-np.mean(x))**2)
		a = np.mean(y)-b*np.mean(x)
		e = y - (a+b*x)
		se_b = (np.sum(e**2) / np.sum((x - np.mean(x))**2))**0.5
		t_b = b / se_b
		p_b = stats.t.sf(np.abs(t_b), df)*2
		# print t_b, stats.t.ppf(0.687, 8)*se_b

		se_a = (np.sum(e**2)/df*(1./df+np.mean(x)**2/np.sum((x-np.mean(x))**2)))**0.5
		t_a = a / se_a
		p_a = stats.t.sf(np.abs(t_a), df)*2
		return np.asarray([a, b, p_a, p_b])


	def correlation_test(self, x, y, tol):
		# Find scaling / correlation between Menv and intensity
		# Return fit parameters and flag indicating which fit is best

		# Always first try linear fit with y = a + b*x:
		fit_m = ofit.lin_test(x, y)
		if (fit_m[3] < tol) & (fit_m[2] > tol):
			return [0, fit_m[1]], 'lin'
		elif (fit_m[3] < tol) & (fit_m[2] < tol):
			return [fit_m[0], fit_m[1]], 'lin'
		# If a linear fit is not good enough, try power-law:
		else:
			fit_m = ofit.lin_test(np.log10(x), np.log10(y))
			if (fit_m[3] < tol) & (fit_m[2] > tol):
				return [0, fit_m[1]], 'pow'
			elif (fit_m[3] < tol) & (fit_m[2] < tol):
				return [fit_m[0], fit_m[1]], 'pow'
			else:
				sys.exit()



ofit = ObsFit()


################################################################################
#
# Basic 1D template model to prove concept (to implement: spectral cubes)
#
class Mod_Template:
    
	############################################################################
	# Random mass plus radial distributions
	############################################################################
    def main(self):

        config={}
        f=open('image_setup_change.dat','r')
        for line in f.readlines():
            config[line.split()[0]]=line.split()[1]
    
        # Parameters relating to new image
        dist = float(config['bob']) # distance to cluster in pc
        pixel_size = float(config['psize']) # pixel size in arcsec
        resolution = float(config['beam']) # resolution of new image
        dim_pix = int(config['dim']) # image size in pixels
    
        tol = float(config['tol'])   # fit tolerance: if probability greater, then hypothesis is rejected at 1 sigma

        classI_scale = 0.1	
    
        ### Parameters relating to template observations
        # NOTE: JCMT observations still on T_A^* scale and intensity integration does not include dv (0.43 km/s)
        # the factor "fudge" takes care of that below
        beam_width = 15. # beam size in arcsec
        fudge = 0.43
        jy_k = 15.625   # K to Jy conversion specific to the JCMT (http://docs.jach.hawaii.edu/JCMT/HET/GUIDE/het_guide/); applies to T_A^*
        eta_a = 0.53    # again, specific to the JCMT
    
        dist0 = 200.   # Reference distance (normalization)
        obs = np.genfromtxt(config['obs'])
        model = np.genfromtxt(config['dist'], skip_header=1)
    
        menv = obs[:,3]
        # lbol = obs[:,5]   # not worried about Lbol for the moment, can come later
    
        area_beam = obs[:,5] * obs[:,4]**2 / (np.pi * (beam_width/2.)**2)
        # i_dist = obs[:,8]*(obs[:,2]/dist0)**2 * fudge * jy_k/eta_a / area_beam
        i_dist = obs[:,8]*(obs[:,2]/dist)**2 * fudge * jy_k/eta_a / area_beam
    
        fit, flag = ofit.correlation_test(menv, i_dist, tol)
    
        # r0 = np.mean((((obs[:,2]*obs[:,4])**2*obs[:,5])/np.pi)**0.5)/dist # average radius of emitting region in arcsec
        r0 = np.mean((((obs[:,5]*(obs[:,4])**2)**0.5/np.pi)**2 - (beam_width/2.)**2)**0.5*obs[:,2])/dist
        sep = np.mean(obs[:,2]*obs[:,7])/dist # average separation between outflow lobe and protostar in arcsec
        npix = (r0 / pixel_size)**2  # number of pixels per lobe
        npix_beam = 2.*np.pi*(resolution/2./(2.*np.log(2.))**0.5)**2 / pixel_size**2   # number of pixels per beam
    
        # im = np.zeros([dim_pix, dim_pix])
        im=[]
        # Isolate Class 0 and I sources from the model
        # cl0 = np.asarray(((model[:,6] == 10)).nonzero())[0]
        cl0 = np.asarray(((model[:,6] == 10) | (model[:,6] == 2)).nonzero())[0]
        cl1 = np.asarray((model[:,6] == 11).nonzero())[0]
 
        for i in cl0:

            cl0int=fit[0] + fit[1]*model[i,2]
            im.append(cl0int)
        
        for i in cl1:

            cl1int=classI_scale * (fit[0] + fit[1]*model[i,2])
            im.append(cl1int)
        
        im=[np.sum(im)]
        mass=[sum(model[:,2])/0.03]
        N=[len(model[:,2])]

        print('Total emission from cluster is '+str(im))
        print('Total mass in cluster is '+str(mass))

        config={}
        for line in open("cluster_setup_change.dat","r").readlines():
            config[line.split()[0]]=float(line.split()[1])

        imf_type = config['imf']
        tffscale = config['tff']
        SFE = config['SFE']

        filename='enter_file_name_here.csv'

        if os.path.exists(filename):
            append_write = 'a' # append if already exist
        else:
            append_write = 'w' # make a new file if not

        with open(filename, append_write) as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(zip(im,mass,N))


################################################################################
#
# Main
#
if __name__ == "__main__":
    template=Mod_Template()
    template.main()


