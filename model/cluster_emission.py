import numpy as np
import pylab
import matplotlib.pyplot as plt
import matplotlib
import astropy
from astropy.io import fits
from astropy.convolution import convolve, Gaussian2DKernel
from scipy import stats
import sys

################################################################################
#
# set plot parameters
#
class Mod_MyPlot:

	def __init__(self):
		pass

	def set_defaults(d):

		figsize=[12,15]
		normal = 24
		small = 12
		tiny = 8
		line = 1.5
	
		params={'axes.labelsize': normal,
		    'axes.linewidth': line,
			'lines.markeredgewidth': line,
			'font.size': normal,
			'legend.fontsize': normal,
			'xtick.labelsize': normal,
			'ytick.labelsize': normal,
			'xtick.major.size': small,
			'xtick.minor.size': tiny,
			'ytick.major.size': small,
			'ytick.minor.size': tiny,
			'savefig.dpi':300,
			'text.usetex': True,
			'figure.figsize': figsize,
		    	}

		plt.rc('font',**{'family':'serif','serif':['Times']})
		plt.rc('lines', lw=line)
		plt.rc('axes', linewidth=line)
		plt.tight_layout(pad=0.1)

		pylab.rcParams.update(params)

	def set_ticks(xmajor,ymajor,xminor,yminor):
		ax=pylab.gca()
		ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(xmajor))
		ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(xminor))
		ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(ymajor))
		ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(yminor))
		pylab.draw()

	def set_xticks(xmajor,xminor):
		ax=pylab.gca()
		ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(xmajor))
		ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(xminor))
		pylab.draw()

	def set_yticks(ymajor,yminor):
		ax=pylab.gca()
		ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(ymajor))
		ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(yminor))
		pylab.draw()


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
# Model addition
#
class Mod_outflow:
	
	def __init__(self):
		pass

	def twod_gaussian(self, x, y, x0, y0, sigma_x, sigma_y, amp):
		return amp * np.exp(-((x-x0)**2/2./sigma_x**2 + (y-y0)**2/2./sigma_y**2))

	def add_lobe(self, image, x0, y0, i_peak, r):
		dim = len(image)
		x, y = np.meshgrid(np.arange(dim)-dim/2, np.arange(dim)-dim/2)
		new_image = outflow.twod_gaussian(x, y, x0, y0, r/1.517, r/1.517, i_peak)
		return new_image

outflow = Mod_outflow()

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
			print "A proportionality I = %4.2f * Menv is acceptable at 1 sigma" %(fit_m[1])
			print "Fit probabilities are %6.4f and %6.4f respectively" %(fit_m[2], fit_m[3])
			return [0, fit_m[1]], 'lin'
		elif (fit_m[3] < tol) & (fit_m[2] < tol):
			print "A linear fit with I = %4.2f + %4.2f * Menv is acceptable at 1 sigma" %(fit_m[0], fit_m[1])
			print "Fit probabilities are %6.4f and %6.4f respectively" %(fit_m[2], fit_m[3])
			return [fit_m[0], fit_m[1]], 'lin'
		# If a linear fit is not good enough, try power-law:
		else:
			fit_m = ofit.lin_test(np.log10(x), np.log10(y))
			if (fit_m[3] < tol) & (fit_m[2] > tol):
				print "A power-law fit with log(I) = %4.2f * log(Menv) is acceptable at 1 sigma" %(fit_m[0], fit_m[1])
				print "Fit probabilities are %6.4f and %6.4f respectively" %(fit_m[2], fit_m[3])
				return [0, fit_m[1]], 'pow'
			elif (fit_m[3] < tol) & (fit_m[2] < tol):
				print "A power-law fit with log(I) = %4.2f + %4.2f * log(Menv) is acceptable at 1 sigma" %(fit_m[0], fit_m[1])
				print "Fit probabilities are %6.4f and %6.4f respectively" %(fit_m[2], fit_m[3])
				return [fit_m[0], fit_m[1]], 'pow'
			else:
				print 'I and Menv are not correlated, neither linearly nor by power-law'
				print 'Model is terminated, please try again'
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

		self.myplot=Mod_MyPlot()
	
		config={}
		for line in file("image_setup.dat","r").readlines():
			config[line.split()[0]]=line.split()[1]
	
		# Parameters relating to new image
		dist = float(config['D']) # distance to cluster in pc
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
		i_dist = obs[:,8]*(obs[:,2]/dist)**2 * fudge * jy_k/eta_a / area_beam
	
		fit, flag = ofit.correlation_test(menv, i_dist, tol)
	
		# r0 = np.mean((((obs[:,2]*obs[:,4])**2*obs[:,5])/np.pi)**0.5)/dist # average radius of emitting region in arcsec
		r0 = np.mean((((obs[:,5]*(obs[:,4])**2)**0.5/np.pi)**2 - (beam_width/2.)**2)**0.5*obs[:,2])/dist
		# print r0
		sep = np.mean(obs[:,2]*obs[:,7])/dist # average separation between outflow lobe and protostar in arcsec
		npix = (r0 / pixel_size)**2  # number of pixels per lobe
		npix_beam = 2.*np.pi*(resolution/2./(2.*np.log(2.))**0.5)**2 / pixel_size**2   # number of pixels per beam
	
		im = np.zeros([dim_pix, dim_pix])
		half_im = dim_pix / 2
	
		# Isolate Class 0 and I sources from the model
		# cl0 = np.asarray(((model[:,6] == 10)).nonzero())[0]
		cl0 = np.asarray(((model[:,6] == 10) | (model[:,6] == 2)).nonzero())[0]
		cl1 = np.asarray((model[:,6] == 11).nonzero())[0]

		for i in cl0:
			sepp = np.sin(model[i,3])*sep
			xob = (model[i,0]*const.pcau/dist + np.sin(model[i,4]*np.pi/180.)*sepp)/pixel_size#+half_im
			yob = (model[i,1]*const.pcau/dist + np.cos(model[i,4]*np.pi/180.)*sepp)/pixel_size#+half_im
			xor = (model[i,0]*const.pcau/dist - np.sin(model[i,4]*np.pi/180.)*sepp)/pixel_size#+half_im
			yor = (model[i,1]*const.pcau/dist - np.cos(model[i,4]*np.pi/180.)*sepp)/pixel_size#+half_im
	
			if (np.abs(xob) < half_im) & (np.abs(xor) < half_im) & (np.abs(yob) < half_im) & (np.abs(yor) < half_im):
				if flag == 'lin': 
					# i_peak = 2. * np.pi * (r0/pixel_size/1.517)**2 / (fit[0] + fit[1]*model[i,2])/npix*npix_beam
					i_peak =  (fit[0] + fit[1]*model[i,2])/npix*npix_beam / (2. * np.pi * (r0/pixel_size/1.517)**2)
				if flag == 'pow': 
					i_peak = 2. * np.pi * (r0/pixel_size/1.517)**2 / (10.**(fit[0] + fit[1]*np.log10(model[i,2])))/npix*npix_beam
				print i_peak, model[i,2]
				im = im + outflow.add_lobe(im, xob, yob, i_peak, r0/pixel_size)
				im = im + outflow.add_lobe(im, xor, yor, i_peak, r0/pixel_size)
	
			# im[xob-dx:xob+dx,yob-dx:yob+dx] = im[xob-dx:xob+dx,yob-dx:yob+dx]+iob /npix*npix_beam
			# im[xor-dx:xor+dx,yor-dx:yor+dx] = im[xor-dx:xor+dx,yor-dx:yor+dx]+ior /npix*npix_beam
		
		for i in cl1:
			sepp = np.sin(model[i,3])*sep
			xob = (model[i,0]*const.pcau/dist + np.sin(model[i,4]*np.pi/180.)*sepp)/pixel_size#+half_im)
			yob = (model[i,1]*const.pcau/dist + np.cos(model[i,4]*np.pi/180.)*sepp)/pixel_size#+half_im)
			xor = (model[i,0]*const.pcau/dist - np.sin(model[i,4]*np.pi/180.)*sepp)/pixel_size#+half_im)
			yor = (model[i,1]*const.pcau/dist - np.cos(model[i,4]*np.pi/180.)*sepp)/pixel_size#+half_im)
	
			if (np.abs(xob) < half_im) & (np.abs(xor) < half_im) & (np.abs(yob) < half_im) & (np.abs(yor) < half_im):
				if flag == 'lin': 
					i_peak = classI_scale * 2. * np.pi * (r0/pixel_size/1.517)**2 / (fit[0] + fit[1]*model[i,2])/npix*npix_beam
				if flag == 'pow': 
					i_peak = classI_scale * 2. * np.pi * (r0/pixel_size/1.517)**2 / (10.**(fit[0] + fit[1]*np.log10(model[i,2])))/npix*npix_beam
			
				im = im + outflow.add_lobe(im, xob, yob, i_peak, r0/pixel_size)
				im = im + outflow.add_lobe(im, xor, yor, i_peak, r0/pixel_size)
		
		beam = Gaussian2DKernel(resolution/pixel_size/(2.*(2.*np.log(2.))**0.5))
		im_obs = convolve(im, beam, boundary='extend')/npix_beam
		
		header = fits.Header()
		header['BMAJ'] = resolution / 3600.
		header['BMIN'] = resolution / 3600.
		header['BPA'] = 0.0
		header['BTYPE'] = 'Intensity'
		header['BUNIT'] = 'JY/BEAM '
		header['EQUINOX'] = 2.000000000000E+03
		header['CTYPE1'] = 'RA---SIN'
		header['CRVAL1'] = 0.0
		header['CDELT1'] =  pixel_size/3600.
		header['CRPIX1'] =  half_im
		header['CUNIT1'] = 'deg     '
		header['CTYPE2'] = 'DEC--SIN'
		header['CRVAL2'] = 0.0
		header['CDELT2'] =  pixel_size/3600.
		header['CRPIX2'] =  half_im
		header['CUNIT2'] = 'deg     '
		header['RESTFRQ'] =   3.384090000000E+11
		header['SPECSYS'] = 'LSRK    '
		hdu = fits.PrimaryHDU(im_obs, header=header)
		hdu.writeto('cluster_emission.fits', clobber = True)

		print "Peak intensity in image is %4.2f Jy km/s/beam" %(im_obs.max())
		
		rnge = pixel_size*dim_pix/2.   # plotting range of image
		plt.imshow(im_obs, vmin=0, vmax=im_obs.max(), aspect='equal', extent=(rnge,-rnge,-rnge,rnge), cmap='PuRd')
		
		plt.xlabel('x (arcsec)')
		plt.ylabel('y (arcsec)')
	
		plt.minorticks_on()
		
		cbar = plt.colorbar()
		cbar.set_label('Jy km s$^{-1}$ beam$^{-1}$')
	
		self.myplot.set_defaults()
		self.myplot.set_defaults()
	
		plt.savefig('cluster_emission.pdf')
		plt.show()
	

################################################################################
#
# Main
#
if __name__ == "__main__":
	template=Mod_Template()
	template.main()
