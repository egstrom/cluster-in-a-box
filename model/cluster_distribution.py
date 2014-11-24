import numpy
import numpy.random
from math import *
import pylab
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from matplotlib.patches import Ellipse
from astropy.io import fits
import pdb
from scipy.special import lambertw
import sys


################################################################################
#
# Various functions used
#
class Mod_MyFunctions:

	def __init__(self):
		pass

	def imf(self, x):

		# Chabrier (2003) IMF for young clusters plus disk stars: lognorm and power-law tail
		ml = numpy.asarray((x <= 1.0).nonzero())[0]
		mh = numpy.asarray((x > 1.0).nonzero())[0]
		y = numpy.zeros(len(x))
		for i in ml: y[i] = 0.158/x[i]/log(10.) * exp(-(log10(x[i]) - log10(0.079))**2/2./0.69**2)
		for i in mh: y[i] = 4.4e-2/log(10.) * x[i]**(-2.3)
		return y


	def mass_dist(self, pdf, 
		mmin = 0.01, 
		mmax = 100.0, 
		ymin = 0, 
		ymax = 1, 
		N = 3000):

		result = []
		while len(result) < N:
			x = numpy.random.uniform(mmin, mmax, size=1000)
			y = numpy.random.uniform(ymin, ymax, size=1000)
			result.extend(x[numpy.where(y < pdf(x))])
		self.md = numpy.array(result[:N])
		return self.md

	def age_dist(self, age=1.0):

		# Half-lives from Dunham et al. (2014) and age distribution from C2D (Evans et al. 2009) and Sadavoy et al. 2014
		# Note: Not correct half-lives: would need to be recalculated under the assumption of consecutive decay which is why
		# the calculated age distribution does not match the input criteria

		### Relative population fractions observed in Perseus where t = 1 Myr
		age_frac = numpy.asarray([0.068, 0.211, 0.102, 0.545, 0.075])
		age_0 = 1.0

		lmbda = numpy.zeros(5)
		for i in range(0,len(age_frac)): lmbda[i] = log(age_frac[i])/(-age_0)

		self.frac = numpy.zeros(5)
		self.frac[0] = exp(-lmbda[0]*age)
		self.frac[1] = lmbda[0]/(lmbda[1]-lmbda[0]) * (exp(-lmbda[0]*age) - exp(-lmbda[1]*age))
		self.frac[2] = lmbda[0]*lmbda[1] * (exp(-lmbda[0]*age)/(lmbda[1]-lmbda[0])/(lmbda[2]-lmbda[0]) + 
			exp(-lmbda[1]*age)/(lmbda[0]-lmbda[1])/(lmbda[2]-lmbda[1]) + 
			exp(-lmbda[2]*age)/(lmbda[0]-lmbda[2])/(lmbda[1]-lmbda[2]))
		self.frac[3] = lmbda[0]*lmbda[1]*lmbda[2] * (exp(-lmbda[0]*age)/(lmbda[1]-lmbda[0])/(lmbda[2]-lmbda[0])/(lmbda[3]-lmbda[0]) +
			exp(-lmbda[1]*age)/(lmbda[0]-lmbda[1])/(lmbda[2]-lmbda[1])/(lmbda[3]-lmbda[1]) + 
			exp(-lmbda[2]*age)/(lmbda[0]-lmbda[2])/(lmbda[1]-lmbda[2])/(lmbda[3]-lmbda[2]) + 
			exp(-lmbda[3]*age)/(lmbda[0]-lmbda[3])/(lmbda[1]-lmbda[3])/(lmbda[2]-lmbda[3]))
		# self.frac[4] = lmbda[0]*lmbda[1]*lmbda[2]*lmbda[3] * (exp(-lmbda[0]*age)/(lmbda[1]-lmbda[0])/(lmbda[2]-lmbda[0])/(lmbda[3]-lmbda[0])/(lmbda[4]-lmbda[0]) +
		#	exp(-lmbda[1]*age)/(lmbda[0]-lmbda[1])/(lmbda[2]-lmbda[1])/(lmbda[3]-lmbda[1])/(lmbda[4]-lmbda[1]) + 
		#	exp(-lmbda[2]*age)/(lmbda[0]-lmbda[2])/(lmbda[1]-lmbda[2])/(lmbda[3]-lmbda[2])/(lmbda[4]-lmbda[2]) + 
		#	exp(-lmbda[3]*age)/(lmbda[0]-lmbda[3])/(lmbda[1]-lmbda[3])/(lmbda[2]-lmbda[3])/(lmbda[4]-lmbda[3]) + 
		#	exp(-lmbda[4]*age)/(lmbda[0]-lmbda[4])/(lmbda[1]-lmbda[4])/(lmbda[2]-lmbda[4])/(lmbda[3]-lmbda[4]))

		# Assume that everything ends up as Class III; no main sequence. An ok assumption when the interest is on Class 0/I sources
		self.frac[4] = 1. - sum(self.frac[:4])

		return self.frac


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
# Constants in CGS units
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
# Module for mass, radial distributions
#
class Mod_MassRad:
    
	############################################################################
	# Random mass plus radial distributions
	############################################################################
    def __init__(self):
    	pass

    def mass_radius(self,
    	N = 3000, 
    	N0 = 300, 
    	r0 = 1.0, 
    	alpha = 0.33, 
    	p = 1.0,
    	dv = 2.0,
		age = 1.0):

		# Protostellar masses
		# 1. Define IMF

		self.myf=Mod_MyFunctions()
		age_temp = self.myf.age_dist(age=age)
		print age_temp

		m_temp = self.myf.mass_dist(self.myf.imf, mmin = 0.01, mmax = 100., ymin = 0., ymax = 50., N = N)

		# m_temp = 1./numpy.random.lognormal(mean=mean, sigma=sigma, size=(N))
		print ' '
		print 'min(M), max(M) = %4.2f, %4.2f Msun' %(min(m_temp), max(m_temp))

		# 2. Sort out BD, LM and HM stars; for this project, ignore BD and HM
		hm = numpy.asarray((m_temp > 10.).nonzero())[0]
		lm = numpy.asarray(((m_temp <= 10.) & (m_temp > 0.05)).nonzero())[0]
		bd = numpy.asarray((m_temp <= 0.05).nonzero())[0]

		print ' '
		print 'Number of HM cores: %3i' %(numpy.size(hm))
		print 'Number of LM cores: %3i' %(numpy.size(lm))
		print 'Number of BD cores: %3i' %(numpy.size(bd))

		# 3. If Class 0 source, assume envelope mass is 3 times higher; for Class I's, M_env is twice higher

		nC0 = numpy.round(numpy.size(lm)*age_temp[0])
		nCI = numpy.round(numpy.size(lm)*age_temp[1])

		print ' '
		print 'Number of LM Class 0 sources: %3i' %(nC0)
		print 'Number of LM Class I sources: %3i' %(nCI)

		self.m = m_temp 

		lm0 = lm[:nC0]
		lmi = lm[nC0:nC0+nCI]
		lmii = lm[nC0+nCI:]

		self.m[lm0] = 3. * m_temp[lm0]
		self.m[lmi] = 1.5 * m_temp[lmi]

		print 'Total cluster mass: %6.2f' %(sum(self.m))

		self.mass_flag = numpy.zeros(N)
		self.mass_flag[hm] = 2
		self.mass_flag[lm0] = 10
		self.mass_flag[lmi] = 11
		self.mass_flag[lmii] = 12
		self.mass_flag[bd] = 0

		# Spatial distribution	
		r = r0 * (N/N0)**alpha
		rad = r*numpy.random.power(2.-p, size=(N))
		rad_m = rad*(self.m/min(self.m))**(-0.1)
		phi = numpy.random.rand(N)*2*pi

		self.x = numpy.zeros(N)
		self.y = numpy.zeros(N)

		for i in range(0,N): 
			self.x[i] = rad_m[i]*cos(phi[i])
			self.y[i] = rad_m[i]*sin(phi[i])

		# Outflow inclination, PA and protostellar velocity dispersion
		self.i = numpy.random.rand(N)*90.
		self.pa = numpy.random.rand(N)*180.
		self.vel = numpy.random.normal(dv, size=N)

		f=open('distribution.dat','w')
		f.write('x(pc)     y(pc)      M(Msun)    i(deg)     PA(deg)    vel(km/s)  Mass flag\n')
		for i in range(0,N): f.write('%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10i\n' %(self.x[i],self.y[i],self.m[i],self.i[i],self.pa[i],self.vel[i],self.mass_flag[i]))
		f.close()

		print 'Rmax = %4.2f pc' %r


################################################################################
#
# Main Module for distribution
#
class Mod_distribution:

	############################################################################
	# define configuration etc
	############################################################################
	def __init__(self):

		########################################################################
		########################################################################
		###                                                                  ###
		###       EDIT THIS PART ONLY                                        ###
		###                                                                  ###
		########################################################################
		########################################################################


    	### Cluster parameters
		self.N = 3000   # number of sources
		self.dist = 3000.   # Distance in pc

		### Initial mass function (currently only Chabrier 2003 IMF available)
		self.imf_type = 'standard'

		### Radial distribution of stars, from Adams et al. (2014)
		self.r0 = 1.   # initial radius (pc)
		self.N0 = 300.   # initial number of stars
		self.alpha = 1./3.   # power-law index for maximum cluster radius
		self.p = 1.0   # power-law index for radius PDF

		self.dv = 2.0   # internal velocity dispersion (only relevant if creating spectral cubes; not implemented at the moment)

		self.age = 0.1   # Cluster age

		self.massrad=Mod_MassRad()
		self.myplot=Mod_MyPlot()


	############################################################################
	# begin calculation
	############################################################################
	def calc(self):

		self.massrad.mass_radius(
			N = self.N, 
			N0 = self.N0, 
			r0 = self.r0, 
			alpha = self.alpha, 
			p = self.p,
			dv = self.dv,
			age = self.age)


		mass = self.massrad.m
		temp = numpy.zeros(self.N)
		temp2 = numpy.zeros(20)
		for i in range(0,self.N): temp[i] = log10(mass[i])

		### Plotting cluster characteristics
		n, bins, patches = plt.hist(temp,20,range=[-2,2], histtype='step')
		for i in range(0,numpy.size(n)): 
			if n[i] > 0: 
				temp2[i] = log10(n[i])
		plt.clf()
		fig = plt.figure(figsize=(8, 8))
		for i in range(0,self.N): plt.plot(self.massrad.x[i],self.massrad.y[i],'ob', ms=3*mass[i]**0.5, alpha=0.5)
		plt.minorticks_on()
		plt.xlabel('x (pc)')
		plt.ylabel('y (pc)')

		self.myplot.set_defaults()
		self.myplot.set_defaults()

		a = plt.axes([0.20, 0.19, 0.2, 0.2])
		plt.plot(bins[1:]-0.1,temp2, drawstyle='steps-mid')
		plt.xlabel(r'log($M$) ($M_\odot$)')
		plt.ylabel(r'log(d$N$/d$M$)')
		plt.minorticks_on()
		a.set_xticks([-2,-1,0,1,2])
		a.set_yticks([0,1,2,3])

		plt.savefig('cluster_template.pdf')
		plt.show()

################################################################################
#
# Main
#
if __name__ == "__main__":
    distribution=Mod_distribution()
    distribution.calc()


