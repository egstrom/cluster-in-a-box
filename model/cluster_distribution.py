import numpy
import numpy.random
from math import *
import pylab
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import sys
from sympy.solvers import nsolve
from sympy import Symbol, exp


################################################################################
#
# Various functions used
#
class Mod_MyFunctions:

	def __init__(self):
		pass

	def imf(self, x, imf_type):

	# Chabrier (2003) IMF for young clusters plus disk stars: lognorm and power-law tail
		mnorm = 1.0
		A1 = 0.158
		mc = 0.079
		sigma = 0.69
		A2 = 4.43e-2
		x0 = -1.3

		if imf_type == 0:
			a1 = A1 * np.exp(-((x - np.log10(mc))**2)/2.0/sigma**2)
			a2 = A2 * (10.0**x)**(x0-1)
			return np.where(x <= np.log10(mnorm), a1, a2)

		if imf_type == 1:
			a1 = A1 * np.exp(-((x - np.log10(mc))**2)/2.0/sigma**2)
			a2 = A2 * (10.0**x)**(x0-0)
			return np.where(x <= np.log10(mnorm), a1, a2)

		if imf_type == 2:
			a1 = A1 * np.exp(-((x - np.log10(mc))**2)/2.0/sigma**2)
			a2 = A2 * (10.0**x)**(x0-2)
			return np.where(x <= np.log10(mnorm), a1, a2)


	def mass_dist(self,
		mmin = 0.01, 
		mmax = 100.0, 
		Mcm = 10000, 
		imf_type = 0,
		SFE = 0.03):
	
		mmin_log = np.log10(mmin)
        	mmax_log = np.log10(mmax)

        	chunksize = 1000
        	result = np.array([], dtype=np.float64)
        	while result.sum() <= SFE * Mcm:
           	 	x = np.random.uniform(mmin_log, mmax_log, size=chunksize)
            		y = np.random.uniform(0, 1, size=chunksize)
            		result = np.hstack((result, 10 ** x[y < myf.imf(x, imf_type)]))
        	return result[result.cumsum() <= SFE * Mcm]

	def age_dist(self, m_temp, r, tffscale = 1.0):
		density=3*sum(m_temp)/(4*np.pi*r**3)
		G=4.43*10**(-3) ##pc^3 Msol^-1 Myr^-2
		tff=0.5427*1/np.sqrt(G*density) #free fall time for cluster

		age=tffscale*tff*np.random.uniform(0.0,1.0,size=1) #calculating current age from free fall time tff

		lmbdaSF=1/age

		lmbda=numpy.zeros(5)

		### Relative population fractions observed in clouds where t = 1.2 Myr 

		lmbda[0]=14.7
		lmbda[1]=7.9
		lmbda[2]=8.0
		lmbda[3]=0.347

		#Calculate new fractional populations, setting Class III equal to any leftovers

		self.frac = numpy.zeros(5)

		self.frac[0] = (lmbdaSF/lmbda[0])*(1-np.exp(-lmbda[0]*age))

		self.frac[1] = (lmbdaSF/lmbda[1])*(1-(lmbda[1]/(lmbda[1]-lmbda[0]))*(np.exp(-lmbda[0]*age))-(lmbda[0]/(lmbda[0]-lmbda[1]))*np.exp(-lmbda[1]*age))

		self.frac[2] = (lmbdaSF/lmbda[2])*(1-(lmbda[1]*lmbda[2])/((lmbda[2]-lmbda[0])*(lmbda[1]-lmbda[0]))*(np.exp(-lmbda[0]*age))-
		    (lmbda[0]*lmbda[2])/((lmbda[2]-lmbda[1])*(lmbda[0]-lmbda[1]))*(np.exp(-lmbda[1]*age))-
		    (lmbda[0]*lmbda[1])/((lmbda[0]-lmbda[2])*(lmbda[1]-lmbda[2]))*(np.exp(-lmbda[2]*age)))

		self.frac[3] = (lmbdaSF/lmbda[3])*(1-(lmbda[1]*lmbda[2]*lmbda[3])/((lmbda[1]-lmbda[0])*(lmbda[2]-lmbda[0])*(lmbda[3]-lmbda[0]))*(np.exp(-lmbda[0]*age)) -
		    (lmbda[0]*lmbda[2]*lmbda[3])/((lmbda[0]-lmbda[1])*(lmbda[2]-lmbda[1])*(lmbda[3]-lmbda[1]))*(np.exp(-lmbda[1]*age)) -
		    (lmbda[0]*lmbda[1]*lmbda[3])/((lmbda[0]-lmbda[2])*(lmbda[1]-lmbda[2])*(lmbda[3]-lmbda[2]))*(np.exp(-lmbda[2]*age)) -
		    (lmbda[0]*lmbda[1]*lmbda[2])/((lmbda[0]-lmbda[3])*(lmbda[1]-lmbda[3])*(lmbda[2]-lmbda[3]))*(np.exp(-lmbda[3]*age)))

		# Assume that everything ends up as Class III; no main sequence. An ok assumption when the interest is on Class 0/I sources

		self.frac[4] = 1. - sum(self.frac[:4])

		return self.frac

myf=Mod_MyFunctions()


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
        Mcm=10000,
        N0=300,
        r0=1.0,
        alpha=0.33,
        p=1.0,
        dv=2.0,
        tffscale=1.0,
        SFE=0.03,
        imf_type=0):

        f=open('output_distribution','w')

        m_temp = myf.mass_dist(mmin = 0.01, mmax = 100., Mcm = Mcm, imf_type = imf_type, SFE=SFE)
        N=len(m_temp)

        f.write('min(M), max(M) = %4.2f, %4.2f Msun\n' %(min(m_temp), max(m_temp)))

        self.m = m_temp
        self.N = N

        f.write('Total cluster mass: %6.2f \n' %(sum(self.m)))

        # Spatial distribution  
        r = r0 * (N/N0)**alpha
        rad = r*numpy.random.power(2.-p, size=(N))
        rad_m = rad*(self.m/min(self.m))**(-0.15)
        phi = numpy.random.rand(N)*2*pi

        self.x = numpy.zeros(N)
        self.y = numpy.zeros(N)

        f.write('Rmax = %4.2f pc \n' %r)

        # 2. Calculate age distribution

        age_temp= myf.age_dist(m_temp, r, tffscale = tffscale)

        f.write('Age distribution (Class 0, I, Flat, II, III): %4.2f, %4.2f, %4.2f, %4.2f, %4.2f \n' %(age_temp[0], age_temp[1], age_temp[2], age_temp[3], age_temp[4])) 


        # 3. Sort out BD, LM and HM stars; for this project, ignore BD and HM
        hm = numpy.asarray((m_temp > 10.).nonzero())[0]
        lm = numpy.asarray(((m_temp <= 10.) & (m_temp > 0.05)).nonzero())[0]
        bd = numpy.asarray((m_temp <= 0.05).nonzero())[0]

        f.write(' \n')
        f.write('Number of HM cores: %3i \n' %(numpy.size(hm)))
        f.write('Number of LM cores: %3i \n' %(numpy.size(lm)))
        f.write('Number of BD cores: %3i \n' %(numpy.size(bd)))


        nC0 = int(numpy.round(numpy.size(lm)*age_temp[0]))
        nCI = int(numpy.round(numpy.size(lm)*age_temp[1]))

        f.write(' \n')
        f.write('Number of LM Class 0 sources: %3i \n' %(nC0))
        f.write('Number of LM Class I sources: %3i \n' %(nCI))

        lm0 = lm[:nC0]
        lmi = lm[nC0:nC0+nCI]
        lmii = lm[nC0+nCI:]

        # 4. If Class 0 source, assume envelope mass is 3 times higher; for Class I's, M_env is 1.5 times highe

        self.m[lm0] = 3. * m_temp[lm0]
        self.m[lmi] = 1.5 * m_temp[lmi]

        self.mass_flag = numpy.zeros(N)
        self.mass_flag[hm] = 2
        self.mass_flag[lm0] = 10
        self.mass_flag[lmi] = 11
        self.mass_flag[lmii] = 12
        self.mass_flag[bd] = 0

        f.close()

        for i in range(0,N): 
            self.x[i] = rad_m[i]*cos(phi[i])
            self.y[i] = rad_m[i]*sin(phi[i])

        # Outflow inclination, PA and protostellar velocity dispersion
        self.i = numpy.random.rand(N)*90.
        self.pa = numpy.random.rand(N)*180.
        self.vel = numpy.random.normal(dv, size=N)

        #print ('Spatial distribution calculated')

        f=open('distribution.dat','w')
        
        f.write('x(pc)     y(pc)      M(Msun)    i(deg)     PA(deg)    vel(km/s)  Mass flag\n')
        
        for i in range(0,N): f.write('%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10i\n' %(self.x[i],self.y[i],self.m[i],self.i[i],self.pa[i],self.vel[i],self.mass_flag[i]))
        
        f.close()


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

		config={}
		for line in open("cluster_setup_change.dat","r").readlines():
		    config[line.split()[0]]=float(line.split()[1])
		    # print "%20s=%.5e" % (line.split()[0],config[line.split()[0]])

		### Cluster parameters
		self.Mcm = int(config['Mcm'])   # Molecular Cloud Mass

		### Initial mass function (currently only Chabrier 2003 IMF available)
		self.imf_type = config['imf']

		### Free fall time scaling and star formation efficiency
		self.tffscale = config['tff']
		self.SFE = config['SFE']

		### Radial distribution of stars, from Adams et al. (2014)
		self.r0 = config['r0']   # initial radius (pc)
		self.N0 = int(config['N0'])   # initial number of stars
		self.alpha = config['alpha']   # power-law index for maximum cluster radius
		self.p = config['p']   # power-law index for radius PDF

		self.dv = config['dv']   # internal velocity dispersion (only relevant if creating spectral cubes; not implemented at the moment)

		self.massrad=Mod_MassRad()
		self.myplot=Mod_MyPlot()


	############################################################################
	# begin calculation
	############################################################################
	def calc(self):

		self.massrad.mass_radius(
		    Mcm = self.Mcm, 
		    N0 = self.N0, 
		    r0 = self.r0, 
		    alpha = self.alpha, 
		    p = self.p,
		    dv = self.dv,
		    tffscale = self.tffscale,
		    SFE = self.SFE,
		    imf_type = self.imf_type)


		mass = self.massrad.m
		N = self.massrad.N
		temp = numpy.zeros(self.N)
		temp2 = numpy.zeros(20)
		for i in range(0,self.N): temp[i] = log10(mass[i])

		### Plotting cluster characteristics
		n, bins, patches = plt.hist(temp,20,range=[-2,2], histtype='step')
		for i in range(0,numpy.size(n)): 
			if n[i] > 0: 
				temp2[i] = log10(n[i])

		fig = plt.figure(figsize=(8, 8))
		for i in range(0,self.N): plt.plot(self.massrad.x[i],self.massrad.y[i],'ob', ms=3*mass[i]**0.5, alpha=0.5)
		plt.minorticks_on()
		plt.xlabel('x (pc)')
		plt.ylabel('y (pc)')

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


