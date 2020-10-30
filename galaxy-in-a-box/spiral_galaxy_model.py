import numpy as np
import pylab
import matplotlib.pyplot as plt
import matplotlib
import math
import os
from os import listdir
from os.path import isfile, join
import sys	
from sympy.solvers import nsolve
from sympy import Symbol, exp
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
from matplotlib.colors import LogNorm
import copy
import csv
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve

###Creating Galaxy spatial and mass Distributions ###

def spatial_mass_dist(
	N=1e4,
	mmin=1e4,
	mmax=1e6):
	### Mass Distribution ###
	y=-1.64 #Mass spectra slope from Mark Krumholz 2019 original -1.64
	MassDist=[]
	massrange=np.arange(mmin,mmax,1)
	for m in massrange:
		MassDis=(m)**(y)
		MassDist.append(MassDis)
	MassDistribution=(MassDist/max(MassDist))*mmax
	mass=[]
	while len(mass) < (N):
		MassDist=np.random.choice(MassDistribution,size=1)
		if MassDist > mmin:
			mass.append(MassDist)
	new_MassDistribution = [[i[0]] for i in mass]

	### Initial Parameters for spatial Distribution###
	B=2.5 #controls arm sweep and bar size
	B1=3.2
	A=12 #controls the amount of winding
	A1=20.5
	mu=0 # centered on function spiral value


	Nd=int(N) #disk cluster number
	Nb=int(N*(1/6)) #boulge cluster number
	l=10 #number of stars per point value of function
	lamda=3.5
	beta=1.0/lamda
	maxrad=12 #maximum radius of galaxy in kpc

	###Creating exponential distribution###
	rexp=[]
	R=np.arange(0,10,0.0001)
	for i in R:
		lob=beta*np.exp(-(i*beta))
		rexp=np.append(rexp,lob)
	m=np.random.choice(rexp,int(Nd/(2))) #radially follows an exponential distribution randomly chosen
	### Generating spiral arm shape given parameters ###
	no=m*2.8*math.pi/(max(m)) #values of phi with designated highest phi value
	Rad=rexp*2.8*math.pi/(max(rexp))
	n=no[no<(1.85*math.pi)]

	X=[]
	Y=[]

	for phi in n:
		r=4/(math.log(B*math.tan((phi)/(2*A)))) #actual values for spiral arm randomly chosen
		sigma=1/(2+0.5*phi) #spread from spiral value following normal distribution
		rx=np.random.normal(mu,sigma,1)
		ry=np.random.normal(mu,sigma,1)
		x=r*math.cos(phi)+rx
		X.append(x)
		y=r*math.sin(phi)+ry
		Y.append(y)
	X.extend(np.negative(X)[::-1])
	Y.extend(np.negative(Y)[::-1])
	X=(np.array(np.array(X))).flatten()
	Y=(np.array(np.array(Y))).flatten()
	new_Y = [[i] for i in Y]
	X=X[::-1]
	new_X = [[i] for i in X]
	SpatialArray=np.append(new_X,new_Y,1)

	u=no[no>(1.85*math.pi)]
	X1=[]
	Y1=[]

	for phi in u:
		r=5/(math.log(B1*math.tan((phi)/(2*A1))))
		sigma=1/(2+0.5*phi) #spread from spiral value following normal distribution
		rx=np.random.normal(mu,sigma,1)
		ry=np.random.normal(mu,sigma,1)
		x=r*math.cos(phi)+rx
		X1.append(x)
		y=r*math.sin(phi)+ry
		Y1.append(y)

	X1.extend(np.negative(X1)[::-1])
	Y1.extend(np.negative(Y1)[::-1])
	X1=(np.array(np.array(X1))).flatten()
	Y1=(np.array(np.array(Y1))).flatten()
	new_Y1 = [[i] for i in Y1]
	X1=X1[::-1]
	new_X1 = [[i] for i in X1]
	SpatialArray1=np.append(new_X1,new_Y1,1)


	### Combining Spatial and Mass Distributions into one array ###

	SpatialArray=np.append(SpatialArray, SpatialArray1,0)
	SpatialMassArray=np.append(SpatialArray, new_MassDistribution,1)
	SpatialX = SpatialMassArray[:, 0]
	SpatialY = SpatialMassArray[:, 1]
	Mass = (SpatialMassArray[:, 2]).flatten()


	return Mass, SpatialX, SpatialY

mass, X, Y = spatial_mass_dist(N=1e4, mmin=1e4, mmax=1e6)


import Galaxy_clusters as cd
import Galaxycluster_emission as ce
Mcm=mass
tff=[1.0]
IMF=[0]
SFE=[0.03]
for n in tff:
	for j in IMF:
		for s in SFE:
			for i in Mcm:
				f = open("cluster_setup.dat")
				fout = open("cluster_setup_change.dat", "wt")
				for line in f: 
					fout.write(line.replace('10000', str(i)))
					for line in f:
						fout.write(line.replace('0.03', str(s)))
						for line in f:
							fout.write(line.replace('0.0',str(j)))
							for line in f:
								fout.write(line.replace('1.0',str(n)))
				f.close()
				fout.close()
				newname="distributions_Mcm="+str(i)+"_tff="+str(n)+"_imf="+str(j)+"_SFE="+str(s)+".dat"
				distribution=cd.Mod_distribution()
				distribution.calc()
				os.rename("distribution.dat",newname)
				g=open("image_setup.dat")
				gout = open("image_setup_change.dat", "wt")
				for line in g: 
					gout.write(line.replace('distribution.dat', newname))
				g.close()
				gout.close()
				template=ce.Mod_Template()
				template.main()
				try: 
					os.remove(newname)
				except OSError:
					pass


im = []
mass = []
N = []
#Give same file name as in Galaxy_emission code
with open('galaxycluster_emission.csv', 'r') as file:
    reader = csv.reader(file, delimiter='\t')
    for row in reader:
        im.append(float(row[0]))
        mass.append(float(row[1]))
        N.append(float(row[2]))

mass = [[i] for i in mass]
ims = [[i] for i in im]
N = [[i] for i in N]

comb=np.append(ims,mass,1)
comb=np.append(comb,N,1)

comb1=[]

while (len(comb1)+len(comb))<10000:
    comb1.extend(comb)

for i in comb:
    comb1.extend([i])
    if len(comb1)==10000:
        break

print(np.amax(X))

dims=(1401,1401) #total grid dimensions i.e. galaxy size in pixels each pixel is 17.4pc
Galaxyarray=np.zeros(dims)
for i in range(0,len(X)):
    R=(comb1[i][1]/(np.pi*144))**(1/2)

    if 2*R > 17.4:
        dim=int(2*R/17.4)
        d = comb1[i][0]/(dim**2)
        data = np.zeros((dim,dim))
        data.fill(d)
    else:
        d = comb1[i][0]
        data=np.zeros((3,3))
        data[1,1]=d
    
    x=X[i]*56
    y=Y[i]*56
    
    Galaxyarray[int((x+(dims[0]-len(data))/2)):int((x+(dims[0]+len(data))/2)),int((y+(dims[1]-len(data))/2)):int((y+(dims[1]+len(data))/2))]+=data
#print(np.amax(x))
config={}
f=open('image_setup_change.dat','r')
for line in f.readlines():
    config[line.split()[0]]=line.split()[1]
    
# Parameters relating to new image
dist = float(config['bob']) # distance to cluster in pc
pixel_size = float(config['psize']) # pixel size in arcsec
resolution = float(config['beam']) # resolution of new image
dim_pix = int(config['dim']) # image size in pixels
npix_beam = 2.*np.pi*(resolution/2./(2.*np.log(2.))**0.5)**2 / pixel_size**2   # number of pixels per beam

beam = Gaussian2DKernel(resolution/pixel_size/(2.*(2.*np.log(2.))**0.5))
im_obs = convolve(Galaxyarray, beam, boundary='extend')/npix_beam

im_obs[im_obs==0]=1e-100

#plt.figure()
fig, ax = plt.subplots()
my_cmap = copy.copy(matplotlib.cm.get_cmap('jet')) # copy the default cmap
my_cmap.set_bad([0,0,0])
plt.imshow(im_obs, interpolation='nearest', cmap=my_cmap, norm=LogNorm(vmin=1e-8, vmax=np.amax(im_obs)))
plt.colorbar()
plt.xlabel('kpc from center')
plt.ylabel('kpc from center')
plt.xticks(np.arange(0,1401,175))
plt.yticks(np.arange(0,1401,175))
labels=[-12,-9,-6,-3,0,3,6,9,12]
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
plt.savefig('Galaxy_template.pdf',format='pdf')



plt.show()
