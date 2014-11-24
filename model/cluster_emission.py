import numpy
import pylab
import matplotlib
from astropy.convolution import convolve, Gaussian2DKernel

### Observational set up:
# d = 200 pc
# I / pixel = obs_scale * M_env
# average area = 25 pixels
# pixel size = 7.5"

clf()

dist = 3000 # distance to cluster in pc
aupc = 206265. # AU in a pc

beam_width = 15. # beam size in arcsec
obs_scale = 10.823

model = genfromtxt('distribution.dat', skip_header=1)
prop = genfromtxt('/Users/kristensen/work/observations/jcmt/lm_ch3oh_7-6/template_data/properties.dat')

cl0 = asarray((model[:,6] == 10).nonzero())[0]
cl1 = asarray((model[:,6] == 11).nonzero())[0]

r0 = mean((((prop[:,2]*prop[:,4])**2*prop[:,5]))**0.5)/aupc # average radius of emitting region, pc
sep = mean(prop[:,2]*prop[:,7])/aupc # average separation between outflow lobe and protostar

im = zeros([128,128])
half_im = 64
dpix = 0.05 # pixel scale in pc
dx = round(r0 / dpix / 2.)

for i in cl0:
	sepp = sin(model[i,3])*sep
	xob = round((model[i,0] + sin(model[i,4]*pi/180.)*sepp)/dpix+half_im)
	yob = round((model[i,1] + cos(model[i,4]*pi/180.)*sepp)/dpix+half_im)
	xor = round((model[i,0] - sin(model[i,4]*pi/180.)*sepp)/dpix+half_im)
	yor = round((model[i,1] - cos(model[i,4]*pi/180.)*sepp)/dpix+half_im)
	ior = obs_scale*model[i,2]
	iob = obs_scale*model[i,2]

	im[xob-dx:xob+dx,yob-dx:yob+dx] = im[xob-dx:xob+dx,yob-dx:yob+dx]+iob
	im[xor-dx:xor+dx,yor-dx:yor+dx] = im[xor-dx:xor+dx,yor-dx:yor+dx]+ior

for i in cl1:
	sepp = sin(model[i,3])*sep
	xob = round((model[i,0] + sin(model[i,4]*pi/180.)*sepp)/dpix+half_im)
	yob = round((model[i,1] + cos(model[i,4]*pi/180.)*sepp)/dpix+half_im)
	xor = round((model[i,0] - sin(model[i,4]*pi/180.)*sepp)/dpix+half_im)
	yor = round((model[i,1] - cos(model[i,4]*pi/180.)*sepp)/dpix+half_im)
	ior = obs_scale*model[i,2]
	iob = obs_scale*model[i,2]

	im[xob-dx:xob+dx,yob-dx:yob+dx] = im[xob-dx:xob+dx,yob-dx:yob+dx]+iob
	im[xor-dx:xor+dx,yor-dx:yor+dx] = im[xor-dx:xor+dx,yor-dx:yor+dx]+ior

beam_pix = (pi*beam_width**2)/(dpix*aupc/dist)**2

beam = Gaussian2DKernel(beam_width/(dpix*aupc/dist)/(2.*(2.*log(2.))**0.5))
im_obs = convolve(im, beam, boundary='extend')/beam_pix

print im_obs.max()

imshow(im_obs, vmin=0, vmax=1, aspect='equal', extent=(-2.56,2.56,-2.56,2.56), cmap='PuRd')
# for i in cl0: plot(model[i,0],model[i,1],'xr')
# for i in cl1: plot(model[i,0],model[i,1],'xb')


xlabel('x (pc)')
ylabel('y (pc)')

colorbar(label='Jy / beam')

fig = matplotlib.pyplot.gcf()
fig.set_size_inches(8,6)

savefig('template_basic.pdf')

clf()

imshow(im_obs[502:522,502:522], vmin=0, vmax=1, aspect='equal', extent=(-0.1,0.1,-0.1,0.1))

xlabel('x (pc)')
ylabel('y (pc)')

colorbar(label='Jy / beam')

fig = matplotlib.pyplot.gcf()
fig.set_size_inches(8,6)

savefig('template_basic_zoom.pdf')


