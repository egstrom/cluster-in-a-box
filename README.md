cluster-in-a-box
================

Statistical model of sub-millimeter emission from embedded protostellar clusters. The paper describing the model is currently in press (Kristensen & Bergin 2015, ApJL). The model is written in python and uses astropy. 

If using this model, please cite the DOI along with the paper (link to appear when paper is published). 
<a href="http://dx.doi.org/10.5281/zenodo.13184"><img src="https://zenodo.org/badge/doi/10.5281/zenodo.13184.svg" alt="10.5281/zenodo.13184"></a>

The model consists of three modules grouped in two scripts. The first (cluster_distribution) generates the cluster based on the number of stars, input initial mass function, spatial distribution and age distribution. The second (cluster_emission) takes an input file of observations, determines the mass-intensity correlation and generates outflow emission for all low-mass Class 0 and I sources. The output is stored as a FITS image where the flux density is determined by the desired resolution, pixel scale and cluster distance. 

Future updates to be implemented:
* Implement an evolutionary module; currently the number of stars and age distribution are input but sometimes a cloud mass and lifetime are desirable instead, with the appropriate star formation efficiency as a free parameter. 
* Implement other transitions and species, in particular water and high-J CO as based on Herschel observations. 
* Implement velocity distribution and generate (position, position, velocity) cubes rather than just images of velocity-integrated intensity. 
