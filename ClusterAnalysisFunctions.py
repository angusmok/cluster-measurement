#------------------------------------------------------------------------------
###
#------------------------------------------------------------------------------
###
#------------------------------------------------------------------------------

#######################################################
# Declares Libraries
#######################################################
### Standard Libaries ###
import sys
import os
### Additional Libraries ###
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as mplpath
import numpy as np
import astropy.wcs as wcs
import aplpy
### Cluster Analysis Libraries ###
import astropy.io.fits as fits
from astropy.stats import mad_std, sigma_clipped_stats
from astropy.nddata.utils import Cutout2D
from photutils import DAOStarFinder, aperture_photometry, CircularAperture, CircularAnnulus
from photutils.utils import calc_total_error
from photutils.background import Background2D
from scipy import ndimage
#######################################################
# Set Parameters
#######################################################
### Non-Passing Variables
sn_vmin = 0.5
sn_vmax = 20

# Create ClusterTemp Folder to prevent creating more folders if run with AnalysisFunctions
if not os.path.exists('./ClusterTemp/'):
	os.makedirs('./ClusterTemp')
#------------------------------------------------------------------------------
###
#------------------------------------------------------------------------------
###
#------------------------------------------------------------------------------

print('Import - ClusterAnalysisFunctions - v2020.07')

def readimageanddatared(image_name, fits_data_slice, output_name, plot_title, fits_cut, fileloc = 0):

	"""
	Function: Read images and create simple plots
	"""

	print('>>> Running readimageanddatared')
	if fileloc == 0:
		image_file = fits.open(image_name + '.fits')
	else:
		image_file = fits.open('./All_Files/' + image_name + '.fits')
	image_file.info()
	image_org_wcs = wcs.WCS(image_file[fits_data_slice].header)
	
	# Create cut out 
	if fits_cut[0] > 0:
		image_sci_full = image_file[fits_data_slice].data
		image_sci_cutout = Cutout2D(image_sci_full, (fits_cut[0], fits_cut[1]), (fits_cut[2] * 2, fits_cut[3] * 2), wcs = image_org_wcs)
		image_sci = image_sci_cutout.data
		image_wcs = image_sci_cutout.wcs

	else:
		image_sci_full = image_file[fits_data_slice].data
		image_sci = image_sci_full
		image_wcs = image_org_wcs
	
	# Create rms normalized results (based on region chosen)
	image_sigma = mad_std(image_sci)
	image_sci_rmsnorm = image_sci / image_sigma
	image_sci_full_rmsnorm = image_sci_full / image_sigma

	# Create arrays for plotting (setting negatives to zero)
	image_sci_rmsnorm_plot = image_sci_rmsnorm
	image_sci_full_rmsnorm_plot = image_sci_full_rmsnorm
	image_sci_rmsnorm_plot[image_sci_rmsnorm_plot < 0] = 0
	image_sci_full_rmsnorm_plot[image_sci_full_rmsnorm_plot < 0] = 0

	# Create background image
	bkg = Background2D(image_sci, (7, 7))

	# Calculate Error Array
	image_exptime = float(image_file[0].header['EXPTIME'])
	print('From Header: Exposure Time = {}'.format(image_exptime))
	image_error = calc_total_error(image_sci, bkg.background_rms, image_exptime)

	# Plot Full File + Rectangle
	fig, ax = plt.subplots(1, figsize = (20, 20))
	plt.imshow(np.sqrt(image_sci_full_rmsnorm), origin = 'lower', cmap = 'Greys_r', vmin = sn_vmin, vmax = sn_vmax)
	plt.plot(fits_cut[0], fits_cut[1], 'rs')
	rect = patches.Rectangle((fits_cut[0] - fits_cut[2], fits_cut[1] - fits_cut[3]), fits_cut[2] * 2, fits_cut[3] * 2, linewidth = 1, edgecolor = 'r',facecolor = 'none')
	ax.add_patch(rect)
	plt.title(plot_title)
	plt.axis([fits_cut[0] - 1000, fits_cut[0] + 1000, fits_cut[1] - 1000, fits_cut[1] + 1000])
	plt.savefig(output_name + '01_rms_full.jpg')
	plt.close()

	# Plot cut region
	fig, ax = plt.subplots(1, figsize = (20, 20))
	plt.imshow(np.sqrt(image_sci_rmsnorm_plot), origin = 'lower', cmap = 'Greys_r', vmin = sn_vmin, vmax = sn_vmax)
	plt.title(plot_title)
	plt.savefig(output_name + '02_rms_cut.jpg')
	plt.close()

	# Plot cut region (background image)
	fig, ax = plt.subplots(1, figsize = (20, 20))
	plt.imshow(bkg.background, origin = 'lower', cmap = 'Greys_r')
	plt.title(plot_title)
	plt.savefig(output_name + '03_background.jpg')
	plt.close()

	print('Size of Output Array: {}'.format(len(image_sci)))
	print('{} Pixel Value Range: {:.2e} to {:.2e}'.format(image_name, np.nanmin(image_sci), np.nanmax(image_sci)))
	print('{} S/N Value Range: {:.2e} to {:.2e}'.format(image_name, np.nanmin(image_sci_rmsnorm), np.nanmax(image_sci_rmsnorm)))
	
	return image_sci, image_sigma, image_wcs, image_error, image_exptime