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

print('Import - ClusterAnalysisFunctions - v2020.07')

#------------------------------------------------------------------------------
###
#------------------------------------------------------------------------------
###
#------------------------------------------------------------------------------

def readimageanddatared(image_name, fits_data_slice, output_name, plot_title, fits_cut, fileloc = 0):

	'''
	Function: Read images and create simple plots
	'''

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

def clusterfindsubfunc(image_array, image_sigma, fwhm, threshold, apertureplotrad = 7):

	'''
	Function: Find cluster and output results (wrapper around DAOfild)
	'''

	daofind = DAOStarFinder(fwhm = fwhm, threshold = threshold * image_sigma)
	sources = daofind(image_array)

	# Output source detection table
	for col in sources.colnames:
		sources[col].info.format = '%.8g'
	sources_cut_xcentroid = sources['xcentroid']
	sources_cut_ycentroid = sources['ycentroid']
	positions = convert2xNtoarray((sources_cut_xcentroid, sources_cut_ycentroid))
	positions_out = positions
	apertures = CircularAperture(positions_out, r = fwhm)
	apertures_plot = CircularAperture(positions_out, r = fwhm + apertureplotrad)

	print('Found {} using FWHM = {} and Threshold = {}'.format(len(sources_cut_xcentroid), fwhm, threshold))

	return sources, positions, apertures_plot

def clusterfind(image_array, image_sigma, output_name, plot_title, fwhm, flag_type = 0):

	'''
	Function: Find clusters and output results
	'''

	print('>>> Finding clusters on: {}'.format(plot_title))
	
	O30_10_sources, O30_10_positions, O30_10_apertures_plot = clusterfindsubfunc(image_array, image_sigma, fwhm, 10., apertureplotrad = 2)
	O30_5_sources, O30_5_positions, O30_5_apertures_plot = clusterfindsubfunc(image_array, image_sigma, fwhm, 5., apertureplotrad = 1.5)
	O30_4_sources, O30_4_positions, O30_4_apertures_plot = clusterfindsubfunc(image_array, image_sigma, fwhm, 4., apertureplotrad = 1)
	O30_3_sources, O30_3_positions, O30_3_apertures_plot = clusterfindsubfunc(image_array, image_sigma, fwhm, 3., apertureplotrad = 1)
	if flag_type == 2:
		O30_2_sources, O30_2_positions, O30_2_apertures_plot = clusterfindsubfunc(image_array, image_sigma, fwhm, 2., apertureplotrad = 1)
	if flag_type == 1.5:
		O30_1_5_sources, O30_1_5_positions, O30_1_5_apertures_plot = clusterfindsubfunc(image_array, image_sigma, fwhm, 1.5, apertureplotrad = 1)

	# Create arrays for plotting (setting negatives to zero)
	image_array_plot = image_array / image_sigma
	image_array_plot[image_array_plot < 0] = 0

	plt.figure(figsize = (20, 20))
	plt.imshow(np.sqrt(image_array_plot), origin = 'lower', cmap = 'Greys_r', vmin = sn_vmin, vmax = sn_vmax)
	plt.title(plot_title)
	plt.legend(title = 'Blue = 4, Green = 5, Red = 10')
	O30_10_apertures_plot.plot(color = 'red', lw = 2.5, alpha = 0.9)
	O30_5_apertures_plot.plot(color = 'green', lw = 2.5, alpha = 0.9)
	O30_4_apertures_plot.plot(color = 'blue', lw = 2.5, alpha = 0.9)
	plt.savefig(output_name + '03_rms_det_1.jpg')
	plt.close()

	plt.figure(figsize = (20, 20))
	plt.imshow(np.sqrt(image_array_plot), origin = 'lower', cmap = 'Greys_r', vmin = sn_vmin, vmax = sn_vmax)
	# plt.imshow(np.log10(image_array_plot + 0.0001), origin = 'lower', cmap = 'Greys_r', vmin = np.log10(sn_vmin + 1.5), vmax = np.log10(sn_vmax + 70))
	plt.title(plot_title)
	if flag_type == 1.5:
		O30_1_5_apertures_plot.plot(color = 'blue', lw = 2.5, alpha = 0.9)
	if flag_type == 2:
		O30_2_apertures_plot.plot(color = 'blue', lw = 2.5, alpha = 0.9)
	if flag_type == 3:
		O30_3_apertures_plot.plot(color = 'blue', lw = 2.5, alpha = 0.9)
	elif flag_type == 4:
		O30_4_apertures_plot.plot(color = 'blue', lw = 2.5, alpha = 0.9)
	elif flag_type == 5:
		O30_5_apertures_plot.plot(color = 'blue', lw = 2.5, alpha = 0.9)
	elif flag_type == 10:
		O30_10_apertures_plot.plot(color = 'blue', lw = 2.5, alpha = 0.9)
	plt.savefig(output_name + '03_rms_det_2.jpg')
	plt.close()

	if flag_type == 1.5:
		return O30_1_5_sources, O30_1_5_positions
	elif flag_type == 2:
		return O30_2_sources, O30_2_positions
	elif flag_type == 3:
		return O30_3_sources, O30_3_positions
	elif flag_type == 4:
		return O30_4_sources, O30_4_positions
	elif flag_type == 5:
		return O30_5_sources, O30_5_positions
	elif flag_type == 10:
		return O30_10_sources, O30_10_positions
	else:
		return 0, 0

def measureaperture(positions_array, image_array, output_name, param_array, zero_point, image_error, image_exptime):

	'''
	Perform aperture measurements + error calculations
	'''

	print('>>> Performing Aperture Photometry on {}'.format(output_name))
	aperture = CircularAperture(positions_array, r = param_array[0])
	annulus_aperture = CircularAnnulus(positions_array, r_in = param_array[1], r_out = param_array[2])
	annulus_masks = annulus_aperture.to_mask(method = 'center')
	apers_comb = [aperture, annulus_aperture]

	bkg_median = []
	for mask in annulus_masks:
		annulus_data = mask.multiply(image_array)
		annulus_data_1d = annulus_data[mask.data > 0]
		_, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)
		bkg_median.append(median_sigclip)
	bkg_median = np.array(bkg_median)
	phot = aperture_photometry(image_array, aperture, error = image_error)
	phot['annulus_median'] = bkg_median
	phot['aper_bkg'] = bkg_median * aperture.area
	phot['aper_sum_bkgsub'] = phot['aperture_sum'] - phot['aper_bkg']

	# image_sigma = mad_std(image_array)
	# image_error_simple = calc_total_error(image_sci, image_sigma, image_exptime)

	print('Output Array:')
	i = 0
	for col in phot.colnames:
		print('{} - {} (ex: {:.4f})'.format(i, phot[col].info.name, phot[0][col]))
		i = i + 1
	print(phot)

	# print('Test: {:.4f} vs {:.4f}'.format(phot[i][7], phot[i]['aper_sum_bkgsub']))

	f = open(output_name + '04_cat.txt', 'w')
	out_mag_array = []
	out_mag_bkgsub_array = []
	out_mag_bkgsub_error_array = []
	for i in range(0, len(phot)):

		# Convert 
	 	out_mag = -2.5 * np.log10(phot[i]['aperture_sum']) + zero_point
	 	out_mag_bkgsub = -2.5 * np.log10(phot[i]['aper_sum_bkgsub']) + zero_point

	 	out_mag_bkgsub_error = (phot[i]['aperture_sum_err'] / phot[i]['aperture_sum']) / 1.08
	 	
	 	out_mag_array.append(out_mag)
	 	out_mag_bkgsub_array.append(out_mag_bkgsub)
	 	out_mag_bkgsub_error_array.append(out_mag_bkgsub_error)
	 	
	 	#
	 	f.write('{}, {:.4f}, {:.4f}, {:.4f}, {:.4f}\n'.format(phot[i]['id'], phot[i]['xcenter'], phot[i]['ycenter'], out_mag, out_mag_bkgsub))
	f.close()
	
	np_out_mag_bkgsub_array = np.array(out_mag_bkgsub_array)
	np_out_mag_bkgsub_error_array = np.array(out_mag_bkgsub_error_array)

	plt.figure(figsize = (20, 20))
	plt.hist(np_out_mag_bkgsub_array, 10, facecolor = 'blue', alpha = 0.5)
	plt.hist(out_mag_array, 10, facecolor = 'green', alpha = 0.5)
	plt.title(output_name)
	plt.savefig(output_name + '04_cat_hist.png')
	plt.close()
 
	return np_out_mag_bkgsub_array, np_out_mag_bkgsub_error_array

def converttonumpy_gen(array, arrayname):

	'''
	Function: Output numpy array from original array
	'''

	# Convert to numpy array, check shape
	np_array = np.ndarray((len(array), len(array[0])), dtype = object)
	for i in range(0, len(array)):
		np_array[i] = tuple(array[i])
	print('- {}'.format(arrayname))

	return np_array

# Function: Read in model data
def read_BC03_M72_LEG():

	'''
	Function: Read model data
	'''

	read_BC03_M72_LEG_file = open('../bc2003_lr_BaSeL_m72_chab_ssp.legus_uvis1_color', 'r')
	read_BC03_M72_LEG_array = []
	###
	i = 1
	while 1:
		read_BC03_M72_LEG_line = read_BC03_M72_LEG_file.readline()
		if not read_BC03_M72_LEG_line:
			break
		else:
			if i > 30:
				read_BC03_M72_LEG_cols = read_BC03_M72_LEG_line.split()
				if len(read_BC03_M72_LEG_cols) != 19:
					print('Warning - Length of Data = {} != 19'.format(len(read_BC03_M72_LEG_cols)))
				else:
					read_BC03_M72_LEG_array.append([np.power(10, float(read_BC03_M72_LEG_cols[0])), float(read_BC03_M72_LEG_cols[5]), float(read_BC03_M72_LEG_cols[6]), float(read_BC03_M72_LEG_cols[10])])
		i = i + 1

	np_read_BC03_M72_LEG_array = converttonumpy_gen(read_BC03_M72_LEG_array, '')

	return np_read_BC03_M72_LEG_array

def read_BC03_M62():

	'''
	Function: Read model data
	'''

	read_BC03_M62_file = open('../bc2003_lr_BaSeL_m62_chab_ssp.acs_wfc_color', 'r')
	read_BC03_M62_array = []
	###
	i = 1
	while 1:
		read_BC03_M62_line = read_BC03_M62_file.readline()
		if not read_BC03_M62_line:
			break
		else:
			if i > 30:
				read_BC03_M62_cols = read_BC03_M62_line.split()
				if len(read_BC03_M62_cols) != 18:
					print('Warning - Length of Data = {} != 18'.format(len(read_BC03_M62_cols)))
				else:
					read_BC03_M62_array.append([np.power(10, float(read_BC03_M62_cols[0])), float(read_BC03_M62_cols[5]), float(read_BC03_M62_cols[7]), float(read_BC03_M62_cols[11])])
		i = i + 1

	np_read_BC03_M62_array = converttonumpy_gen(read_BC03_M62_array, '')

	return np_read_BC03_M62_array

def read_BC03_M42():

	'''
	Function: Read model data
	'''

	read_BC03_M42_file = open('../bc2003_lr_BaSeL_m42_chab_ssp.acs_wfc_color', 'r')
	read_BC03_M42_array = []
	###
	i = 1
	while 1:
		read_BC03_M42_line = read_BC03_M42_file.readline()
		if not read_BC03_M42_line:
			break
		else:
			if i > 30:
				read_BC03_M42_cols = read_BC03_M42_line.split()
				if len(read_BC03_M42_cols) != 18:
					print('Warning - Length of Data = {} != 18'.format(len(read_BC03_M42_cols)))
				else:
					read_BC03_M42_array.append([np.power(10, float(read_BC03_M42_cols[0])), float(read_BC03_M42_cols[5]), float(read_BC03_M42_cols[7]), float(read_BC03_M42_cols[11])])
		i = i + 1

	np_read_BC03_M42_array = converttonumpy_gen(read_BC03_M42_array, '')

	return np_read_BC03_M42_array
 
def isolateimageextension(image_name, extension):

	'''
	Funtion: Output isolated 
	'''

	if not os.path.exists(image_name + '_image.fits'):
		header = fits.getheader(image_name + '.fits', extension)
		data = fits.getdata(image_name + '.fits', extension)
		fits.writeto(image_name + '_image.fits', data, header)

	return 0

def convert2xNtoarray(array):

	'''
	Function: Convert 2xN to array
	'''

	output_array = []
	for i in range(0, len(array[0])):
		output_array.append((array[0][i], array[1][i]))

	return output_array

def filterbyloc(array, polygon_reg_path):
	
	'''
	Function: Filter array by location
	'''

	output_array_in = []
	output_array_out = []
	print('-> Running filterbyloc: ({})'.format(len(array)))
	for i in range(0, len(array)):
		if polygon_reg_path.contains_point([array[i][0], array[i][1]]) == True:
			output_array_in.append((array[i][0], array[i][1]))
		else:
			output_array_out.append((array[i][0], array[i][1]))
	print('-> Output: ({} -> {} in, {} out)'.format(len(array), len(output_array_in), len(output_array_out)))

	return output_array_in, output_array_out

# Function:
def filterbyval(array, array2, array2_value, flag):

	'''
	Function: Filter array by value
	'''

	output_array = []
	print('-> Running filterbyval: ({} = {})'.format(len(array), len(array2)))
	for i in range(0, len(array)):
		if np.isfinite(array2[i]):
			if flag == 'gte':
				if array2[i] >= array2_value:
					 output_array.append((array[i][0], array[i][1]))
			elif flag == 'lt':
				if array2[i] < array2_value:
					 output_array.append((array[i][0], array[i][1]))
	print('-> Output: ({})'.format(len(output_array)))

	return output_array

def printthreearraysbs(array1, array2, array3):

	'''
	Function: Print three arrays of equal length 
	'''

	for i in range(0, len(array1)):
		print('{:.2e}, {:.3f}, {:.3f}'.format(array1[i], array2[i], array3[i]))