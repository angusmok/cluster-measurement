# cluster-measurement
This repository contains useful python functions, including some functions for photometry and detection.

List of Current Functions:
- readimageanddatared(image_name, fits_data_slice, output_name, plot_title, fits_cut, fileloc = 0):
- clusterfindsubfunc(image_array, image_sigma, fwhm, threshold, apertureplotrad = 7):
- clusterfind(image_array, image_sigma, output_name, plot_title, fwhm, flag_type = 0):
- measureaperture(positions_array, image_array, output_name, param_array, zero_point, image_error, image_exptime):
- converttonumpy_gen(array, arrayname):
- read_BC03_M72_LEG():
- read_BC03_M62():
- read_BC03_M42():
- isolateimageextension(image_name, extension):
- convert2xNtoarray(array):
- filterbyloc(array, polygon_reg_path):
- filterbyval(array, array2, array2_value, flag):
- printthreearraysbs(array1, array2, array3):