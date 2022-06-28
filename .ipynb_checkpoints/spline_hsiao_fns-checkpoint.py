import numpy as np
import h5py

def invKD_irr(x):
  """
  Compute K^{-1}D for a set of spline knots.
  ​
  For knots y at locations x, the vector, y'' of non-zero second
  derivatives is constructed from y'' = K^{-1}Dy, where K^{-1}D
  is independent of y, meaning it can be precomputed and reused for
  arbitrary y to compute the second derivatives of y.
  ​
  Parameters
  ----------
  x : :py:class:`numpy.array`
    Numpy array containing the locations of the cubic spline knots.

  Returns
  -------
  KD : :py:class:`numpy.array`
    y independednt matrix whose product can be taken with y to
    obtain a vector of second derivatives of y.
  """
  n = len(x)
  K = np.zeros((n-2,n-2))
  D = np.zeros((n-2,n))


  K[0,0:2] = [(x[2] - x[0])/3, (x[2] - x[1])/6]
  K[-1, -2:n-2] = [(x[n-2] - x[n-3])/6, (x[n-1] - x[n-3])/3]

  for j in np.arange(2,n-2):
    row = j - 1
    K[row, row-1:row+2] = [(x[j] - x[j-1])/6, (x[j+1] - x[j-1])/3, (x[j+1] - x[j])/6]
  for j in np.arange(1,n-1):
    row = j - 1
    D[row, row:row+3] = [1./(x[j] - x[j-1]), -(1./(x[j+1] - x[j]) + 1./(x[j] - x[j-1])), 1./(x[j+1] - x[j])]

  M = np.zeros((n,n))
  M[1:-1, :] = np.linalg.solve(K,D)
  return M

def spline_coeffs_irr(x_int, x, invkd, allow_extrap=True):
	"""
	Compute a matrix of spline coefficients.
​
	Given a set of knots at x, with values y, compute a matrix, J, which
	can be multiplied into y to evaluate the cubic spline at points
	x_int.
​
	Parameters
	----------
	x_int : :py:class:`numpy.array`
		Numpy array containing the locations which the output matrix will
		interpolate the spline to.
	x : :py:class:`numpy.array`
		Numpy array containing the locations of the spline knots.
	invkd : :py:class:`numpy.array`
		Precomputed matrix for generating second derivatives. Can be obtained
		from the output of ``invKD_irr``.
	allow_extrap : bool
		Flag permitting extrapolation. If True, the returned matrix will be
		configured to extrapolate linearly beyond the outer knots. If False,
		values which fall out of bounds will raise ValueError.
	
	Returns
	-------
	J : :py:class:`numpy.array`
		y independednt matrix whose product can be taken with y to evaluate
		the spline at x_int.
	"""
	n_x_int = len(x_int)
	n_x = len(x)
	X = np.zeros((n_x_int,n_x))

	if not allow_extrap and ((max(x_int) > max(x)) or (min(x_int) < min(x))):
		raise ValueError("Interpolation point out of bounds! " + 
			"Ensure all points are within bounds, or set allow_extrap=True.")
	
	for i in range(n_x_int):
		x_now = x_int[i]
		if x_now > max(x):
			h = x[-1] - x[-2]
			a = (x[-1] - x_now)/h
			b = 1 - a
			f = (x_now - x[-1])*h/6.0

			X[i,-2] = a
			X[i,-1] = b
			X[i,:] = X[i,:] + f*invkd[-2,:]
		elif x_now < min(x):
			h = x[1] - x[0]
			b = (x_now - x[0])/h
			a = 1 - b
			f = (x_now - x[0])*h/6.0

			X[i,0] = a
			X[i,1] = b
			X[i,:] = X[i,:] - f*invkd[1,:]
		else:
			q = np.where(x[0:-1] <= x_now)[0][-1]
			h = x[q+1] - x[q]
			a = (x[q+1] - x_now)/h
			b = 1 - a
			c = ((a**3 - a)/6)*h**2
			d = ((b**3 - b)/6)*h**2

			X[i,q] = a
			X[i,q+1] = b
			X[i,:] = X[i,:] + c*invkd[q,:] + d*invkd[q+1,:]

	return X

def read_model_grid(grid_file=None, grid_name=None):
    """
    Read the Hsiao grid file
    Parameters
    ----------
    grid_file : None or str
        Filename of the Hsiao model grid HDF5 file. If ``None`` reads the
        ``hsiao.hdf5`` file included with the :py:mod:`SNmodel`
        package.
    grid_name : None or str
        Name of the group name in the HDF5 file to read the grid from. If
        ``None`` uses ``default``
    Returns
    -------
    phase : array-like
        The phase array of the grid with shape ``(nphase,)``
    wave : array-like
        The wavelength array of the grid with shape ``(nwave,)``
    flux : array-like
        The DA white dwarf model atmosphere flux array of the grid.
        Has shape ``(nphase, nwave)``
    Notes
    -----
        There are no easy command line options to change this deliberately
        because changing the grid file essentially changes the entire model,
        and should not be done lightly, without careful comparison of the grids
        to quantify differences.
    See Also
    --------
    :py:class:`SNmodel.SNmodel`
    """

    if grid_file is None:
        # grid_file = os.path.join('templates','hsiao.h5')
        grid_file = '../parameter files/hsiao.h5'

    # # if the user specfies a file, check that it exists, and if not look inside the package directory
    # if not os.path.exists(grid_file):
    #     grid_file = get_pkgfile(grid_file)

    if grid_name is None:
        grid_name = "default"

    with h5py.File(grid_file, 'r') as grids:
        try:
            grid = grids[grid_name]
        except KeyError as e:
            message = '{}\nGrid {} not found in grid_file {}. Accepted values are ({})'.format(e, grid_name,\
                    grid_file, ','.join(list(grids.keys())))
            raise ValueError(message)

        phase = grid['phase'][()].astype('float64')
        wave  = grid['wave'][()].astype('float64')
        flux  = grid['flux'][()].astype('float64')

    return phase, wave, flux

def interpolate_hsiao(t_int, l_int, t_hsiao, l_hsiao, f_hsiao):
	"""
	Interpolates the Hsiao template to a particular time.
	Returns a slice of the Hsiao template, evaluated at an
	arbitrary time and grid of wavelengths.
	Parameters
	----------
	t_int : float
		Time at which to evaluate template
	l_int : :py:class:`numpy.array`
		Grid of rest frame wavelengths at which to return slice
	t_hsiao : :py:class:`numpy.array`
		Vector of times at which the template is known
	l_hsiao : :py:class:`numpy.array`
		Vector of wavelengths at which the template is known
	f_hsiao : :py:class:`numpy.array`
		Grid of fluxes coresponding to the provided time and 
		wavelength vectors.
	
	Returns
	-------
	f_int : :py:class:`numpy.array`
		Vector of fluxes defining the Hsiao template at wavelegths
		l_int at time t_int.     
	"""
	dt = t_hsiao[1] - t_hsiao[0]
	mask0 = (t_hsiao <= t_int)*(t_hsiao > t_int - dt)
	mask1 = (t_hsiao > t_int)*(t_hsiao <= t_int + dt)
	t0 = t_hsiao[mask0]
	t1 = t_hsiao[mask1]

	f_int = (f_hsiao[mask0, :]*(t1 - t_int) + f_hsiao[mask1, :]*(t_int - t0))/dt

	f_int = np.interp(l_int, l_hsiao, f_int.flatten())

	return f_int
