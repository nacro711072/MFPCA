import numpy as np
import scipy as sp
from scipy import signal
import arrayfire as af
from numba import autojit

###
# Kernel function
###
def EpaFun(x):
    """ Kernel function
    input :
        x : n * p matrix
    """
    w = np.zeros(x.shape[0])
    s = (np.absolute(x) < 1).all(axis = 1)
    w[s] = np.prod((1 - np.compress(s, x, axis = 0)**2), axis = 1)
    return(w)

def GauFun(x):
    return(np.exp(-np.sum(x**2, 1) / 2))

###
# BinData
###

def Check_Bound(x, x0):
    d = x0.shape[-1]
    x_min_out = (x < x0.reshape(-1, d).min(0)).any(1)
    x_max_out = (x > x0.reshape(-1, d).max(0)).any(1)
    return(~(x_min_out | x_max_out))

#@autojit(cache=True)
def Bin_Data(x, y, x0, bin_weight = True):
    grid_shape, d = np.asarray(x0.shape[:-1]), x0.shape[-1]
    bin_width = np.ptp(x0.reshape(-1, d), 0) / (grid_shape - 1)
    boundary = Check_Bound(x, x0)
    x = np.compress(boundary, x, 0)
    y = np.compress(boundary, y)
    if bin_weight is True:
        bin_shape = grid_shape + np.ones(d, 'int')
        newx_int = ((x - x0.reshape(-1, d).min(0)) / bin_width).astype('int')
        newx_float = (x - x0.reshape(-1, d).min(0)) / bin_width - newx_int
        bin_number = np.sum(np.asarray([newx_int + (i & (1 << np.arange(d)[::-1]) > 0).astype('int') for i in range(2**d)]) *
                            np.append(bin_shape[::-1].cumprod()[-2::-1], 1), 2)
        w = np.array([1 - newx_float, newx_float])
        index = np.indices([2] * d).reshape(d, -1).T
        linear_w = w.take(0, 2).T
        for i in range(1, d):
            linear_w = np.einsum('ij,ki->ijk', linear_w, w.take(i, 2)).reshape(-1, 2**(i + 1)) #outer
        binx = np.bincount(bin_number.reshape(-1), linear_w.T.reshape(-1), minlength = bin_shape.prod())
        sumy = np.bincount(bin_number.reshape(-1), (linear_w.T * y).reshape(-1), minlength = bin_shape.prod())
        binx = binx.reshape(bin_shape)
        sumy = sumy.reshape(bin_shape)
        for i in range(bin_shape.size):
            binx = np.delete(binx, -1, i)
            sumy = np.delete(sumy, -1, i)
    else:
        position = np.rint((x - x0.reshape(-1, d).min(0)) / bin_width).astype('int')
        bin_number = np.sum(position * np.append(grid_shape[::-1].cumprod()[-2::-1], 1), 1)
        binx = np.bincount(bin_number, minlength = np.prod(grid_shape)).reshape(grid_shape)
        sumy = np.bincount(bin_number, y, minlength = np.prod(grid_shape)).reshape(grid_shape)
    return(np.asarray([sumy, binx]))

def Extend_Bin_Data(bin_data, r):
    if np.all(r == 0):
        return(bin_data)
    sumy, binx = bin_data
    extend_range = [[k] * 2 for k in r]
    sumy = np.pad(sumy, extend_range, 'constant', constant_values = 0)
    binx = np.pad(binx, extend_range, 'constant', constant_values = 0)
    return(np.asarray([sumy, binx]))

###
# Main
###

def Get_Range(bin_width, h, ker_fun):
    r = np.floor(h / bin_width).astype(np.int32)
    if ker_fun == 'Gaussian':
        r = 4 * r
    return(r)

def Get_Delta_x(bin_width, r):
    return((np.indices(2 * r + 1).reshape(r.size, -1).T - r) * bin_width)

def Get_Weight(delta_x, h, ker_fun):
    if ker_fun == 'Epan':
        return(EpaFun(delta_x / h))
    elif ker_fun == 'Gaussian':
        return(GauFun(delta_x / h))

def Get_Linear_Solve(xt, weight, bin_data_num, bin_data_y, r):
    p, n = xt.shape
    xtw = xt * weight
    n_grid = np.prod(bin_data_num.shape - (2 * r))
    s = np.zeros((p, p, n_grid))
    t = np.zeros((p, n_grid))
    for i in range(p):
        for j in range(i, p):
            if i is 0:
                kernel = xtw.take(j, 0).reshape(2 * r + 1)
                t[j] = sp.signal.fftconvolve(kernel, bin_data_y, mode = 'valid').reshape(-1)
            else:
                kernel = np.reshape(xtw.take(i, 0) * xt.take(j, 0), 2 * r + 1)
            s[i, j] = sp.signal.fftconvolve(kernel, bin_data_num, mode = 'valid').reshape(-1)
            s[j, i] = s[i, j]
    for i in range(1, p):
        s[i, i] += 1e-12
    s = s.reshape(p**2, -1).T.reshape(-1, p, p)
    try:
        fit_y = np.linalg.solve(s, t.T).take(0, 1)
    except:
        indptr = np.arange(n_grid + 1)
        indices = np.arange(n_grid)
        s_sparse = sp.sparse.bsr_matrix((s, indices, indptr), shape = (p * n_grid, p * n_grid)).tocsc()
        fit_y = sp.sparse.linalg.spsolve(s_sparse, t.T.reshape(-1))[::p]
    return(fit_y)

def Lpr_For_Bin(bin_data, bin_width, h, ker_fun, dtype):
    d = bin_width.size
    r = Get_Range(bin_width, h, ker_fun)
    delta_x = Get_Delta_x(bin_width, r)
    weight = Get_Weight(delta_x, h, ker_fun)
    big_x = np.hstack((np.ones((delta_x.shape[0], 1)), delta_x))
    if d <= 4:
        biny, binx = bin_data
        s, t = Get_Linear_Equation_Gpu(af.from_ndarray(big_x.astype(dtype)),
               af.from_ndarray(weight.astype(dtype)),
               af.from_ndarray(binx.astype(dtype)),
               af.from_ndarray(biny.astype(dtype)),
               r,
               dtype)
        try:
            fit_y = np.linalg.solve(s, t.reshape(-1, d + 1)).take(0, 1)
        except:
            n_grid = s.shape[0]
            indptr = np.arange(n_grid + 1)
            indices = np.arange(n_grid)
            s_sparse = sp.sparse.bsr_matrix((s, indices, indptr), shape = ((d + 1) * n_grid, (d + 1) * n_grid)).tocsc()
            fit_y = sp.sparse.linalg.spsolve(s_sparse, t)[::(d + 1)]
    else:
        biny, binx = Extend_Bin_Data(bin_data, r)
        fit_y = Get_Linear_Solve(big_x.T, weight, binx, biny, r)
    return(fit_y)

def Lpr(x, y, x0, h, binning = True, bin_weight = True, ker_fun = 'Epan', dtype = 'f4'):
    if x.shape[0] != y.size:
        ValueError("Data size of x, y are not equal!")
    if x.shape[1] != x0.shape[-1]:
        ValueError("Different dimension between x and x0!")
    d = x.shape[1]
    non_nan_value = ~np.isnan(y)
    if binning is True:
        bin_width = np.ptp(x0.reshape(-1, d), 0) / (np.asarray(x0.shape[:-1]) - 1)
        bin_data = Bin_Data(np.compress(non_nan_value, x, 0), np.compress(non_nan_value, y), x0, bin_weight)
        fit_y = Lpr_For_Bin(bin_data, bin_width, h, ker_fun, dtype)
    else:
        x0 = x0.reshape(-1, d)
        fit_y = np.zeros(x0.shape[0])
        x = np.compress(non_nan_value, x, 0)
        y = np.compress(non_nan_value, y)
        for i in range(x0.shape[0]):
            I = np.identity(d + 1)
            I[0][0] = 0
            delta_x = x - x0.take(i, 0)
            weight = Get_Weight(delta_x, h, ker_fun)
            big_x = np.hstack((np.ones((x.shape[0], 1)), delta_x))
            xtw = big_x.T * weight
            s = np.matmul(xtw, big_x)
            t = np.matmul(xtw, y)
            fit_y[i] = np.linalg.lstsq(s + I, t)[0][0]
    return(fit_y)

#####

# CV for partition method
def Partition_Data_Size(N, ratio = 0.85):
    def proposition(x, N):
        return(x + x**ratio - N)
    n_train = np.ceil(sp.optimize.brenth(proposition, N / 2, N, args=(N))).astype(int)
    return(n_train)

def CV_Partition(x, y, x0, h, n_train = None, binning = True, bin_weight = True, ker_fun = 'Epan', dtype = 'f4'):
    n_h, d = h.shape
    non_nan_val = ~np.isnan(y)
    y = np.compress(non_nan_val, y)
    x = np.compress(non_nan_val, x, 0)
    random_order = np.random.permutation(y.size)
    if n_train is None:
        n_train = Partition_Data_Size(x.shape[0])
    train_x, test_x = np.split(x.take(random_order, 0), [n_train])
    train_y, test_y = np.split(y.take(random_order), [n_train])
    x0_min = x0.reshape(-1, d).min(0)
    x0_max = x0.reshape(-1, d).max(0)
    grid_x = tuple(np.linspace(x0_min.take(i), x0_max.take(i), x0.shape[i]) for i in range(d))
    ssq = np.zeros(n_h)
    if binning is True:
        bin_width = np.ptp(x0.reshape(-1, d), 0) / (np.asarray(x0.shape[:-1]) - 1)
        bin_data = Bin_Data(train_x, train_y, x0, bin_weight)
        for i in range(n_h):
            fit_y = Lpr_For_Bin(bin_data, bin_width, h.take(i, 0), ker_fun, dtype).reshape(x0.shape[:-1])
            if np.isnan(fit_y).any():
                ssq[i] = np.nan
                continue
            inter_fun = sp.interpolate.RegularGridInterpolator(grid_x, fit_y)
            ssq[i] = ((test_y - inter_fun(test_x))**2).sum()
    else:
        for i in range(n_h):
            fit_y = Lpr(train_x, train_y, test_x, h.take(i, 0),
                        binning = binning, ker_fun = ker_fun)
            ssq[i] = ((test_y - fit_y)**2).sum()
    h_opt = h.take(np.nanargmin(ssq), 0)
    return(h_opt)

#########################

###
# GPU Arrayfire
###

# Set backend
def Set_arrayfire_backend(backend):
    if backend not in ['cuda', 'opencl', 'cpu']:
        raise ValueError("Must be 'cuda', 'opencl' or 'cpu'")
    af.set_backend(backend)

@autojit(cache=True)
def Convolve4(signal, kernel):
    if kernel.elements() is 1:
        return(signal * kernel.to_list()[0])
    signal_d = signal.shape
    if len(kernel.shape) is 3:
        out = af.constant(0, signal_d[0], signal_d[1], signal_d[2], signal_d[3])
        for i in range(signal.shape[3]):
            out[:, :, :, i] = af.fft_convolve3(signal[:, :, :, i], kernel)
        return(out)
    extend_space = kernel.shape[3] - 1
    out = af.constant(0, signal_d[0], signal_d[1], signal_d[2], signal_d[3] + extend_space)
    for i in range(extend_space + 1):
        for j in range(signal.shape[3]):
            out[:, :, :, i + j] = out[:, :, :, i + j] + af.fft_convolve3(signal[:, :, :, j], kernel[:, :, :, i])
    return(out[:, :, :, (extend_space / 2):(-extend_space / 2)])

@autojit(cache=True)
def Get_Linear_Equation_Gpu(x, weight, bin_data_num, bin_data_y, r, dtype = 'f4'):
    n, p = x.shape
    d = p - 1
    if dtype is 'f4':
        dtype = af.Dtype.f32
    elif dtype is 'f8':
        dtype = af.Dtype.f64
    xw = af.constant(0, n, p, dtype = dtype)
    for ii in af.ParallelRange(p):
        xw[:, ii] = x[:, ii] * weight
    s = af.constant(0, p, p, np.prod(bin_data_num.shape), dtype = dtype)
    t = af.constant(0, p, np.prod(bin_data_num.shape), dtype = dtype)
    ker_d = np.ones(4, dtype = 'int')
    ker_d[:d] = 2 * r + 1
    if d is 4:
        for i in range(p):
            for j in range(i, p):
                if i is 0:
                    kernel = af.moddims(xw[:, j], ker_d[0], ker_d[1], ker_d[2], ker_d[3])
                    t[j] = af.flat(af.reorder(Convolve4(bin_data_y, kernel), 3, 2, 1, 0))
                else:
                    kernel = af.moddims(xw[:, i] * x[:, j], ker_d[0], ker_d[1], ker_d[2], ker_d[3])
                s[i, j] = af.flat(af.reorder(Convolve4(bin_data_num, kernel), 3, 2, 1, 0))
                s[j, i] = s[i, j]
    elif d < 4:
        for i in range(p):
            for j in range(i, p):
                if i is 0:
                    kernel = af.moddims(xw[:, j], ker_d[0], ker_d[1], ker_d[2], ker_d[3])
                    if kernel.elements() is 1:
                        t[j] = af.flat((bin_data_y * kernel.to_list()[0]).T)
                    else:
                        t[j] = af.flat(af.fft_convolve(bin_data_y, kernel).T)
                else:
                    kernel = af.moddims(xw[:, i] * x[:, j], ker_d[0], ker_d[1], ker_d[2], ker_d[3])
                if kernel.elements() is 1:
                    s[i, j] = af.flat((bin_data_num * kernel.to_list()[0]).T)
                else:
                    s[i, j] = af.flat(af.fft_convolve(bin_data_num, kernel).T)
                s[j, i] = s[i, j]
    for i in range(1, p):
        s[i, i] += 1e-12
    return([np.array(s).reshape(p**2, -1).T.reshape(-1, p, p), np.array(af.flat(t))])
