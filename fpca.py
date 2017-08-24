import lpr
###
import numpy as np
import scipy as sp
from scipy import interpolate
from numba import autojit
import random

###

class Fpca(object):
    def __init__(self, x, y, x0, h_mean, h_cov, h_cov_dia, fve = 0.85,
                 binning = True, bin_weight = True, ker_fun = 'Epan', bw_select = 'Partition', dtype = 'f4'):
        self.__Check_Input(x, y, x0, h_mean, h_cov, h_cov_dia, fve, binning, bin_weight, ker_fun, bw_select, dtype)
        self.__num_fun = len(x)
        self.__d = x0.shape[-1]
        self.__grid_shape = np.asarray(x0.shape[:-1])
        self.__bin_width = np.ptp(x0.reshape(-1, self.__d), 0) / (self.__grid_shape - 1)
        self.__n_train = lpr.Partition_Data_Size(len(x))
        x0_min = x0.reshape(-1, self.__d).min(0)
        x0_max = x0.reshape(-1, self.__d).max(0)
        self.__grid = tuple(np.linspace(x0_min.take(i), x0_max.take(i), x0.shape[i]) for i in range(self.__d))
        self.__Main(x, y, x0, h_mean, h_cov, h_cov_dia, fve, binning, bin_weight, ker_fun, bw_select, dtype)
        self.__Standardize_Output()

    def __str__(self):
        print("Number of grid: %s" %" * ".join(map(str, self.__grid_shape)))
        print("Number of random function: %d" %self.__num_fun)
        print("="*20)
        print("Eigen pairs: %d" %self.num_eig_pairs)
        print("Sigma2: %f" %self.sigma2)
        print("Bandwidth of mean: [%s]" %", ".join(map(str, self.mean_bw)))
        print("Bandwidth of cov: [%s]" %", ".join(map(str, self.cov_bw)))
        print("Bandwidth of cov_diag: [%s]" %", ".join(map(str, self.cov_diag_bw)))
        return("="*20)

    def __Check_Input(self, x, y, x0, h_mean, h_cov, h_cov_dia, fve, binning, bin_weight, ker_fun, bw_select, dtype):
        if len(x) != len(y):
            raise ValueError("Different length between x and y.")
        # if len(y.shape) is not 2:
        #     raise ValueError("Input y must be 2-d.(functions, points)")
        if x[0].shape[-1] != x0.shape[-1]:
            raise ValueError("Different dimension bwtween x and x0.")
        if len(h_mean.shape) != 2:
            raise ValueError("Input h_mean must be 2darray.")
        if len(h_cov.shape) != 2:
            raise ValueError("Input h_cov must be 2darray.")
        if len(h_cov_dia.shape) != 2:
            raise ValueError("Input h_cov_diag must be 2darray.")
        if h_mean.shape[1] != h_cov.shape[1] * 2 != h_cov_dia.shape[1] != x0.shape[-1]:
            raise ValueError("Bandwidth dimesion error.")
        if type(fve) is not float:
            raise TypeError("Type of fve must be float.")
        elif fve >= 1 or fve <= 0:
            raise ValueError("fve must between 0 and 1.")
        if type(binning) is not bool:
            raise TypeError("Type of binning must be bool.")
        if type(bin_weight) is not bool:
            raise TypeError("Type of binning must be bool.")
        if ker_fun not in ['Gaussian', 'Epan']:
            raise ValueError("ker_fun should be 'Gaussian' or 'Epan'.")
        if bw_select not in ['Partition', 'LeaveOneOut']:
            raise ValueError("bw_select should be 'Partition' or 'LeaveOneOut'.")
        if dtype not in ['f4', 'f8']:
            raise ValueError("dtype must be 'f4' or 'f8'.")

    def __CV_Leave_One_Curve(self, x0, candidate_h, ker_fun, binning, dtype, **kwargs):
        """
        input variable:
        """
        grid_shape = np.array(x0.shape[:-1])
        d = x0.shape[-1]
        sse = np.ones(candidate_h.shape[0])
        bin_width = np.ptp(x0.reshape(-1, d), 0) / (grid_shape - 1)
        if binning == True:
            bin_data = kwargs['bin_data']
            test_bin_data = kwargs['test_bin_data']
            for i in range(candidate_h.shape[0]):
                h = candidate_h.take(i, 0)
                for test_data in test_bin_data:
                    train_bin_data = bin_data - test_data
                    fit_y = lpr.Lpr_For_Bin(train_bin_data, bin_width, h, ker_fun, dtype)
                    if np.any(np.isnan(fit_y)):
                        sse[i] = np.nan
                        break
                    nozero = (test_data.take(1, 0) != 0).flatten()
                    test_bin_data_nozero = np.compress(nozero, test_data.reshape(2, -1), 1)
                    sse[i] += (((test_bin_data_nozero.take(0, 0) - np.compress(nozero, fit_y) * test_bin_data_nozero.take(1, 0))**2) /
                                 test_bin_data_nozero.take(1, 0)).sum()
        else:
            x, y = kwargs['x'], kwargs['y']
            if self.__num_fun > 100:
                test_index = np.random.random_integers(0, self.__num_fun, 100)
            else:
                test_index = range(self.__num_fun)
            for i in range(n_h):
                h = candidate_h.take(i, 0)
                for j in test_index:
                    fit_y = lpr.Lpr(np.vstack(x[-j]), np.hstack(y[-j]), x[j], h.take(i, 0),
                                    binning = binning, ker_fun = ker_fun)
                    if np.any(np.isnan(fit_y)):
                        sse[i] = np.nan
                        break
                    ssq[i] += ((y[j] - fit_y)**2).sum()
        opt_h = candidate_h.take(np.nanargmin(sse), 0)
        return(opt_h)

    # @autojit(cache=True)
    def __BW_Partition(self, x0, candidate_h, ker_fun, binning, dtype, **kwargs):
        """
        input variable:
        """
        grid_shape = np.asarray(x0.shape[:-1])
        n_h, d = candidate_h.shape
        bin_width = np.ptp(x0.reshape(-1, d), 0) / (grid_shape - 1)
        ssq = np.zeros(n_h)
        if binning == True:
            train_bin_data = kwargs['train_bin_data']
            test_bin_data = kwargs['test_bin_data']
            for i in range(n_h):
                fit_y = lpr.Lpr_For_Bin(train_bin_data, bin_width, candidate_h.take(i, 0), ker_fun, dtype).reshape(x0.shape[:-1])
                if np.isnan(fit_y).any():
                    ssq[i] = np.nan
                    continue
                nozero = (test_bin_data.take(1, 0) != 0).flatten()
                test_bin_data_nozero = np.compress(nozero, test_bin_data.reshape(2, -1), 1)
                ssq[i] = (((test_bin_data_nozero.take(0, 0) - np.compress(nozero, fit_y.flatten()) * test_bin_data_nozero.take(1, 0))**2) /
                            test_bin_data_nozero.take(1, 0)).sum()
        else:
            x, y = kwargs['x'], kwargs['y']
            combined = list(zip(x, y))
            random.shuffle(combined)
            train_data, test_data = combined[:self.__n_train], combined[self.__n_train:]
            train_x, train_y = list(zip(*train_data))
            test_x, test_y = list(zip(*test_data))
            for i in range(n_h):
                fit_y = lpr.Lpr(np.vstack(train_x), np.hstack(train_y), np.vstack(test_x), h.take(i, 0),
                                binning = binning, ker_fun = ker_fun)
                ssq[i] = ((test_y - fit_y)**2).sum()
        h_opt = candidate_h.take(np.nanargmin(ssq), 0)
        return(h_opt)

    @autojit(cache=True)
    def __Fit_Mean(self, x, y, x0, candidate_h, binning, bin_weight, ker_fun, bw_select, dtype, bin_data = None):
        if binning == True:
            if bw_select == 'Partition':
                rand_index = np.random.permutation(self.__num_fun)
                train_bin_data, test_bin_data = np.split(bin_data.take(rand_index, 0), [self.__n_train])
                self.mean_bw = self.__BW_Partition(x0 = x0,
                                                   candidate_h = candidate_h,
                                                   ker_fun = ker_fun,
                                                   binning = binning,
                                                   dtype = dtype,
                                                   train_bin_data = train_bin_data.sum(0),
                                                   test_bin_data = test_bin_data.sum(0))
            elif bw_select == 'LeaveOneOut':
                if self.__num_fun > 100:
                    test_index = np.random.random_integers(0, self.__num_fun - 1, 100)
                else:
                    test_index = range(self.__num_fun)
                test_bin_data = bin_data.take(test_index, 0)
                self.mean_bw = self.__CV_Leave_One_Curve(x0 = x0,
                                                          candidate_h = candidate_h,
                                                          ker_fun = ker_fun,
                                                          binning = binning,
                                                          dtype = dtype,
                                                          bin_data = bin_data.sum(0),
                                                          test_bin_data = test_bin_data)

            self.mean_fun = lpr.Lpr_For_Bin(bin_data = bin_data.sum(0),
                                                bin_width = self.__bin_width,
                                                h = self.mean_bw,
                                                ker_fun = ker_fun,
                                                dtype = dtype)
        else:
            if bw_select == 'Partition':
                self.mean_bw = self.__BW_Partition(x0 = x0,
                                                   candidate_h = candidate_h,
                                                   ker_fun = ker_fun,
                                                   binning = binning,
                                                   dtype = dtype,
                                                   x = x, y = y)
            elif bw_select == 'LeaveOneOut':
                self.mean_bw = self.__CV_Leave_One_Curve(x0 = x0,
                                                          candidate_h = candidate_h,
                                                          ker_fun = ker_fun,
                                                          binning = binning,
                                                          dtype = dtype,
                                                          x = x, y = y)

            self.mean_fun = lpr.Lpr(np.vstack(x), np.hstack(y), x0, self.mean_bw,
                                    binning, bin_weight, ker_fun, dtype)

    @autojit(cache=True)
    def __Get_Row_Cov(self, x, y, cov_x0, binning, bin_weight, bin_data = None):
        cov_grid_shape = cov_x0.shape[:-1]
        num_fun = len(bin_data)
        if binning == True:
            bin_cov_diag_data = np.zeros((2, ) + cov_grid_shape)
            bin_cov_y = np.matmul(bin_data.take(0, 1).reshape(num_fun, -1).T, bin_data.take(0, 1).reshape(num_fun, -1))
            bin_cov_x = np.matmul(bin_data.take(1, 1).reshape(num_fun, -1).T, bin_data.take(1, 1).reshape(num_fun, -1))
            bin_cov_data = np.asarray([bin_cov_y, bin_cov_x])
            for i in range(num_fun):
                xx = np.tile(x[i], 2)
                yy = y[i]**2
                bin_cov_diag_data += lpr.Bin_Data(xx, yy, cov_x0, bin_weight)
            bin_cov_data = bin_cov_data.reshape((2, ) + cov_grid_shape) - bin_cov_diag_data
            return(bin_cov_data)
        else:
            xx = [None] * self.__num_fun
            yy = [None] * self.__num_fun
            for i in range(self.__num_fun):
                num_pt = y[i].size
                yy[i] = np.delete(np.outer(y[i], y[i]).flatten(), np.arange(0, num_pt**2, num_pt + 1))
                xx[i] = np.delete(np.vstack(np.dstack(x[i][np.mgrid[:num_pt, :num_pt]])), np.arange(0, num_pt**2, num_pt + 1), 0)
            return([xx, yy, cov_x0])

    # @autojit(cache=True)
    def __Fit_Cov(self, x, y, x0, candidate_h, binning, bin_weight, ker_fun, bw_select, dtype, bin_data = None):
        cov_x0 = x0.reshape(-1, self.__d)[np.mgrid[0:self.__grid_shape.prod(), 0:self.__grid_shape.prod()].T].reshape(np.append(self.__grid_shape.repeat(2), -1))
        if binning == True:
            if bw_select == 'Partition':
                rand_index = np.random.permutation(self.__num_fun)
                train_bin_data, test_bin_data = np.split(bin_data.take(rand_index, 0), [self.__n_train])
                train_cov_bin_data = self.__Get_Row_Cov([x[i] for i in rand_index[:self.__n_train]], [y[i] for i in rand_index[:self.__n_train]], cov_x0, binning, bin_weight, train_bin_data)
                test_cov_bin_data = self.__Get_Row_Cov([x[i] for i in rand_index[self.__n_train:]], [y[i] for i in rand_index[self.__n_train:]], cov_x0, binning, bin_weight, test_bin_data)
                self.cov_bw = self.__BW_Partition(x0 = cov_x0,
                                                   candidate_h = candidate_h,
                                                   ker_fun = ker_fun,
                                                   binning = binning,
                                                   dtype = dtype,
                                                   train_bin_data = train_cov_bin_data,
                                                   test_bin_data = test_cov_bin_data)
            elif bw_select == 'LeaveOneOut':
                if self.__num_fun > 100:
                    test_index = np.random.random_integers(0, self.__num_fun - 1, 100)
                else:
                    test_index = range(self.__num_fun)
                test_cov_bin_data = np.zeros((len(test_index), 2) + cov_x0.shape[:-1])
                for i in range(len(test_index)):
                    testi = test_index[i]
                    test_cov_bin_data[i] = self.__Get_Row_Cov([x[testi]], [y[testi]], cov_x0, binning, bin_weight, bin_data.take(testi, 0).reshape((1, ) + bin_data.shape[1:]))
                cov_bin_data = self.__Get_Row_Cov(x, y, cov_x0, binning, bin_weight, bin_data)
                self.cov_bw = self.__CV_Leave_One_Curve(x0 = cov_x0,
                                                        candidate_h = candidate_h,
                                                        ker_fun = ker_fun,
                                                        binning = binning,
                                                        dtype = dtype,
                                                        bin_data = cov_bin_data,
                                                        test_bin_data = test_cov_bin_data)

            fit_yy = lpr.Lpr_For_Bin(bin_data = self.__Get_Row_Cov(x, y, cov_x0, binning, bin_weight, bin_data),
                                         bin_width = np.tile(self.__bin_width, 2),
                                         h = self.cov_bw,
                                         ker_fun = ker_fun,
                                         dtype = dtype).reshape(np.repeat(np.prod(self.__grid_shape), 2))

        else:
            xx, yy, cov_x0 = self.__Get_Row_Cov(x, y, x0, binning, bin_weight)
            if bw_select == 'Partition':
                self.cov_bw = self.__BW_Partition(x0 = cov_x0,
                                                  candidate_h = candidate_h,
                                                  ker_fun = ker_fun,
                                                  bin_weight = bin_weight,
                                                  binning = binning,
                                                  dtype = dtype,
                                                  x = xx, y = yy)
            elif bw_select == 'LeaveOneOut':
                self.cov_bw = self.__CV_Leave_One_Curve(x0 = cov_x0,
                                                          candidate_h = candidate_h,
                                                          ker_fun = ker_fun,
                                                          binning = binning,
                                                          bin_weight = bin_weight,
                                                          dtype = dtype,
                                                          x = xx, y = yy)
            fit_yy = lpr.Lpr(x = np.vstack(xx),
                             y = np.hstack(yy),
                             x0 = cov_x0,
                             h = self.cov_bw,
                             binning = binning,
                             bin_weight = bin_weight,
                             ker_fun = ker_fun,
                             dtype = dtype).reshape(np.repeat(np.prod(self.__grid_shape), 2))

        fit_yy = (fit_yy.T + fit_yy) / 2
        self.cov_fun = fit_yy - np.outer(self.mean_fun, self.mean_fun)

    @autojit(cache=True)
    def __Fit_EigPairs(self, fve):
        e_val, e_vec = np.linalg.eig(self.cov_fun)
        adjust_eig_fun = e_vec / np.sqrt(np.prod(self.__bin_width))
        adjust_eig_val = e_val * np.prod(self.__bin_width)
        positive_eig_val_index = adjust_eig_val > 0
        self.eig_fun = np.compress(positive_eig_val_index, adjust_eig_fun, 1)
        self.eig_val = np.compress(positive_eig_val_index, adjust_eig_val)
        self.num_eig_pairs = np.count_nonzero(np.cumsum(self.eig_val) / self.eig_val.sum() <= fve) + 1
        self.eig_val = self.eig_val.take(np.arange(self.num_eig_pairs))
        self.eig_fun = self.eig_fun.take(np.arange(self.num_eig_pairs), 1).T

    # @autojit(cache=True)
    def __Fit_Sigma2(self, x, y, x0, candidate_h, binning, bin_weight, ker_fun, bw_select, dtype):
        yy = [yi**2 for yi in y]
        if binning == True:
            cov_diag_bin_data = np.asarray(list(map(lambda xy: lpr.Bin_Data(xy[0], xy[1], x0, bin_weight), zip(x, yy))))
            if bw_select == 'Partition':
                rand_index = np.random.permutation(self.__num_fun)
                train_bin_data, test_bin_data = np.split(cov_diag_bin_data.take(rand_index, 0), [self.__n_train])
                self.cov_diag_bw = self.__BW_Partition(x0 = x0,
                                                       candidate_h = candidate_h,
                                                       ker_fun = ker_fun,
                                                       binning = binning,
                                                       dtype = dtype,
                                                       train_bin_data = train_bin_data.sum(0),
                                                       test_bin_data = test_bin_data.sum(0))
            elif bw_select == 'LeaveOneOut':
                if self.__num_fun > 100:
                    test_index = np.random.random_integers(0, self.__num_fun - 1, 100)
                else:
                    test_index = range(self.__num_fun)
                test_bin_data = cov_diag_bin_data.take(test_index, 0)
                self.cov_diag_bw = self.__CV_Leave_One_Curve(x0 = x0,
                                                             candidate_h = candidate_h,
                                                             ker_fun = ker_fun,
                                                             binning = binning,
                                                             dtype = dtype,
                                                             bin_data = cov_diag_bin_data.sum(0),
                                                             test_bin_data = test_bin_data)

            self.cov_dia_fun = lpr.Lpr_For_Bin(bin_data = cov_diag_bin_data.sum(0),
                                         bin_width = self.__bin_width,
                                         h = self.cov_diag_bw,
                                         ker_fun = ker_fun,
                                         dtype = dtype) - self.mean_fun**2
        else:
            if bw_select == 'Partition':
                self.cov_bw = self.__BW_Partition(x0 = x0,
                                                  candidate_h = candidate_h,
                                                  ker_fun = ker_fun,
                                                  binning = binning,
                                                  dtype = dtype,
                                                  x = x, y = yy)
            elif bw_select == 'LeaveOneOut':
                self.cov_bw = self.__CV_Leave_One_Curve(x0 = x0,
                                                         candidate_h = candidate_h,
                                                         ker_fun = ker_fun,
                                                         binning = binning,
                                                         dtype = dtype,
                                                         x = x, y = yy)
            self.cov_dia_fun = lpr.Lpr(x = np.vstack(x),
                            y = np.hstack(yy),
                            x0 = x0,
                            h = self.cov_diag_bw,
                            binning = binning,
                            bin_weight = bin_weight,
                            ker_fun = ker_fun,
                            dtype = dtype) - self.mean_fun**2

        interval = np.array([(self.__grid_shape / 4).astype('int'), (3 * self.__grid_shape / 4).astype('int')])
        restruct_cov_fun = np.matmul(np.matmul(self.eig_fun.T, np.diag(self.eig_val)), self.eig_fun)
        sigma2_t = (self.cov_dia_fun - np.diag(restruct_cov_fun)).reshape(self.__grid_shape)
        for i in range(self.__d):
            left, right = interval.take(i, 1)
            sigma2_t = sigma2_t.take(np.arange(left, right), i)
        self.sigma2 = (sigma2_t).mean()
        self.sigma2 = 0 if self.sigma2 < 0 else self.sigma2

    def __Fit_Fpc_Scores(self, x, y):
        n = len(y)
        mean_inter_fun = sp.interpolate.RegularGridInterpolator(self.__grid, self.mean_fun.reshape(self.__grid_shape))
        phi_inter_fun = [sp.interpolate.RegularGridInterpolator(self.__grid,
                         self.eig_fun.take(i, 0).reshape(self.__grid_shape))
                         for i in range(self.num_eig_pairs)]
        fit_pc_score = np.zeros((n, self.num_eig_pairs))
        for i in range(n):
            y_center = y[i] - mean_inter_fun(x[i])
            phi_i = np.array([Eigi(x[i]) for Eigi in phi_inter_fun]) #(Eig, p)
            fit_cov_inv = np.linalg.pinv(np.matmul(phi_i.T * self.eig_val, phi_i) +
                                         np.identity(y_center.size) * (self.sigma2 + 1e-6))
            fit_pc_score[i] = self.eig_val * np.matmul(np.matmul(phi_i, fit_cov_inv), y_center)
        return(fit_pc_score)

    def __Standardize_Output(self):
        self.mean_fun = self.mean_fun.reshape(self.__grid_shape)
        self.cov_fun = self.cov_fun.reshape(np.tile(self.__grid_shape, 2))
        self.eig_fun = self.eig_fun.reshape(np.append(-1, self.__grid_shape))
        self.cov_dia_fun = self.cov_dia_fun.reshape(self.__grid_shape)

    # @autojit(cache=True)
    def __Main(self, x, y, x0, h_mean, h_cov, h_cov_dia, fve, binning, bin_weight, ker_fun, bw_select, dtype):
        """
        input:
            x: ndarray (#functions, #points, #dimension)
            y: ndarray (#functions, #points)
            x0: ndarray
            candidate_h: ndarray (#h, #dimension)
            num_grid: vector
        """
        ###
        if binning is True:
            bin_data = np.asarray(list(map(lambda xy: lpr.Bin_Data(xy[0], xy[1], x0, bin_weight), zip(x, y))))
        else:
            bin_data = None
        self.__Fit_Mean(x, y, x0, h_mean, binning, bin_weight, ker_fun, bw_select, dtype, bin_data)
        ###
        self.__Fit_Cov(x, y, x0, h_cov, binning, bin_weight, ker_fun, bw_select, dtype, bin_data)
        ###
        self.__Fit_EigPairs(fve)
        ###
        self.__Fit_Sigma2(x, y, x0, h_cov_dia, binning, bin_weight, ker_fun, bw_select, dtype)
        ###PACE
        self.fpc_scores = self.__Fit_Fpc_Scores(x, y)

    def Restruct_Fun(self, x, y):
        """
        input:
            x: 3d-array (#functions, #points, #dimension)
            y: 2d-array (#functions, #points)
        """
        fpc_scores = self.__Fit_Fpc_Scores(x, y)
        return([fpc_scores, self.mean_fun + np.matmul(fpc_scores,
                            self.eig_fun.reshape(self.num_eig_pairs, -1)).reshape(np.append(-1, self.__grid_shape))])
