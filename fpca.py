import lpr

###
import numpy as np
import scipy as sp
from scipy import interpolate
import time

###

class Fpca(object):
    def __init__(self, x, y, x0, h_mean, h_cov, h_cov_dia, fve = 0.85,
                 binning = True, bin_weight = True, ker_fun = 'Epan', bw_select = 'Partition'):
        self.__Check_Input(x, y, x0, h_mean, h_cov, h_cov_dia, fve, binning, bin_weight, ker_fun)
        self.__num_fun, self.__num_pt, self.__d = x.shape
        self.__grid_shape = np.asarray(x0.shape[:-1])
        self.__bin_width = np.ptp(x0.reshape(-1, self.__d), 0) / (self.__grid_shape - 1)
        self.__n_train = lpr.Partition_Data_Size(self.__num_fun * self.__num_pt - np.isnan(y).sum())
        x0_min = x0.reshape(-1, self.__d).min(0)
        x0_max = x0.reshape(-1, self.__d).max(0)
        self.__grid = tuple(np.linspace(x0_min.take(i), x0_max.take(i), x0.shape[i]) for i in range(self.__d))
        self.__Main(x, y, x0, h_mean, h_cov, h_cov_dia, fve, binning, bin_weight, ker_fun, bw_select)
        self.__Standardize_Output()

    def __str__(self):
        print("Number of grid: %s" %" * ".join(map(str, self.__grid_shape)))
        print("Number of random function: %d" %self.__num_fun)
        # print("Number of points for each random function: %d" %self.__num_pt)
        print("="*20)
        print("Eigen pairs: %d" %self.num_eig_pairs)
        print("Sigma2: %f" %self.sigma2)
        print("Bandwidth of mean: [%s]" %", ".join(map(str, self.mean_bw)))
        print("Bandwidth of cov: [%s]" %", ".join(map(str, self.cov_bw)))
        print("Bandwidth of cov_diag: [%s]" %", ".join(map(str, self.cov_diag_bw)))
        return("="*20)

    def __Check_Input(self, x, y, x0, h_mean, h_cov, h_cov_dia, fve, binning, bin_weight, ker_fun):
        if len(x.shape) is not 3:
            raise ValueError("Input x must be 3-d.(functions, points, dimension)")
        if len(y.shape) is not 2:
            raise ValueError("Input y must be 2-d.(functions, points)")
        if len(x0.shape[:-1]) is not x0.shape[-1]:
            raise ValueError("Input x0 dimension error.")
        if len(h_mean.shape) is not 2:
            raise ValueError("Input h_mean must be 2-d.")
        if len(h_cov.shape) is not 2:
            raise ValueError("Input h_mean must be 2-d.")
        if len(h_cov_dia.shape) is not 2:
            raise ValueError("Input h_mean must be 2-d.")
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

    def __Cv_Leave_One_Curve(self, x, y, x0, candidate_h, ker_fun):
        """
        input variable:
            x : matrix with dimension (num_of_points, dim)
            y : vecor with length num_of_points
            x0 : narray
            h : vector
            bin_width : vector
        """
        n_grid = np.array(x0.shape[:-1])
        d = x0.shape[-1]
        no_nan_val = ~(np.isnan(y) | np.isnan(x).any(2))
        bin_data_y, bin_data_num = lpr.Bin_Data(np.compress(no_nan_val.reshape(-1), x.reshape(-1, d), 0), np.compress(no_nan_val.reshape(-1), y.reshape(-1)), x0)
        rand_num_fun = np.random.choice(self.__num_fun, 100, replace = False)
        test_fun = x.take(rand_num_fun, 0)
        test_y = y.take(rand_num_fun, 0)
        rand_num_no_nan_val = no_nan_val.take(rand_num_fun, 0)
        sse = np.ones(candidate_h.shape[0])
        for i in range(candidate_h.shape[0]):
            h = candidate_h.take(i, 0)
            r = lpr.Get_Range(self.__bin_width, h, ker_fun)
            delta_x = lpr.Get_Delta_x(self.__bin_width, r)
            weight = lpr.Get_Weight(delta_x, h, ker_fun)
            big_x = np.hstack((np.ones((delta_x.shape[0], 1)), delta_x))
            h_too_small = False
            for fun_i in range(test_fun.shape[0]):
                test_funi = test_fun.take(fun_i, 0)
                test_yi = test_y.take(fun_i, 0)
                fun_no_nan_value = rand_num_no_nan_val.take(fun_i, 0)
                fun_biny, fun_binx = lpr.Bin_Data(np.compress(fun_no_nan_value, test_funi, 0),
                                                  np.compress(fun_no_nan_value, test_yi), x0)
                biny = bin_data_y - fun_biny
                binx = bin_data_num - fun_binx
                ext_biny, ext_binx = lpr.Extend_Bin_Data([biny, binx], r)
                train_y = lpr.Get_Linear_Solve(big_x.T, weight, ext_binx, ext_biny, r)
                if np.any(np.isnan(train_y)):
                    h_too_small = True
                    break
                interp_fun = sp.interpolate.RegularGridInterpolator(self.__grid, train_y.reshape(x0.shape[:-1]))
                interp_y = interp_fun(test_funi)
                sse[i] += np.nansum((test_yi - interp_y)**2)
            if h_too_small:
                sse[i] = np.nan
        opt_h = candidate_h.take(np.nanargmin(sse), 0)
        return(opt_h)

    def __CV_Cov_Partition(self, x, y, x0, h, ker_fun):
        grid_shape, d = np.asarray(x0.shape[:-1]), x0.shape[-1]
        n_grid = np.prod(self.__grid_shape)
        x_displacement = ((x - x0.reshape(-1, d).min(axis = 0)) / self.__bin_width + np.ones(self.__d) / 2).astype(np.int32)
        x_p = np.sum(x_displacement * np.append(grid_shape[::-1].cumprod()[-2::-1], 1), axis = 2)
        xx_p = (x_p.repeat(x_p.shape[1], 1) * n_grid + np.tile(x_p, x_p.shape[1]))
        xx_p = np.delete(xx_p, np.arange(0, self.__num_pt**2, self.__num_pt + 1), 1)
        yy = np.einsum('ij,ik->ijk', y, y).reshape(self.__num_fun, -1)
        yy = np.delete(yy, np.arange(0, self.__num_pt**2, self.__num_pt + 1), 1)
        non_nan_value = ~np.isnan(yy).reshape(-1)
        n_real_val = non_nan_value.sum()
        xx_p = np.compress(non_nan_value, xx_p.reshape(-1))
        yy = np.compress(non_nan_value, yy.reshape(-1))
        n_train = lpr.Partition_Data_Size(n_real_val)
        random_order = np.random.permutation(n_real_val)
        train_xx, test_xx = np.split(xx_p.reshape(-1).take(random_order), [n_train])
        train_yy, test_yy = np.split(yy.reshape(-1).take(random_order), [n_train])
        bin_xx = np.bincount(train_xx, minlength = n_grid**2)
        bin_yy = np.bincount(train_xx, train_yy, minlength = n_grid**2)
        bin_xx = bin_xx.reshape(np.tile(self.__grid_shape, 2))
        bin_yy = bin_yy.reshape(np.tile(self.__grid_shape, 2))
        ssq = np.zeros(h.shape[0])
        for i in range(h.shape[0]):
            fit_y = lpr.Lpr_For_Bin([bin_yy, bin_xx],
                                    np.tile(self.__bin_width, 2),
                                    h.take(i, 0),
                                    ker_fun = ker_fun)
            ssq[i] = ((test_yy - fit_y[test_xx])**2).sum()
            if np.isnan(fit_y).any():
                ssq[i] = np.nan
        h_opt = h.take(np.nanargmin(ssq), 0)
        return([h_opt, xx_p, yy])

    def __CV_Cov_Leave_One_Out(self, x, y, x0, candidate_h, ker_fun):
        grid_shape, d = np.asarray(x0.shape[:-1]), x0.shape[-1]
        n_grid = np.prod(self.__grid_shape)
        x_displacement = np.rint((x - x0.reshape(-1, d).min(axis = 0)) / self.__bin_width).astype(np.int32)
        x_p = np.sum(x_displacement * np.append(grid_shape[::-1].cumprod()[-2::-1], 1), axis = 2)
        xx_p = (x_p.repeat(x_p.shape[1], 1) * n_grid + np.tile(x_p, x_p.shape[1])).reshape(-1)
        yy = np.einsum('ij,ik->ijk', y, y).reshape(-1)
        random_order = np.random.choice(self.__num_fun, 100 if self.__num_fun > 100 else self.__num_fun, replace=False)
        test_fun = x.take(random_order, 0)
        test_xx_p = xx_p.reshape(self.__num_fun, -1).take(random_order, 0)
        test_yy = yy.reshape(self.__num_fun, -1).take(random_order, 0)
        non_nan_value = ~np.isnan(yy)
        xx_p = np.compress(non_nan_value, xx_p)
        yy = np.compress(non_nan_value, yy)
        tot_binx = np.bincount(xx_p, minlength = n_grid**2)
        tot_biny = np.bincount(xx_p, yy, minlength = n_grid**2)
        sse = np.ones(candidate_h.shape[0])
        for i in range(candidate_h.shape[0]):
            h = candidate_h.take(i, 0)
            r = lpr.Get_Range(self.__bin_width, h, ker_fun)
            delta_x = lpr.Get_Delta_x(self.__bin_width, r)
            weight = lpr.Get_Weight(delta_x, h, ker_fun)
            big_x = np.hstack((np.ones((delta_x.shape[0], 1)), delta_x))
            h_too_small = False
            for fun_i in range(test_xx_p.shape[0]):
                test_funi = test_fun.take(fun_i, 0)
                fun_non_nan_value = non_nan_value.reshape(self.__num_fun, -1).take(random_order.take(fun_i), 0)
                fun_xx = np.compress(fun_non_nan_value, test_xx_p.take(fun_i, 0))
                fun_yy = np.compress(fun_non_nan_value, test_yy.take(fun_i, 0))
                fun_binx = np.bincount(fun_xx, minlength = n_grid**2)
                fun_biny = np.bincount(fun_xx, fun_yy, minlength = n_grid**2)
                binx = tot_binx - fun_binx
                biny = tot_biny - fun_biny
                ext_biny, ext_binx = lpr.Extend_Bin_Data([biny.reshape(n_grid, n_grid), binx.reshape(n_grid, n_grid)], r)
                train_yy = lpr.Get_Linear_Solve(big_x.T, weight, ext_binx, ext_biny, r)
                if np.any(np.isnan(train_yy)):
                    h_too_small = True
                    break
                interp_fun = sp.interpolate.RegularGridInterpolator(self.__grid * 2, train_yy.reshape(x0.shape[:-1] * 2))
                interp_y = interp_fun(np.hstack((test_funi.repeat(self.__num_pt, 0),
                                      np.tile(test_funi.reshape(-1), self.__num_pt).reshape(-1, self.__d))))
                sse[i] += np.nansum((test_yy.take(fun_i, 0) - interp_y)**2)
            if h_too_small:
                sse[i] = np.nan
        h_opt = candidate_h.take(np.nanargmin(sse), 0)
        return([h_opt, xx_p, yy])

    def __Fit_Mean(self, x, y, x0, candidate_h, binning, bin_weight, ker_fun, bw_select):
        if bw_select is 'Partition':
            self.mean_bw = lpr.CV_Partition(x.reshape(-1, self.__d), y.reshape(-1), x0, candidate_h,
                                        self.__n_train, binning, bin_weight, ker_fun)
        elif bw_select is 'LeaveOneOut':
            self.mean_bw = self.__Cv_Leave_One_Curve(x, y, x0, candidate_h, ker_fun)

        self.mean_fun = lpr.Lpr(x = x.reshape(-1, self.__d),
                                y = y.reshape(-1),
                               x0 = x0,
                                h = self.mean_bw,
                                binning = binning,
                                bin_weight = bin_weight,
                                ker_fun = ker_fun)

    def __Fit_Cov(self, x, y, x0, candidate_h, binning, ker_fun, bw_select):
        if bw_select is 'Partition':
            self.cov_bw, xx_p, yy = self.__CV_Cov_Partition(x, y, x0, candidate_h, ker_fun)
        elif bw_select is 'LeaveOneOut':
            self.cov_bw, xx_p, yy = self.__CV_Cov_Leave_One_Out(x, y, x0, candidate_h, ker_fun)
        bin_xx = np.bincount(xx_p.reshape(-1), minlength = np.prod(self.__grid_shape)**2)
        bin_yy = np.bincount(xx_p.reshape(-1), yy.reshape(-1), minlength = np.prod(self.__grid_shape)**2)
        fit_yy = lpr.Lpr_For_Bin([bin_yy.reshape(np.tile(self.__grid_shape, 2)), bin_xx.reshape(np.tile(self.__grid_shape, 2))],
                                 np.tile(self.__bin_width, 2),
                                 h = self.cov_bw,
                                 ker_fun = ker_fun)
        self.cov_fun = fit_yy.reshape(np.repeat(np.prod(self.__grid_shape), 2)) - np.outer(self.mean_fun, self.mean_fun)

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

    def __Fit_Sigma2(self, x, y, x0, candidate_h, binning, bin_weight, ker_fun, bw_select):
        if bw_select is 'Partition':
            self.cov_diag_bw = lpr.CV_Partition(x.reshape(-1, self.__d),
                                                y.reshape(-1),
                                                x0,
                                                candidate_h,
                                                self.__n_train,
                                                binning,
                                                bin_weight,
                                                ker_fun)
        elif bw_select is 'LeaveOneOut':
            self.cov_diag_bw = self.__Cv_Leave_One_Curve(x, y, x0, candidate_h, ker_fun)

        self.cov_dia = lpr.Lpr(x = x.reshape(-1, self.__d),
                               y = y.reshape(-1),
                               x0 = x0,
                               h = self.cov_diag_bw,
                               binning = binning,
                               bin_weight = bin_weight,
                               ker_fun = ker_fun) - self.mean_fun**2

        interval = np.array([int(np.prod(self.__grid_shape) / 4), int(3 * np.prod(self.__grid_shape) / 4)])
        self.sigma2 = (self.cov_dia - np.diag(self.cov_fun)).take(interval).mean() * np.prod(self.__bin_width)

    def __Fit_Fpc_Scores(self, x, y):
        n = y.shape[0]
        non_nan_value = ~np.isnan(y)
        mean_inter_fun = sp.interpolate.RegularGridInterpolator(self.__grid, self.mean_fun.reshape(self.__grid_shape))
        phi_inter_fun = np.asarray([sp.interpolate.RegularGridInterpolator(self.__grid,
                                    self.eig_fun.take(i, 0).reshape(self.__grid_shape))(x)
                                    for i in range(self.num_eig_pairs)])
        y_center = y - mean_inter_fun(x)
        fit_pc_score = np.zeros((self.__num_fun, self.num_eig_pairs))
        for i in range(n):
            non_nan_value_i = non_nan_value.take(i, 0)
            phi_i = np.compress(non_nan_value_i, phi_inter_fun.take(i, 1), 1) #(eig_fun, pts)
            y_i = np.compress(non_nan_value_i, y_center.take(i, 0))
            fit_cov_inv = np.linalg.inv(np.matmul(phi_i.T * self.eig_val, phi_i) + np.identity(y_i.size) * self.sigma2)
            fit_pc_score[i] = self.eig_val * np.matmul(np.matmul(phi_i, fit_cov_inv), y_i)
        return(fit_pc_score)

    def __Standardize_Output(self):
        self.mean_fun = self.mean_fun.reshape(self.__grid_shape)
        self.cov_fun = self.cov_fun.reshape(np.tile(self.__grid_shape, 2))
        self.eig_fun = self.eig_fun.reshape(np.append(-1, self.__grid_shape))
        self.cov_dia = self.cov_dia.reshape(self.__grid_shape)

    def __Main(self, x, y, x0, h_mean, h_cov, h_cov_dia, fve, binning, bin_weight, ker_fun, bw_select):
        """
        input:
            x: ndarray (#functions, #points, #dimension)
            y: ndarray (#functions, #points)
            x0: ndarray
            candidate_h: ndarray (#h, #dimension)
            num_grid: vector
        """
        ###
        self.__Fit_Mean(x, y, x0, h_mean, binning, bin_weight, ker_fun, bw_select)
        ###
        self.__Fit_Cov(x, y, x0, h_cov, binning, ker_fun, bw_select)
        ###
        self.__Fit_EigPairs(fve)
        ###
        self.__Fit_Sigma2(x, y**2, x0, h_cov_dia, binning, bin_weight, ker_fun, bw_select)
        ###PACE
        self.fpc_scores = self.__Fit_Fpc_Scores(x, y)

    def Restruect_Fun(self, x, y):
        """
        input:
            x: 3d-array (#functions, #points, #dimension)
            y: 2d-array (#functions, #points)
        """
        fpc_scores = self.__Fit_Fpc_Scores(x, y)
        return([fpc_scores, self.mean_fun + np.matmul(fpc_scores,
                            self.eig_fun.reshape(self.num_eig_pairs, -1)).reshape(np.append(-1, self.__grid_shape))])
