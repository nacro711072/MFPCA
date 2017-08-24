# MFPCA
Multi-dimensional Functional Principal Component Analysis

# Requirements:
1. Numpy
2. Scipy
3. Numba
4. Arrayfire  

**Note:** You also need to have the ArrayFire C/C++ library installed on your machine.You can get it from the following sources.  

* [Download and install binaries](https://arrayfire.com/download-splash/?redirect\_to=/download)  

* [Build and install from github](https://github.com/arrayfire/arrayfire)  

# Usage:
假設有N組資料，d維度，$\bg = (g_1, \dots, g_d)'$為估計點在每個維度上的格點數量。  
1.Lpr: 此函式為LLR的實現方法，其帶寬可由CV\_Partition函式決定。

    Lpr(x, y, x0, h, binning = True, bin\_weight = True, ker\_fun = 'Epan', dtype = 'f4')

* Input
    * x : (N * d) 陣列，觀測點。
    * y : 元素個數為N的向量，觀測值。
    * x0 : (g_1 * <span class="math inline">\dots<\span> * g_d, d)的陣列，估計點。
    * h : 元素個數為d的向量，由使用者輸入一代寬，此帶寬為估計時選用的帶寬。若要進行帶寬選擇，可呼叫CV\_Partition函式來選取適當的帶寬。
    * binning : 布林值。選擇在進行LLR估計時，是否將資料合併到格點上，預設為Ture。
    * bin\_weight : 布林值。當資料合併時，是否進行線性合併，預設為Ture。
    * ker\_fun : 字串，預設為'Epan'。LLR估計時選用的核函數，僅提供Epanechnikov及高斯核函數。
        * 'Epan': Epanechnikov核函數，<span class="math inline">$K(x) = \frac{3}{4}(1 - x^2), \; |x| \leq 1$</span>，其餘為0，跟其他核函數相比，理論上此核函數估計最好。
        * 'Gauaaian': 高斯核函數，$K(x) = \frac{1}{\sqrt{2 \pi}} e^{- x^2 / 2}$，當資料數少時建議選用此核函數。
    * dtype: 字串，預設為'f4'。在進行GPU通用運算時，由於雙浮點數計算速度較慢，因此提供此參數讓使用者選用計算時浮點數的精準度。
* Ouput: 一組元素個數為$\prod_{i = 1}^{d} g_i$的向量，依照輸入參數x0格點順序所得的函數點估計值。

