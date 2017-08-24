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
1. Lpr: 此函式為LLR的實現方法，其帶寬可由CV\_Partition函式決定。  

        Lpr(x, y, x0, h, binning = True, bin_weight = True, ker_fun = 'Epan', dtype = 'f4')


    * 參數輸入:
        * x : (N * d) 陣列，觀測點。
        * y : 元素個數為N的向量，觀測值。
        * x0 : (g_1 * <span class="math inline">\dots</span> * g_d, d)的陣列，估計點。
        * h : 元素個數為d的向量，由使用者輸入一代寬，此帶寬為估計時選用的帶寬。若要進行帶寬選擇，可呼叫CV\_Partition函式來選取適當的帶寬。
        * binning : 布林值。選擇在進行LLR估計時，是否將資料合併到格點上，預設為Ture。
        * bin\_weight : 布林值。當資料合併時，是否進行線性合併，預設為Ture。
        * ker\_fun : 字串，預設為'Epan'。LLR估計時選用的核函數，僅提供Epanechnikov及高斯核函數。
            * 'Epan': Epanechnikov核函數，<span class="math inline">$K(x) = \frac{3}{4}(1 - x^2), \; |x| \leq 1$</span>，其餘為0，
               跟其他核函數相比，理論上此核函數估計最好。
            * 'Gaussian': 高斯核函數，<span class="math inline">$K(x) = \frac{1}{\sqrt{2 \pi}} e^{- x^2 / 2}$</span>，
               當資料數少時建議選用此核函數。
        * dtype: 字串，預設為'f4'。在進行GPU通用運算時，由於雙浮點數計算速度較慢，因此提供此參數讓使用者選用計算時浮點數的精準度。
    * 參數輸出: 一組元素個數為<span class="math inline">$\prod_{i = 1}^{d} g_i$</span>的向量，依照輸入參數x0格點順序所得的函數點估計值。

2. CV\_Partition: 此函式為選取帶寬的實現方法，用於LLR估計。

        CV_Partition(x, y, x0, h, n_train = None, binning = True, bin_weight = True, ker_fun = 'Epan')
    * 參數輸入:
        * x : (N * d) 陣列，觀測點。
        * y : 元素個數為N的向量，觀測值。
        * x0 : $(g_1 * \dots * g_d, d)$的陣列，估計點。
        * h : (n * d)陣列，n為使用者樹入選取代寬的個數。
        * binning : 布林值。選擇在進行LLR估計時，是否將資料合併到格點上，預設為True。
        * bin\_weight : 布林值。當資料合併時，是否進行線性合併，預設為True。
        * ker\_fun : 字串，預設為'Epan'。LLR估計時選用的核函數，僅提供Epanechnikov及高斯核函數。
            * 'Epan': Epanechnikov核函數，$K(x) = \frac{3}{4}(1 - x^2), \; |x| \leq 1$，其餘為0，理論上為最佳的核函數。
            * 'Gaussian': 高斯核函數，$K(x) = \frac{1}{\sqrt{2 \pi}} e^{- x^2 / 2}$，當資料數少時建議選用此核函數。
    * 輸出: 元素個數為d的向量，由輸入參數h裡所選取出使得均方誤差最小的帶寬。

3. Fpca:假設有N組觀測函數，每組觀測函數上有$N_i$個點，維度為d維，$\bg = (g_1, \dots, g_d)'$為估計點在每個維度上的格點數量。

        Fpca(x, y, x0, h_mean, h_cov, h_cov_dia, fve = 0.85, binning = True, bin_weight = True, 
             ker_fun = 'Epan', bw_select = 'Partition', dtype = 'f4')

    * 參數輸入:
        * x : 元素個數為N的list，list裡面為($N_i$ * d)陣列，觀測點。
        * y : 元素個數為N的list，list裡面為$N_i$陣列，觀測值。
        * x0: $(g_1 * \dots * g_d, d)$的陣列，估計點。
        * h\_mean : (n * d)陣列，估計平均函數的候選帶寬。
        * h\_cov : (n * 2d)陣列，估計共變異數函數的候選帶寬。
        * h\_cov\_dia : (n * d)陣列，估計誤差的變異數的候選帶寬。
        * fve : 0 到 1 之間的浮點數，在選取前$K$組特徵對時的準則，預設為0.85。
        * binning: 布林值。選擇在進行LLR估計時，是否將資料合併到格點上，預設為True。
        * bin\_weight: 布林值。當資料合併時，是否進行線性合併，預設為True。
        * ker\_fun: 字串，預設為'Epan'。LLR估計時選用的核函數，僅提供Epanechnikov及高斯核函數。
            * 'Epan': Epanechnikov核函數，$K(x) = \frac{3}{4}(1 - x^2), \; |x| \leq 1$，其餘為0，跟其他核函數相比，理論上此核函數估計最好。
            * 'Gaussian': 高斯核函數，$K(x) = \frac{1}{\sqrt{2 \pi}} e^{- x^2 / 2}$，當資料數少時建議選用此核函數。
        * bw\_select: 字串，預設為'Partition'。LLR選擇帶寬的準則。
            * 'Partition': 將資料切分成訓練集和驗證集，由驗證集估算均方誤差。
            * 'LeaveOneOut': 使用留一曲線交叉驗證法，只隨機抽出100條曲線做驗證，若觀測函數小於100時則全選取。
        * dtype: 字串，預設為'f4'。在進行GPU通用運算時，由於雙浮點數計算速度較慢，因此提供此參數讓使用者選用計算時浮點數的精準度。
            * 'f4':單精度浮點數。
            * 'f8':雙精度浮點數。
    * 輸出會產生fpca物件，其物件的成員變數有:
        * mean\_fun: $(g_1 * \dots * g_d * d)$陣列，由LLR估計的平均函數。
        * cov\_fun: $(g_1 * \dots * g_d * g_1 * \dots * g_d * 2d)$陣列，由LLR估計的共變異數函數。
        * cov\_dia: $(g_1 * \dots * g_d * d)$陣列，由LLR估計在共變異數函數對腳線上的曲線。
        * num\_eig\_pairs: 正整數，由FVE選取的前$K$組特徵對。
        * eig\_fun: $(K * g_1 * \dots * g_d * d)$陣列，$K = \mbox{num\_eig\_pairs}$，經由變異數函數得到的前$K$組特徵函數。
        * fpc\_scores: $(N * K)$陣列，$K = \mbox{num\_eig\_pairs}$，將$X(\bt)$函數投影在特徵函數上的主成份分數。
        * sigma2: 浮點數，為誤差變異數的估計值。
        * mean\_bw: d維向量，估計平均函數時選用的帶寬。
        * cov\_bw: 2 * d維向量，估計共變異函數時選用的帶寬。
        * cov\_dia\_bw: d維向量，估計誤差變異數時選用的帶寬。
* 成員函式: 此函式主要目的在於重建$X(\bt)$函數。由使用者輸入資料點與觀測值，藉由PACE估算主成份分數，再和已估算好的平均函數和特徵函數，將$X(\bt)$曲線重現。

        Restruct_Fun(x, y)
    
    * 輸入參數:
        * x : 元素個數為newN的list，list裡面為($N_i$ * d)陣列，資料點，newN為使用者輸入進來的樣本函數個數。
        * y : 元素個數為N的list，list裡面為$N_i$陣列，資料觀測值。
    * 輸出一個list，依照順序為:
        * fpc\_scores: (N * K)陣列。將$X(\bt)$中心化後，投影在特徵函數上的主成份分數。
        * restruct\_fun: $(N * g_1 * \dots * g_d, d)$陣列。重現在格點上的$X(\bt)$函數。
