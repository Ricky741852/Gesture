import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import scipy.integrate

class GroundTruth:
    def __init__(self, data_len, Sigma):
        self.data_len = data_len# 對應主程式中的L    
        self.Sigma = Sigma  # 對應主程式中的L/6
        self.x = np.arange(0, self.data_len)
        self.truth = self.generate_ground_truth()

    def _pdf_truth(self, sigma=1):  # PDF(Probability Density Function, PDF)機率密度函數，紀錄每個值所占的比例
        """
        Object as point formula 1D
        """
        m = self.data_len // 2
        # print(m)
        x = np.arange(-m, m)
        # print(len(x))
        h = np.exp(-(x * x) / (2 * sigma * sigma))  # 因為是常態分佈，所以均值u為0，sigma=1

        h = [i if i >= 0 else 0 for i in h]

        return h

    def _cdf_truth(self):   # CDF(Cumulative Distribution Function CDF)累積分佈函數，把PDF累加起來的結果

        # 使用scipy.stats.norm.cdf函式，會把帶入的x數值，先做pdf的計算，然後再把pdf算出來的數值拿去做積分
        x = np.linspace(-self.data_len//2, self.data_len//2,self.data_len)
        cdf = scipy.stats.norm.cdf(x, 0, self.Sigma)
        cdf /= cdf[-1] # 確保做正歸化會在0～1之間

        return cdf

    def generate_ground_truth(self):
        """Generate gaussian ground_truth"""

        # truth = self._cdf_truth()
        truth = self._pdf_truth(self.Sigma)
        return truth

    def plot(self):
        plt.plot(self.truth)
        plt.show()


if __name__ == "__main__":
    L = 50
    G = GroundTruth(L, L / 4)
    G.plot()
