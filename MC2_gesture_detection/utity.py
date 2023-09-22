import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import math



def cal_IoU(gTrue,gPred):
    # print(gTrue,gPred)
    if gPred[1] < gTrue[0] or gPred[0] > gTrue[1]:
        IoU = 0
        return IoU

    union = max(gTrue[1],gPred[1]) - min(gTrue[0],gPred[0])
    overlap = min(gTrue[1],gPred[1]) - max(gTrue[0],gPred[0])

    IOU = overlap / union
    return  round(IOU,2)

class Ground_truth():
    def __init__(self,data_len,Sigma):
        self.data_len = data_len
        self.Sigma = Sigma
        self.x = np.arange(0,self.data_len)
        self.truth = self.generate_ground_truth()



    def _cdf_truth(self):
        return scipy.stats.norm.cdf(self.x,self.data_len/2,self.Sigma)

    def _pdf_truth(self,sigma=1):
        """
        Object as point formula 1D
        """
        m = self.data_len // 2
        # print(m)
        x = np.arange(-m,m)
        # print(len(x))
        h = np.exp(-(x*x) / (2*sigma*sigma) )

        h = [i if i>=0 else 0 for i in h]

        return h



    def generate_ground_truth(self):
        '''
        generate  gaussian ground_truth
        '''
        # truth = self._cdf_truth()
        truth = self._pdf_truth(self.Sigma)
        return truth 

    def plot(self):
        # plot the cdf
        # sns.lineplot(x=self.x*2, y=self.norm_cdf)
        
        plt.plot(self.truth)
        print(self.truth)
        plt.show()

if __name__ == "__main__":
    L = 70
    G = Ground_truth(L,L/6)
    # G.generate_ground_truth()
    G.plot()




