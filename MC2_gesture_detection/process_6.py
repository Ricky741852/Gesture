from os import listdir,mkdir
from utility import GroundTruth
import numpy as np
import random
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

SAVE_DIR = 'format_data'  # directory of formatted data
TRAIN_DATA_DIR = '200_new_trainData_20231106'  # directory of raw training data
TEST_DATA_DIR = '200_new_testData_20231106'  # directory of raw testing data
# OTHER_DATA_DIR = 'jendy_other_data'  # directory of raw zero-label data

class Gesture_Data():
    def __init__(self,path,windows_size=128,gesture_label=[-1000,-1000,-1000,-1000,-1000]):
        self.path = path
        self.accomplish_path = None # file path of every data
        self.data_classes = list()  # classification list
        self.data_classes_total = 0 # amount of classification
        # self.sigma = gaussian_sigma
        self.windows_size = windows_size
        self.gesture_label = gesture_label


        self.gesture_raw_data = list()

        self.x = None
        self.y = None

        # self.x_train = None
        # self.y_train = None
        
        # self.x_test = None
        # self.y_test = None
    
    def get_accomplish_path_name(self,index):
        if not self.accomplish_path:
            self._get_file()
        return self.accomplish_path[index][0]
    
    def get_accomplish_path_total(self):
        if not self.accomplish_path:
            self._get_file()
        return len(self.accomplish_path)
    
    def _get_file(self):
        """
        
        """
        data_classes = set()
        accomplish_path = []
        path = self.path
        for subdir in sorted(listdir(path)):
            for i in sorted(listdir(path + '/' + subdir)):
                accomplish_path.append((path + '/' + subdir + '/' + i, subdir))
                data_classes.add(subdir)
                # 格式如：('train1&2_test2/testData2/0/02_2019-01-11-07-51-08_C.Y_Chen.txt', '0')
        self.accomplish_path = accomplish_path        
        self.data_classes = sorted(list(data_classes))
        
        # classification 
        self.data_classes_total = len(self.data_classes)

        # detection
        # self.data_classes_total = 1
        # print(self.data_classes)

    def _get_raw_data_from_file(self,path):
        """
        
        """
        data = list()
        # print(path)
        with open(path,"r") as f:
            raw = f.readline()
            while raw:
                data.append( list(map(int,raw.split(","))) ) 
                raw = f.readline()
        # print(data)
        # self.gesture_raw_data.append(data)
        return data


    def _find_gesture_label(self,data):
        """
        
        """
        label = list()
        for i,raw in enumerate(data):
            if raw == self.gesture_label:
                label.append(i)
                del data[i]
        
        if len(label)%2 != 0:
            # raise RuntimeError(f"Total gesture label {self.gesture_label}  is not 2")
            print(f"Total gesture label {self.gesture_label}  is not even")
            raise
        else:
            return label

    def _generate_ground_truth(self,data,label):
        """
        
        """
        ground_truth = [0]*len(data)
        if label is None:
            return 
        
        for i in range(0,len(label),2):
            ground_truth_len = label[i+1]-label[i]
        
            gaussian_ground_truth = GroundTruth(ground_truth_len,ground_truth_len/6)
            # gaussian_ground_truth.plot()
            # print(gaussian_ground_truth.truth)
            # print(len(data))
            # print(len(gaussian_ground_truth.truth))
            
            
            for j, truth in enumerate(gaussian_ground_truth.truth):
                ground_truth[j+label[i]] = truth

        # for i in range(label[0],len(gaussian_ground_truth.truth)):
        #     print(i-label[0])
        #     print(len(gaussian_ground_truth.truth))
        #     ground_truth[i] = gaussian_ground_truth.truth[i-label[0]]

        return ground_truth

    def generate_test_data(self,index):
        if not self.accomplish_path:
            self._get_file()
        raw_data = self._get_raw_data_from_file(self.accomplish_path[index][0])
        data_len = len(raw_data)
        if data_len < self.windows_size:
            print(f"file data {self.accomplish_path[index]} total line smaller than {self.windows_size}")

        label = self._find_gesture_label(raw_data)
        print(label)
        grund_truth = self._generate_ground_truth(raw_data,label)
        # print(grund_truth)

        data_classes = self.data_classes
        class_name = self.accomplish_path[index][1]
        # print(label)
        return raw_data,label,grund_truth,data_classes,class_name,data_len

    def generate_train_data(self):
        """

        """
        # get file path
        self._get_file()

        x = list()
        x_label = list()
        x_path = list()
        y_hm = list()
        y_wh = list()
        data_classes = self.data_classes
        data_classes_total = self.data_classes_total

        test_count = 0

        # for all file
        for j,p in enumerate(self.accomplish_path):
            # get file data
            raw_data = self._get_raw_data_from_file(p[0])
            raw_data_class = p[1]
            
            # muti class 
            data_classes_index = data_classes.index(raw_data_class)
            
            # single class
            # data_classes_index = 0
            
            # print(raw_data_class,data_classes_index)
            print(p)
            label = self._find_gesture_label(raw_data)
            grund_truth = self._generate_ground_truth(raw_data,label)
            # print(raw_data)
            # print(label)
            # print(grund_truth)

            # split data by slide windows 
            data_len = len(raw_data)
            if data_len < self.windows_size:
                # print(f"file data {p} total line smaller than {self.windows_size}")
                continue
                # raise RuntimeError(f"file data {p} total line smaller than {self.windows_size}")
            if label:
                label_len = (label[1]-label[0]) //2
                label_mid_index = (label[1]+label[0]) //2
                
                assert label_len > 0

            for i in range(data_len-self.windows_size+1):
                split_data = raw_data[i:i+self.windows_size]
                split_ground_truth = grund_truth[i:i+self.windows_size]
                if max(split_ground_truth) != 1:
                    split_ground_truth = [0]*self.windows_size
                
                # x.append(normalize(split_data))
                x.append(np.array(split_data)/360)
                x_label.append(raw_data_class)
                x_path.append(p)
                # y.append({'hm': np.array(split_ground_truth) ,'wh':label_len  })
                
                all_classes_y = np.zeros((data_classes_total,self.windows_size), dtype=np.float32)
                all_classes_y[data_classes_index] = np.array(split_ground_truth)

                wh = np.zeros((self.windows_size), dtype=np.float32)
                if label:
                    if abs(label_mid_index-i) >= self.windows_size:
                        # print(p)
                        # print(label_mid_index)
                        pass
                    else:
                        # print(p)
                        # print(label_mid_index,i)
                        wh[label_mid_index-i] = label_len
          

                # if test_count==2007:
                #     print(p)
                #     print(data_classes_index)
                #     print(all_classes_y)
                #     print(all_classes_y.T.tolist())
                #     print(wh)
                #     input()
                test_count += 1
                y_hm.append(all_classes_y.T)
                y_wh.append(np.reshape(wh,(len(wh),1)))


            

                
            
        # format raw data to training data
        
        self.x = {"x":np.array(x),"path":x_path,"label":x_label}
        self.y = {"hm":np.array(y_hm),"wh":np.array(y_wh)}



    

    def plot(self, index,save_path):
        
        """
        
        """

        print(self.x['path'][index])
        data_classes = self.data_classes

        tottal_data_classes = len(self.data_classes)+1
        fig, (ax1, ax2) = plt.subplots(2,1)
        ax1.set_title(f"class type {self.x['label'][index]}")
        ax1.plot(self.x['x'][index])
        
        print(data_classes)
        # print(self.y['hm'][index].tolist())
        # for i,hm in enumerate(self.y['hm'][index].T):
        #     # print(i)
        #     ax2.plot(hm,label=data_classes[i])
        #     # print(data_classes[i])
        #     # print(hm)
        #     mid,w =  np.argmax(hm),np.max(self.y['wh'][index])
        #     if mid != 0 and mid <= self.windows_size:
        #         print(f"class type {data_classes[i]}")
        #         # print(self.y['wh'][index])
        #         # print(self.y['wh'][index].shape)
        for i, hm in enumerate(self.y['hm'][index].T):
            max_val = np.max(self.y['hm'][index])
            # 在達到最大值後，將後續的數據點都設置為max_val。每當hm的某個數據點達到max_val時，
            # 所有後續的數據點都將被設置為max_val，使圖形在達到最大值後一直保持在那個水平。
            reached_max = False # 用來判斷是否為最大值
            for j, value in enumerate(hm):  # 這是一個內部循環，用於遍歷當前的hm數據。每次循環會得到一個value及其j
                # 如果當前的value等於max_val，那麼將reached_max設置為True，表示已經達到最大值。
                if value == max_val:   
                    reached_max = True 
                # 如果已經達到最大值（reached_max為True），那麼將當前和後續的hm數據點都設置為max_val。
                if reached_max:
                    hm[j] = max_val
            
            ax1.plot(label=data_classes[i])
            ax2.plot(hm, label=f"class {data_classes[i]}")
            
            # print(self.y['hm'][index])
            # min_val = np.min(self.y['hm'][index])
            # print('max_val = ',max_val)
            
            # print(min_val)
            # 以下兩行，這裡看分數的方式改為看最後一個sample的分數，而不是以中心點看，所以這裡需要找出最後一個sample的最高點
            max_point, w= np.argmax(hm),np.max(self.y['wh'][index])
            
            mid = (max_point - w) # 計算出中心點的位置，所以用終點-w(寬度)
            # print('max_point = ',max_point)
            # print(w)
           
            # max_point = max_point/np.max(max_point)
            if mid >= 0 and mid <= self.windows_size:# 確保中心點在位置在此範圍內
                print(f"class type {data_classes[i]}")
                    # print(self.y['wh'][index])
                    # print(self.y['wh'][index].shape)
                    # ax1 = 起始點,ax2 = 終點

                ax2.set_title(f"Ground truth w={w}")
                ax2.axvline(mid,linestyle='--')
                ax2.axvline(mid+w,linestyle='--')
                ax2.axvline(mid-w,linestyle='--')

                
                ax1.axvline(mid,linestyle='--')
                ax1.axvline(mid+w,linestyle='--')
                ax1.axvline(mid-w,linestyle='--')
                

        
        # print(self.y['hm'][index])
        ax2.legend(bbox_to_anchor=(1.0, 1.0, 0.3, 0.2),loc = 'upper right')
        plt.tight_layout()
        plt.show()
        plt.savefig('{}/{}.png'.format(save_path,f"class_type_{self.x['label'][index]}_{index}"))
    

    def analysis(self):
        p,n = [0]*len(self.data_classes), [0]*len(self.data_classes)
        for x,y in zip(self.x['path'],self.y['hm']):
            if np.max(y)==1:
                p[int(x[1])] += 1
            else:
                n[int(x[1])] += 1

        # print(p)
        # print(n)





if __name__ == "__main__":
    # G = Gesture_Data(r"./200_new_trainData")

    # G.generate_train_data()
    G = Gesture_Data(r"./200_new_testData_20231106")

    G.generate_test_data(index = 10)
    plt.show()
    #G.analysis()

    # print(G.x['x'].shape)
    # save_path = "./ground_truth_plot/{}".format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    # mkdir(save_path)
    # input_num = int(input("num (-1 for exit) = "))
    # if input_num != -1 :
    #     totaldata = len(G.x['x'])
    #     # print(G.x)
    #     for i in range(input_num):
    #         rand_index = random.randint(0,totaldata-1)
    #         G.plot(save_path=save_path,index=rand_index)

