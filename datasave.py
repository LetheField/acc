import os
import scipy
import scipy.io as sio

if __name__ == "__main__":
    time = []
    time_err = []
    thre = []
    thre_err = []
    for i in range(40):
        idx = i+1
        with open('saveproto/thredata{}.txt'.format(idx)) as f:
            temp = []
            for line in f:
                temp.append(float(line))
            time.append(temp[0])
            time_err.append(temp[1])
            thre.append(temp[2])
            thre_err.append(temp[3])
    data = {}
    data['time1'] = time
    data['time_err1'] = time_err
    data['thre1'] = thre
    data['thre_err1'] = thre_err
    sio.savemat('dataproto.mat', data)
    