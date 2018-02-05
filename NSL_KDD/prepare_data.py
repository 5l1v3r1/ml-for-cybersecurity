# coding=utf-8
# 将NSL_KDD数据向量化
import csv
import pickle

import numpy as np


# 获取数据集中所有的非数字标签,方面向量化
def get_sym():
    sym = {"protocol_type": [], "service": [], "flag": [], "result": []}
    with open("KDDTrain+.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            if row[1] not in sym["protocol_type"]:
                sym["protocol_type"].append(row[1])
            if row[2] not in sym["service"]:
                sym["service"].append(row[2])
            if row[3] not in sym["flag"]:
                sym["flag"].append(row[3])
            if row[41] not in sym["result"]:
                sym["result"].append(row[41])
    with open("KDDTest+.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            if row[1] not in sym["protocol_type"]:
                sym["protocol_type"].append(row[1])
            if row[2] not in sym["service"]:
                sym["service"].append(row[2])
            if row[3] not in sym["flag"]:
                sym["flag"].append(row[3])
            if row[41] not in sym["result"]:
                sym["result"].append(row[41])
    with open("sym", 'wb') as f:
        pickle.dump(sym, f)


def vector_data():
    Train_list = []
    TrainTFlabel_list = []
    TrainALLlabel_list = []
    Test_list = []
    TestTFlabel_list = []
    TestALLlabel_list = []
    with open("sym", "rb") as f:
        sym = pickle.load(f)
    with open("KDDTrain+.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            row[1] = sym["protocol_type"].index(row[1])
            row[2] = sym["service"].index(row[2])
            row[3] = sym["flag"].index(row[3])
            row[41] = sym["result"].index(row[41])
            Train_list.append(row[0:41])
            TrainALLlabel_list.append(row[41])
            if row[41] == 0:  # 0表示normal,其他表示为攻击流量
                TrainTFlabel_list.append(0)
            else:
                TrainTFlabel_list.append(1)
    with open("KDDTest+.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            row[1] = sym["protocol_type"].index(row[1])
            row[2] = sym["service"].index(row[2])
            row[3] = sym["flag"].index(row[3])
            row[41] = sym["result"].index(row[41])
            Test_list.append(row[0:41])
            TestALLlabel_list.append(row[41])
            if row[41] == 0: # 0表示normal,其他表示为攻击流量
                TestTFlabel_list.append(0)
            else:
                TestTFlabel_list.append(1)
    x_train = np.array(Train_list)
    y_train = np.array(TrainTFlabel_list)
    z_train = np.array(TrainALLlabel_list)
    x_test = np.array(Test_list)
    y_test = np.array(TestTFlabel_list)
    z_test = np.array(TestALLlabel_list)
    np.savez_compressed("./NSL_KDD.npz", x_train=x_train, y_train=y_train, z_train=z_train, x_test=x_test,
                        y_test=y_test, z_test=z_test)

vector_data()
