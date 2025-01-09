import numpy as np
import os
from scipy.io import loadmat,savemat
os.chdir('G:/SEED_dataset/SEED/ExtractedFeatures')
label_data = loadmat('label.mat')['label'][0,]
folder_list = [f for f in os.listdir('.') if f.endswith('.mat')]

# 第i个人的3个session
for i in range(1,16):
    label = label_data[i-1] + 1
    file_name_session_1 = folder_list[(i * 3 - 3)]
    file_name_session_2 = folder_list[(i * 3 - 2)]
    file_name_session_3 = folder_list[(i * 3 - 1)]

    print(file_name_session_1)
    print(file_name_session_2)
    print(file_name_session_3,'\n')

    # 每个人在不同时间的数据
    data_session_1 = loadmat(file_name_session_1)
    data_session_2 = loadmat(file_name_session_2)
    data_session_3 = loadmat(file_name_session_3)
    
    feature_session_1 = []
    feature_session_2 = []
    feature_session_3 = []

    for j in range(1,16):
        feature_path = 'de_LDS' + str(j)
        n = data_session_1[feature_path].shape[1]
        feature_data1 = data_session_1[feature_path].transpose(1,0,2).reshape(n,310)
        feature_data2 = data_session_2[feature_path].transpose(1,0,2).reshape(n,310)
        feature_data3 = data_session_3[feature_path].transpose(1,0,2).reshape(n,310)

        feature_session_1.extend(feature_data1)
        feature_session_2.extend(feature_data2)
        feature_session_3.extend(feature_data3)

    feature_session_1 = np.array(feature_session_1)
    feature_session_2 = np.array(feature_session_2)
    feature_session_3 = np.array(feature_session_3)

    # label_vector = np.full(feature_session_1.shape[0],label)
    label_vector = np.full((feature_session_1.shape[0],), label)
    out_data1 = {'feature':feature_session_1,'label':label_vector}
    out_data2 = {'feature':feature_session_2,'label':label_vector}
    out_data3 = {'feature':feature_session_3,'label':label_vector}

    file_name = str(i) + '.mat'
    file_full_path = "G:/研一课程作业/计算机前沿技术/semi_supervised/de_feature_SEED/feature_for_net_session1_LDS_de/"+file_name
    savemat(file_full_path,{'dataset':out_data1})
    file_full_path = "G:/研一课程作业/计算机前沿技术/semi_supervised/de_feature_SEED/feature_for_net_session2_LDS_de/"+file_name
    savemat(file_full_path,{'dataset':out_data2})
    file_full_path = "G:/研一课程作业/计算机前沿技术/semi_supervised/de_feature_SEED/feature_for_net_session3_LDS_de/"+file_name
    savemat(file_full_path,{'dataset':out_data3})
