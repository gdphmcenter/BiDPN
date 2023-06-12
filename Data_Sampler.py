
import pandas as pd
import numpy as np
import os
import glob
import random
import shutil
import torch
from scipy.io import loadmat
from torchvision import transforms
from PIL import Image
from sklearn import preprocessing
def check_dataset_processed(dataset_fewshot_path):
    return os.path.exists(dataset_fewshot_path)

def Insure_data_image(dataset_raw_path):
    raw_mat_number = len(glob.glob(dataset_raw_path+'/*.mat'))
    raw_img_number = len(glob.glob(dataset_raw_path+'/*.jpg'))
    filename_list = os.listdir(dataset_raw_path)
    matFile_list = []
    imgFile_list = []
    for filename in filename_list:
        if filename.endswith('.mat'):
            matFile_list.append(filename)
        elif filename.endswith('.jpg'):
            imgFile_list.append(filename)
    matFile_list.sort(key=lambda x: (x.split('----')[0], x.split('----')[-1].split('.')[0]))
    imgFile_list.sort(key=lambda x: (x.split('----')[0], x.split('----')[-1].split('.')[0]))
    if raw_img_number == raw_mat_number:
        acc_list = []
        for mat, img in zip(matFile_list, imgFile_list):
            if mat.split('.')[0] == img.split('.')[0]:
                acc_list.append(1)
            elif mat.split('.')[0] != img.split('.')[0]:
                acc_list.append(0)
        if 0 in acc_list:
            print('Data can not match to its correspond image, Please try again!')
            return 0
        else:
            return 1
    else:
        return 0

def transfer_data_from_raw(dataset_raw_path, dataset_fewshot_path):
    if not Insure_data_image(dataset_raw_path=dataset_raw_path):
        print('Exists error, Please check the code!!!')
    filename_list = os.listdir(dataset_raw_path)
    classes_list = []
    for filename in filename_list:
        classes_item = filename.split('----')[0]
        classes_list.append(classes_item)
    classes_list = np.unique(classes_list).tolist()
    random.shuffle(classes_list)
    if len(classes_list) == 64:
        train_classes = classes_list[:44]
        val_classes = classes_list[44:50]
        test_classes = classes_list[50:]
    elif len(classes_list) == 116:
        train_classes = classes_list[:76]
        val_classes = classes_list[76:88]
        test_classes = classes_list[88:]
    train_list = []
    for train_it in train_classes:
        for k in range(1, 201):
            train_list.append(train_it+'----'+str(k))
    val_list = []
    for val_it in val_classes:
        for i in range(1, 21):
            val_list.append(val_it+'----'+str(i))
    test_list = []
    for test_it in test_classes:
        for j in range(1, 21):
            test_list.append(test_it+'----'+str(j))
    train_path = os.path.join(dataset_fewshot_path, 'train')
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    val_path = os.path.join(dataset_fewshot_path, 'val')
    if not os.path.exists(val_path):
        os.makedirs(val_path)
    test_path = os.path.join(dataset_fewshot_path, 'test')
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    for File in filename_list:
        File_id = File.split('.')[0]
        File_path = os.path.join(dataset_raw_path, File)
        if File_id in train_list:
            shutil.copyfile(File_path, os.path.join(train_path, File))
        elif File_id in val_list:
            shutil.copyfile(File_path, os.path.join(val_path, File))
        elif File_id in test_list:
            shutil.copyfile(File_path, os.path.join(test_path, File))
    for mode in ['train', 'val', 'test']:
        mode_path = os.path.join(dataset_fewshot_path, mode)
        mode_list = os.listdir(mode_path)
        mode_unique = []
        for mode_it in mode_list:
            mode_unique.append(mode_it.split('----')[0])
        mode_unique = np.unique(mode_unique).tolist()
        print('===In {}, PNG number is {}==='.format(mode, len(glob.glob(mode_path+'/*.mat'))))
        print('===In {}, MAT number is {}==='.format(mode, len(glob.glob(mode_path+'/*.jpg'))))
        print('===In {}, total number is {}==='.format(mode, len(mode_unique)))
    print('===Data Split is finished===')

def index_to_label(dataset_raw_path):
    filename_list = os.listdir(dataset_raw_path)
    classes_list = []
    for item in filename_list:
        class_item = item.split('----')[0]
        classes_list.append(class_item)
    classes_unique = np.unique(classes_list).tolist()
    index_dict = {}
    for i, name in enumerate(classes_unique):
        index_dict[name] = i
    return index_dict

def load_imgandmat(mode_path, index_dict):
    Filename_list = os.listdir(mode_path)
    Filename_list.sort(key=lambda x: (x.split('----')[0], x.split('----')[-1].split('.')[0]))
    index_name_list = []
    for name in Filename_list:
        index_name_list.append(name.split('.')[0])
    index_unique_list = np.unique(index_name_list).tolist()
    df_dict = {}
    img_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    for item in index_unique_list:
        per_mat_name = item + '.mat'
        per_mat_path = os.path.join(mode_path, per_mat_name)
        per_img_name = item + '.jpg'
        per_img_path = os.path.join(mode_path, per_img_name)
        df_dict[item] = loadmat(per_mat_path)
        per_img = Image.open(per_img_path)
        df_dict[item]['Time_Frequency'] = img_transform(per_img)
    for name, k in df_dict.items():
        del k['__header__']
        del k['__version__']
        del k['__globals__']
    df = pd.DataFrame.from_dict(df_dict).T
    df = df.reset_index().rename({'index': 'File_index'},axis=1)
    DE_time_process = df['DE_time'].values
    DE_time_process = preprocessing.scale(np.hstack(DE_time_process).T)
    DE_time_df = pd.DataFrame(DE_time_process)
    for i in range(512):
        DE_time_df = DE_time_df.rename({i: 'DE_'+str(i)},axis=1)
    FFT_data_process = df['FFT_data'].values
    FFT_data_process = preprocessing.scale(np.hstack(FFT_data_process).T)
    FFT_data_df = pd.DataFrame(FFT_data_process)
    for j in range(256):
        FFT_data_df = FFT_data_df.rename({j: 'FFT_'+str(j)},axis=1)
    df_fusion1 = pd.concat([df.drop(['DE_time', 'FFT_data'],axis=1), DE_time_df, FFT_data_df],axis=1,join='outer')
    df_fusion = df_fusion1.assign(label=df['File_index'].apply(lambda x:index_dict[x.split('----')[0]]))
    return df_fusion

class Sampler():
    def __init__(self, opt, mode):
        self.data_path = opt.data_path
        self.raw_path = opt.raw_path
        self.dataset_name = opt.dataset_name
        self.mode = mode
        if self.mode == 'train':
            self.k = opt.k_train
            self.n = opt.n_train
            self.q = opt.q_train
        else:
            self.k = opt.k_val
            self.n = opt.n_val
            self.q = opt.q_val
        dataset_fewshot_path = os.path.join(self.data_path, self.dataset_name)
        dataset_raw_path = os.path.join(self.raw_path, self.dataset_name)+'_raw'
        if not check_dataset_processed(dataset_fewshot_path):
            transfer_data_from_raw(dataset_raw_path=dataset_raw_path, dataset_fewshot_path=dataset_fewshot_path)
        self.dataset_index_dict = index_to_label(dataset_raw_path=dataset_raw_path)
        mode_path = os.path.join(dataset_fewshot_path, self.mode)
        self.df_fusion = load_imgandmat(mode_path=mode_path, index_dict=self.dataset_index_dict)

    def __iter__(self):
        return self

    def __next__(self):
        filename_list = self.df_fusion['File_index'].unique()
        mode_classes_list = []
        for filename in filename_list:
            file_item = filename.split('----')[0]
            mode_classes_list.append(file_item)
        mode_unique_class = np.unique(mode_classes_list).tolist()
        episodes_sample_k = np.random.choice(mode_unique_class, size=self.k, replace=False)
        #Support dataset list in episode
        Time_frequency_support_episodes = []
        DE_time_support_episodes = []
        FFT_data_support_episodes = []
        y_support_episodes = []
        #Query  dataset list in episode
        Time_frequency_query_episodes = []
        DE_time_query_episodes = []
        FFT_data_query_episodes = []
        y_query_episodes = []
        for it in episodes_sample_k:
            corresponding_label = self.dataset_index_dict[it]
            df_oneclasses = self.df_fusion[self.df_fusion['label'].isin([corresponding_label])]
            df_sampler_inone = df_oneclasses.sample(n=self.n+self.q, replace=False)
            #Support dataset
            df_support_inone = df_sampler_inone.iloc[:self.n]
            Time_frequency_support = torch.stack(df_support_inone['Time_Frequency'].values.tolist())
            DE_time_support = torch.tensor(df_support_inone.iloc[:, 2:514].values.astype(float)).unsqueeze(dim=1)
            FFT_data_support = torch.tensor(df_support_inone.iloc[:, 514:-1].values.astype(float)).unsqueeze(dim=1)
            y_support = torch.tensor(df_support_inone['label'].values.astype(int)).view(-1, 1)
            Time_frequency_support_episodes.append(Time_frequency_support)
            DE_time_support_episodes.append(DE_time_support)
            FFT_data_support_episodes.append(FFT_data_support)
            y_support_episodes.append(y_support)
            #Query dataset
            df_query_inone = df_sampler_inone.iloc[self.n:]
            Time_frequency_query = torch.stack(df_query_inone['Time_Frequency'].values.tolist())
            DE_time_query = torch.tensor(df_query_inone.iloc[:, 2:514].values.astype(float)).unsqueeze(dim=1)
            FFT_data_query = torch.tensor(df_query_inone.iloc[:, 514:-1].values.astype(float)).unsqueeze(dim=1)
            y_query = torch.tensor(df_query_inone['label'].values.astype(int)).view(-1, 1)
            Time_frequency_query_episodes.append(Time_frequency_query)
            DE_time_query_episodes.append(DE_time_query)
            FFT_data_query_episodes.append(FFT_data_query)
            y_query_episodes.append(y_query)
        TF_support = torch.stack(Time_frequency_support_episodes).to(torch.float32)             #tensor[k, n, 3, 224, 224]
        DE_support = torch.stack(DE_time_support_episodes).to(torch.float32)                    #tensor[k, n, 1, 512]
        FFT_support = torch.stack(FFT_data_support_episodes).to(torch.float32)                  #tensor[k, n, 1, 256]
        ys = torch.stack(y_support_episodes).to(torch.long)                                     #tensor[k, n, 1]

        TF_query = torch.stack(Time_frequency_query_episodes).to(torch.float32)                 #tensor[k, q, 3, 224, 224]
        DE_query = torch.stack(DE_time_query_episodes).to(torch.float32)                        #tensor[k, q, 1, 512]
        FFT_query = torch.stack(FFT_data_query_episodes).to(torch.float32)                      #tensor[k, q, 1, 256]
        yq = torch.stack(y_query_episodes).to(torch.long)                                       #tensor[k, q, 1]

        return TF_support, DE_support, FFT_support, ys, TF_query, DE_query, FFT_query, yq