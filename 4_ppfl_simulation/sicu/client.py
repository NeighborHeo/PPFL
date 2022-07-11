# %%
import argparse
import json
import os
import threading
import time
from random import random
import numpy as np
import requests
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, initializers, losses, metrics, regularizers
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import numpy_encoder
import yaml
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
import pickle
from keras.utils import to_categorical

# tf.random.set_random_seed(42)  # tensorflow seed fixing
# %%
'''
    - 데이터 로딩하기
'''

with open('/home/wonseok/PPFL/4_ppfl_simulation/feature_book.yaml') as f:
    feature_book = yaml.load(f, Loader=yaml.FullLoader)
    
#%%
data = pd.read_csv('/home/wonseok/PPFL/4_ppfl_simulation/sicu/data/SICU.csv')
feature = data.drop(columns = ['death','연구등록번호'])
feature_all = data[feature_book["common_features"] + feature_book['sicu_specific']]
label = data['death']

train_X, test_X, train_Y, test_Y = train_test_split(feature_all, label, test_size=0.2, stratify=label, random_state=0)
train_X, valid_X, train_Y, valid_Y = train_test_split(train_X, train_Y, test_size=0.25, stratify=train_Y, random_state=0)


dataset = {"train_X":train_X, "train_Y":train_Y, 'valid_X':valid_X, 'valid_Y':valid_Y,'test_X':test_X, 'test_Y':test_Y}

with open('/home/wonseok/PPFL/4_ppfl_simulation/sicu/dataset.pkl','wb') as f:
        pickle.dump(dataset, f)

# common만 선정
train_X, valid_X, test_X = train_X[feature_book['common_features']], valid_X[feature_book['common_features']], test_X[feature_book['common_features']]
# train_Y, valid_Y, test_Y = to_categorical(train_Y), to_categorical(valid_Y), to_categorical(test_Y)

#%%


#%%

'''
    build ann model
'''
def build_nn_model(
        input_size=len(feature_book['common_features']), n_layers=3, n_hidden_units=30,
        random_seed=None, num_classes=2
):
    """
        creates the MLP network
        :return: model: models.Model`
        """
    # create input layer
    input_layer = layers.Input(shape=input_size, name="input")
    # create intermediate layer
    dense = input_layer
    for i in range(n_layers):
        dense = layers.Dense(
            units=n_hidden_units,
            kernel_initializer=initializers.glorot_uniform(seed=random_seed),
            bias_initializer='zeros',
            activation='relu',
            name='intermediate_dense_{}'.format(i + 1)
        )(dense)
    output_layer = layers.Dense(num_classes,
                                kernel_initializer=initializers.glorot_uniform(seed=random_seed),
                                bias_initializer='zeros',
                                activation='softmax',
                                name='classifier')(dense)
    model = models.Model(input_layer, output_layer)
    model.compile(optimizer=tf.keras.optimizers.SGD(),
                         loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                         metrics=['accuracy'])
    return model

# %%
'''
    request global_weight from server
'''
def request_global_weight():
    print("request_global_weight start")
    result = requests.get(ip_address)
    result_data = result.json()
    global_weight = None
    if result_data is not None:
        global_weight = []
        for i in range(len(result_data)):
            temp = np.array(result_data[i], dtype=np.float32)
            global_weight.append(temp)
    print("request_global_weight end")
    return global_weight

# %%
'''
    update local weight to server
'''
def update_local_weight(local_weight = []):
    print("update local weight start ")
    local_weight_to_json = json.dumps(local_weight, cls=numpy_encoder.NumpyEncoder)
    requests.put(ip_address, data=local_weight_to_json)
    print("update local weight end")
    
# %%
def train_local(global_weight = None):
    print("train local start")
    model = build_nn_model()
    global train_X
    global train_Y
    if global_weight is not None:
        global_weight = np.array(global_weight)
        model.set_weights(global_weight)
    
    model.fit(train_X, train_Y, epochs=10, batch_size=50)
    print("train local end")
    return model.get_weights()
# %%
def delay_compare_weight():
    print("current_round : {}, max_round : {}".format(current_round, max_round))
    if current_round < max_round:
        threading.Timer(delay_time, task).start()
    else:
        '''
        if input_number == 0:
            print_result()
        '''
# %%
def request_current_round():
    result = requests.get(request_round)
    result_data = result.json()
    return result_data

#%%
def request_total_round():
    result = requests.get(request_total_round_url)
    print("this is the total round : ",result.json())
    result_data =  result.json()
    return result_data

# %%
def validation(global_lound = 0, local_weight = []):
    print("validation start")
    if local_weight is not None:
        model = build_nn_model()
        model.set_weights(local_weight)
        result = model.predict(valid_X)
        
        answer_vec = to_categorical(valid_Y)
        answer = valid_Y
        
        auroc_ovr = metrics.roc_auc_score(answer_vec, result, multi_class='ovr')
        auroc_ovo = metrics.roc_auc_score(answer_vec, result, multi_class='ovo')
        result = np.argmax(result, axis=1)
        
        cm = confusion_matrix(answer, result)
        acc = accuracy_score(answer, result)
        f1 = f1_score(answer, result, average=None)
        f2 = f1_score(answer, result, average='micro')
        f3 = f1_score(answer, result, average='macro')
        f4 = f1_score(answer, result, average='weighted')
        print("acc : {}".format(acc))
        print("auroc ovr : {}".format(auroc_ovr))
        print("auroc ovo : {}".format(auroc_ovo))
        print("f1 None : {}".format(f1))
        print("f2 micro : {}".format(f2))
        print("f3 macro : {}".format(f3))
        print("f4 weighted : {}".format(f4))
        print("cm : \n", cm)
        save_result(model, global_lound, global_acc=acc, f1_score=f2, auroc=auroc_ovo)
        print("validation end")
# %%
def save_result(model, global_rounds, global_acc, f1_score, auroc):
    test_name="FL4"
    create_directory("{}".format(test_name))
    create_directory("{}/model".format(test_name))
    if global_acc >= 0.8 :
        file_time = time.strftime("%Y%m%d-%H%M%S")
        model.save_weights("{}/model/{}-{}-{}-{}.h5".format(test_name, file_time, global_rounds, global_acc, f1_score))
    save_csv(test_name=test_name, round = global_rounds, acc = global_acc, f1_score=f1_score, auroc=auroc)


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def save_csv(test_name = "", round = 0, acc = 0.0, f1_score = 0, auroc = 0):
    with open("{}/result.csv".format(test_name), "a+") as f:
        f.write("{}, {}, {}\n".format(round, acc, f1_score, auroc))
# %%
def task():
    print("--------------------")
    '''
        1. global weight request
        2. global weight & local weight compare
        3. start learning & validation processing            
        5. global weight update
        6. delay, next round
    '''
    global current_round
    global_round = request_current_round()
    print("global round : {}, local round :{}".format(global_round, current_round))
    
    total_round = request_total_round()
    
    if global_round == total_round :
        global_weight = request_global_weight()
        with open('final_global_weight.pkl', 'wb') as f:
            pickle.dump(global_weight,f)
        sys.exit(0)
        
    if global_round == current_round:
        print("task train")
        # start next round
        global_weight = request_global_weight()
        local_weight = train_local(global_weight)
        # validation 0 clinet
        if input_number == 0 :
            validation(global_round, global_weight)
        update_local_weight(local_weight)
        delay_compare_weight()
        current_round += 1
    else:
        print("task retry")
        delay_compare_weight()
    print("end task")
    print("====================")
    

# %%
def print_result():
    print("====================")

# %%
from tensorflow.python.keras.callbacks import EarlyStopping
def single_train():
    early_stopping = EarlyStopping(patience=5)
    model = build_nn_model()
    model.fit(train_X, train_Y, epochs=1000, batch_size=32, verbose=1, validation_data=[test_X, test_Y], callbacks=[early_stopping])
    result = model.predict(test_X)
    result = np.argmax(result, axis=1)
    cm = confusion_matrix(test_Y, result)
    print("cm : ", cm)
    acc = accuracy_score(test_Y, result)
    print("acc : {}".format(acc))
    f1 = f1_score(test_Y, result, average=None)
    f2 = f1_score(test_Y, result, average='micro')
    f3 = f1_score(test_Y, result, average='macro')
    f4 = f1_score(test_Y, result, average='weighted')
    print("f1 : {}".format(f1))
    print("f2 : {}".format(f2))
    print("f3 : {}".format(f3))
    print("f4 : {}".format(f4))
# %%
if __name__ == "__main__":
    parameter = argparse.ArgumentParser()
    parameter.add_argument("--number", default=0)
    parameter.add_argument("--currentround", default=0)
    parameter.add_argument("--maxround", default=3000)
    args = parameter.parse_args()
    
    input_number = int(args.number)
    current_round = int(args.currentround)
    max_round = int(args.maxround)
    
    np.random.seed(42)
    np.random.seed(input_number)
    
    print("args : {}".format(input_number))
    global_round = 0
    delay_time = 5  # round check every 5 sec
    # split_train_images, split_train_labels = split_data(input_number)
    
    base_url = "http://127.0.0.1:8000/"
    ip_address = "http://127.0.0.1:8000/weight"
    request_round = "http://127.0.0.1:8000/round"
    request_total_round_url = "http://127.0.0.1:8000/total_round"
    
    start_time = time.time()
    task()