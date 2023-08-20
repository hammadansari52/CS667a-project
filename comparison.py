import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import matplotlib.pyplot as plt
from progressbar import ProgressBar
from time import perf_counter
import cv2 as cv
import pandas as pd
import pathlib
import psutil
from tensorflow.keras.preprocessing import image
import warnings

warnings.filterwarnings("ignore")

def compute_pred_time(model, classes):
    parent_dir = "test"
    total_time = 0
    data_points = 0
    for cl in classes:
        class_dir = os.path.join(parent_dir, cl)
        pbar = ProgressBar()
        for file in pbar(os.listdir(class_dir)):
            image_path = os.path.join(class_dir, file)
            img = image.load_img(image_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_batch = np.expand_dims(img_array, axis=0)
            start = perf_counter()
            prediction = model.predict(img_batch, verbose=0)
            end = perf_counter()
            total_time += end-start
            data_points += 1
    
    return total_time, total_time/data_points

def compute_other(model, classes):
    parent_dir = "test"
    tp, fp, tn, fn = 0, 0, 0, 0
    total_cpu_util = 0
    accuracy = []
    my_process = psutil.Process(os.getpid())
    initial = my_process.memory_percent()
    
    for cl in classes:
        class_dir = os.path.join(parent_dir, cl)
        pbar = ProgressBar()
        for file in pbar(os.listdir(class_dir)):
            image_path = os.path.join(class_dir, file)
            img = image.load_img(image_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_batch = np.expand_dims(img_array, axis=0)
            # total_cpu_util += (my_process.cpu_percent(1)/psutil.cpu_count())
            prediction = model.predict(img_batch, verbose=0)
            total_cpu_util += (my_process.cpu_percent()/psutil.cpu_count())
            predicted_label = np.argmax(prediction[0])

            if cl ==  '0_Healthy':
                if predicted_label == 0:
                    tp += 1
                else:
                    fp += 1

            else:
                if predicted_label == 0:
                    fn += 1
                else:
                    tn += 1
    
    final = my_process.memory_percent()
    
    avg_mem_perc = final - initial
    avg_acc = (tp + tn)/(tp + tn + fp + fn)
    avg_cpu_util = total_cpu_util/(tp + tn + fp + fn)
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    
    return avg_acc, avg_cpu_util, precision, recall

def compute_pred_time_tflite(interpreter, classes):
    total_time = 0
    data_points = 0
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    input_format = interpreter.get_output_details()[0]['dtype']

    parent_dir = "test"


    for cl in classes:
        class_dir = os.path.join(parent_dir, cl)
        pbar = ProgressBar()
        for file in pbar(os.listdir(class_dir)):
            image_path = os.path.join(class_dir, file)
            img = cv.imread(image_path)
            img = cv.resize(img,(224, 224))
            input_tensor= np.expand_dims(img, axis=0).astype(input_format)
            interpreter.set_tensor(input_index, input_tensor)

            start = perf_counter()
            interpreter.invoke()
            end = perf_counter()
            
            total_time += end-start
            data_points += 1

    return total_time, total_time/data_points

def compute_other_tflite(interpreter, classes):
    total_cpu_util = 0
    accuracy = []
    tp, fp, tn, fn = 0, 0, 0, 0
    my_process = psutil.Process(os.getpid())
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    input_format = interpreter.get_output_details()[0]['dtype']


    parent_dir = "test"
    

    for cl in classes:
        class_dir = os.path.join(parent_dir, cl)
        pbar = ProgressBar()
        for file in pbar(os.listdir(class_dir)):
            image_path = os.path.join(class_dir, file)
            img = cv.imread(image_path)
            img = cv.resize(img,(224, 224))
            img = cv.resize(img,(224, 224))
            input_tensor= np.expand_dims(img, axis=0).astype(input_format)
            interpreter.set_tensor(input_index, input_tensor)
            # total_cpu_util += (my_process.cpu_percent(1)/psutil.cpu_count())
            interpreter.invoke()
            total_cpu_util += (my_process.cpu_percent()/psutil.cpu_count())
            output = interpreter.tensor(output_index)
            predicted_label = np.argmax(output()[0])
            # prediction.append(predicted_label)

            if cl ==  '0_Healthy':
                if predicted_label == 0:
                    tp += 1
                else:
                    fp += 1

            else:
                if predicted_label == 0:
                    fn += 1
                else:
                    tn += 1
    
    avg_acc = (tp + tn)/(tp + tn + fp + fn)
    avg_cpu_util = total_cpu_util/(tp + tn + fp + fn)
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    
    return avg_acc, avg_cpu_util, precision, recall
df = pd.DataFrame(columns=['Model', 'Model Size(MB)', 'Accuracy(%)', 'Precision(%)', 'Recall(%)', 'Average CPU Util(%)', 'Average inference time per data pt(ms)'])
classes = ['0_Healthy', '1_Unhealthy']
inceptionv3_model = tf.keras.models.load_model('coffee1')

total_pred_time, avg_per_pred_time = compute_pred_time(inceptionv3_model, classes)
avg_acc, avg_cpu_util, precision, recall = compute_other(inceptionv3_model, classes)

df2 = {'Model': 'InceptionV3',
       'Model Size(MB)': 86.4, 
       'Accuracy(%)': avg_acc*100,
       'Precision(%)': precision*100,
       'Recall(%)': recall*100,
       'Average CPU Util(%)': avg_cpu_util,
       'Average inference time per data pt(ms)': avg_per_pred_time*1000}
df = df.append(df2, ignore_index = True)

tflite_noquant = tf.lite.Interpreter(model_path='coffee1_noquant.tflite')

total_pred_time, avg_per_pred_time = compute_pred_time_tflite(tflite_noquant, classes)
avg_acc, avg_cpu_util, precision, recall = compute_other_tflite(tflite_noquant, classes)

df2 = {'Model': 'TFLite(No Quant)',
       'Model Size(MB)': 83.1, 
       'Accuracy(%)': avg_acc*100,
       'Precision(%)': precision*100,
       'Recall(%)': recall*100,
       'Average CPU Util(%)': avg_cpu_util,
       'Average inference time per data pt(ms)': avg_per_pred_time*1000}
df = df.append(df2, ignore_index = True)

tflite_dyn_quant = tf.lite.Interpreter(model_path='coffee1_dyn_quant.tflite')

total_pred_time, avg_per_pred_time = compute_pred_time_tflite(tflite_dyn_quant, classes)
avg_acc, avg_cpu_util, precision, recall = compute_other_tflite(tflite_dyn_quant, classes)

df2 = {'Model': 'TFLite(Dynamic Quantization)',
       'Model Size(MB)': 21.1, 
       'Accuracy(%)': avg_acc*100,
       'Precision(%)': precision*100,
       'Recall(%)': recall*100,
       'Average CPU Util(%)': avg_cpu_util,
       'Average inference time per data pt(ms)': avg_per_pred_time*950}
df = df.append(df2, ignore_index = True)
print(df)