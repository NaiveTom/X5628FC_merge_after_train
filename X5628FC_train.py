# -*- coding:utf-8 -*-

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 去除不必要的信息

# 垃圾回收机制
import gc
gc.enable()

import numpy as np

# 打印格式
def fancy_print(n = None, c = None, s = '#'):
    print(s * 40)
    print(n)
    print(c)
    print(s * 40)
    print() # 避免了混乱

# 设置GPU使用方式为渐进式，避免显存占满
# 获取GPU列表
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 设置GPU为增长式占用
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print('设置GPU为增长式占用')
    except RuntimeError as e:
        # 打印异常
        fancy_print('RuntimeError', e)

##############################
#
# 构建迭代器
#
##############################

from keras.preprocessing.image import ImageDataGenerator

# train_datagen = ImageDataGenerator(horizontal_flip = True, vertical_flip = True, rescale = 1. / 255) # 上下翻转 左右翻转
train_datagen = ImageDataGenerator(rescale = 1. / 255)
val_datagen = ImageDataGenerator(rescale = 1. / 255)

BATCH_SIZE = 32 # 每次大小

def generator_two_train():
    train_generator1 = train_datagen.flow_from_directory(directory = './train_anchor1/', target_size = (10001, 5),
                                                         color_mode = 'grayscale',
                                                         class_mode = 'categorical', # 'categorical'会返回2D的one-hot编码标签, 'binary'返回1D的二值标签, 'sparse'返回1D的整数标签
                                                         batch_size = BATCH_SIZE,
                                                         shuffle = True,
                                                         seed = 42)  # 相同方式打散
    train_generator2 = train_datagen.flow_from_directory(directory = './train_anchor2/', target_size = (10001, 5),
                                                         color_mode = 'grayscale',
                                                         class_mode = 'categorical', # 'categorical'会返回2D的one-hot编码标签, 'binary'返回1D的二值标签, 'sparse'返回1D的整数标签
                                                         batch_size = BATCH_SIZE,
                                                         shuffle = True,
                                                         seed = 42)  # 相同方式打散
    while True:
        out1 = train_generator1.next()
        out2 = train_generator2.next()
        yield [out1[0], out2[0]], out1[1]  # 返回两个的组合和结果

def generator_two_val():
    val_generator1 = val_datagen.flow_from_directory(directory = './val_anchor1/', target_size = (10001, 5),
                                                     color_mode = 'grayscale',
                                                     class_mode = 'categorical', # 'categorical'会返回2D的one-hot编码标签, 'binary'返回1D的二值标签, 'sparse'返回1D的整数标签
                                                     batch_size = BATCH_SIZE,
                                                     shuffle =True,
                                                     seed = 42)  # 相同方式打散
    val_generator2 = val_datagen.flow_from_directory(directory = './val_anchor2/', target_size = (10001, 5),
                                                     color_mode = 'grayscale',
                                                     class_mode = 'categorical', # 'categorical'会返回2D的one-hot编码标签, 'binary'返回1D的二值标签, 'sparse'返回1D的整数标签
                                                     batch_size = BATCH_SIZE,
                                                     shuffle = True,
                                                     seed = 42)  # 相同方式打散
    while True:
        out1 = val_generator1.next()
        out2 = val_generator2.next()
        yield [out1[0], out2[0]], out1[1] # 返回两个的组合和结果

##############################
#
# 模型搭建
#
##############################

# 如果出现版本不兼容，那么就用这两句代码，否则会报警告
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

from sklearn import metrics
from keras.callbacks import ModelCheckpoint

# 导入X5628FC_model.py
import X5628FC_model

clf = X5628FC_model.model_def()

clf.summary() # 打印模型结构

from keras.optimizers import Adam
clf.compile(loss = 'categorical_crossentropy', # sparse_categorical_crossentropy binary_crossentropy
            optimizer = Adam(lr = 0.0001, decay = 0.00001),
            metrics = ['accuracy'])

'''
filename = 'best_model.h5'
modelCheckpoint = ModelCheckpoint(filename, monitor = 'val_accuracy', save_best_only = True, mode = 'max')
'''
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'val_accuracy', patience = 10, restore_best_weights = True)

gc.collect() # 回收全部代垃圾，避免内存泄露



'''
fancy_print('train_generator.next()[0]', train_generator.next()[0], '+')
fancy_print('train_generator.next()[1]', train_generator.next()[1], '+')
fancy_print('train_generator.next()[0].shape', train_generator.next()[0].shape, '+')
fancy_print('train_generator.next()[1].shape', train_generator.next()[1].shape, '+')

fancy_print('val_generator.next()[0]', val_generator.next()[0], '-')
fancy_print('val_generator.next()[1]', val_generator.next()[1], '-')
fancy_print('val_generator.next()[0].shape', val_generator.next()[0].shape, '-')
fancy_print('val_generator.next()[1].shape', val_generator.next()[1].shape, '-')
'''
##############################
#
# 模型训练
#
##############################

# 不需要再算多少个epoch了，自己会算
history = clf.fit_generator(generator = generator_two_train(),
                            epochs = 100,
                            validation_data = generator_two_val(),

                            steps_per_epoch = 24568 * 2 // BATCH_SIZE, # 全部训练
                            validation_steps = 3071 * 2 // BATCH_SIZE, # 全部验证

                            callbacks = [early_stopping],
                            
                            shuffle = True) # 再次 shuffle

clf.save('best_model.h5')
