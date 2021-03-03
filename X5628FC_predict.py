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
# 测试集迭代器
#
##############################

from keras.preprocessing.image import ImageDataGenerator

BATCH_SIZE = 32 # 每次大小

test_datagen = ImageDataGenerator(rescale = 1. / 255)

# 这个是用来产生label的
test_generator = test_datagen.flow_from_directory(directory = './test_anchor1/', target_size=(10001, 5),
                                                  color_mode = 'grayscale',
                                                  class_mode = 'categorical',
                                                  # "categorical"会返回2D的one-hot编码标签, "binary"返回1D的二值标签."sparse"返回1D的整数标签
                                                  batch_size = BATCH_SIZE,
                                                  shuffle = False)  # 不要乱

def generator_two_test():
    test_generator1 = test_datagen.flow_from_directory(directory = './test_anchor1/', target_size = (10001, 5),
                                                      color_mode = 'grayscale',
                                                      class_mode = 'categorical', # "categorical"会返回2D的one-hot编码标签, "binary"返回1D的二值标签."sparse"返回1D的整数标签
                                                      batch_size = BATCH_SIZE,
                                                      shuffle = False) # 不要乱

    test_generator2 = test_datagen.flow_from_directory(directory = './test_anchor2/', target_size = (10001, 5),
                                                      color_mode = 'grayscale',
                                                      class_mode = 'categorical', # "categorical"会返回2D的one-hot编码标签, "binary"返回1D的二值标签."sparse"返回1D的整数标签
                                                      batch_size = BATCH_SIZE,
                                                      shuffle = False) # 不要乱
    while True:
        out1 = test_generator1.next()
        out2 = test_generator2.next()
        yield [out1[0], out2[0]] # , out1[1]  # 返回两个的组合和结果



##############################
#
# 加载已经训练好的模型
#
##############################

# 跳过训练，直接加载模型
from keras.models import load_model
clf = load_model('best_model.h5')

gc.collect() # 回收全部代垃圾，避免内存泄露



##############################
#
# 预测
#
##############################

# 新加入内容，用来评估模型质量
# 计算auc和绘制roc_curve
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# 先测算准确度
# score = clf.evaluate_generator(generator = [test_generator1, test_generator2], steps = len(test_generator1))
# fancy_print('loss & acc', score)

# 打印所有内容
# np.set_printoptions(threshold = np.inf)

# 利用model.predict获取测试集的预测概率
y_prob = clf.predict_generator(generator = generator_two_test(), steps = 3072 * 2 // BATCH_SIZE)
fancy_print('y_prob', y_prob, '.')
fancy_print('y_prob.shape', y_prob.shape, '-')



# 获得label
label_test_tag = test_generator.class_indices
label_test_name = test_generator.filenames

label_test = []
for i in label_test_name:
    label = i.split('\\')[0] # 把类别分出来
    label_test.append(int(label_test_tag[label])) # 弄成数字呗

from keras.utils.np_utils import to_categorical
label_test = to_categorical(label_test)

fancy_print('label_test', label_test, '.')
fancy_print('label_test.shape', label_test.shape, '-')
gc.collect() # 回收全部代垃圾，避免内存泄露



# 为每个类别计算ROC曲线和AUC
fpr = dict()
tpr = dict()
roc_auc = dict()



# 二分类问题
n_classes = label_test.shape[1] # n_classes = 2
fancy_print('n_classes', n_classes) # n_classes = 2

# 使用实际类别和预测概率绘制ROC曲线
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(label_test[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fancy_print('fpr', fpr)
fancy_print('tpr', tpr)
fancy_print('cnn_roc_auc', roc_auc)



plt.figure()
lw = 2
plt.plot(fpr[0], tpr[0], color = 'darkorange', 
         lw = lw, label = 'ROC curve (area = %0.2f)' % roc_auc[1])
plt.plot([0, 1], [0, 1], color = 'navy', lw = lw, linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC)')
plt.legend(loc = "lower right")

plt.show()
