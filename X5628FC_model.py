# -*- coding:utf-8 -*-

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.layers import Input

from keras.backend import concatenate

from keras import Model



##############################
#
# 模型结构
#
##############################

def model_def():

    dropout_rate = 0.5 # 舍弃比率
  


    # 模型1
    input_1 = Input(shape = (10001, 5, 1))
  
    model_1 = Conv2D(64, kernel_size = [24, 1], strides = [4, 1], kernel_initializer='he_uniform', padding = 'same', activation = 'relu')(input_1)
    model_1 = Conv2D(64, kernel_size = [24, 1], strides = [4, 1], kernel_initializer='he_uniform', padding = 'same', activation = 'relu')(model_1)
    model_1 = Conv2D(64, kernel_size = [24, 1], strides = [4, 1], kernel_initializer='he_uniform', padding = 'same', activation = 'relu')(model_1)
  
    model_1 = MaxPooling2D(pool_size = (2, 1), strides = (2, 1))(model_1)
    model_1 = BatchNormalization()(model_1)
  
    model_1 = Conv2D(128, kernel_size = [24, 1], strides = [4, 1], kernel_initializer='he_uniform', padding = 'same', activation = 'relu')(model_1)
    model_1 = Conv2D(128, kernel_size = [24, 1], strides = [4, 1], kernel_initializer='he_uniform', padding = 'same', activation = 'relu')(model_1)
    model_1 = Conv2D(128, kernel_size = [24, 1], strides = [4, 1], kernel_initializer='he_uniform', padding = 'same', activation = 'relu')(model_1)
  
    model_1 = MaxPooling2D(pool_size = (2, 1), strides = (2, 1))(model_1)
  
    model_1 = Flatten()(model_1)
    model_1 = BatchNormalization()(model_1)
  
    model_1 = Dense(2048, activation = 'relu')(model_1)
    model_1 = Dropout(dropout_rate)(model_1)



    # 模型2
    input_2 = Input(shape = (10001, 5, 1))
  
    model_2 = Conv2D(64, kernel_size = [24, 1], strides = [4, 1], kernel_initializer='he_uniform', padding = 'same', activation = 'relu')(input_2)
    model_2 = Conv2D(64, kernel_size = [24, 1], strides = [4, 1], kernel_initializer='he_uniform', padding = 'same', activation = 'relu')(model_2)
    model_2 = Conv2D(64, kernel_size = [24, 1], strides = [4, 1], kernel_initializer='he_uniform', padding = 'same', activation = 'relu')(model_2)
  
    model_2 = MaxPooling2D(pool_size = (2, 1), strides = (2, 1))(model_2)
    model_2 = BatchNormalization()(model_2)
  
    model_2 = Conv2D(128, kernel_size = [24, 1], strides = [4, 1], kernel_initializer='he_uniform', padding = 'same', activation = 'relu')(model_2)
    model_2 = Conv2D(128, kernel_size = [24, 1], strides = [4, 1], kernel_initializer='he_uniform', padding = 'same', activation = 'relu')(model_2)
    model_2 = Conv2D(128, kernel_size = [24, 1], strides = [4, 1], kernel_initializer='he_uniform', padding = 'same', activation = 'relu')(model_2)
  
    model_2 = MaxPooling2D(pool_size = (2, 1), strides = (2, 1))(model_2)
  
    model_2 = Flatten()(model_2)
    model_2 = BatchNormalization()(model_2)
  
    model_2 = Dense(2048, activation = 'relu')(model_2)
    model_2 = Dropout(dropout_rate)(model_2)



    # 合并模型
    merge = concatenate([model_1, model_2] , axis=-1)
    
    merge = Dense(2048, activation = 'relu')(merge)
    merge = Dropout(dropout_rate)(merge)
    
    merge = Dense(1000, activation = 'relu')(merge)
    merge = Dense(2, activation = 'softmax')(merge)

    model = Model(inputs = [input_1, input_2], outputs = merge)
  
    return model



##############################
#
# 检修区
#
##############################

if __name__ == '__main__':
  
    # 用来放测试代码
    pass

    classifier = model_def()

    from keras.utils import plot_model
    plot_model(classifier, to_file='model.png')
