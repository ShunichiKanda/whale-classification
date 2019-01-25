import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
import cv2
from matplotlib.pyplot import imshow
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras import layers
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Model
import keras.backend as K
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

#データ読み込み
csv_train_path = os.path.abspath('input/train.csv')
img_train_path = os.path.abspath('input/train')

#データフレーム作成
train_df = pd.read_csv(csv_train_path)
train_df['Image_path'] = [os.path.join(img_train_path,whale) for whale in train_df['Image']]
train_df2 = pd.get_dummies(train_df['Id']) #[[3],[1]]を[[0, 0, 1], [1, 0, 0]]に変換する。機械学習ではこの処理が必須

#画像データをXに入れるための関数
def prepareImages(data, m, dataset):
    print("Preparing images")
    X_train = np.zeros((m, 100, 100, 3))
    count = 0
    
    for fig in data['Image']:
        #ここで画像サイズは100x100x3に縮小
        img = image.load_img("input/"+dataset+"/"+fig, target_size=(100, 100, 3))
        x = image.img_to_array(img)
        x = preprocess_input(x)
        X_train[count] = x
        #これを書いておけば、時間がかかっても一応動いていることは分かる
        if (count%500 == 0):
            print("Processing image: ", count+1, ", ", fig)
        count += 1
    
    return X_train

#上の関数を使って、X_trainにデータを入れる
X_train = prepareImages(train_df, train_df.shape[0], "train")
y_train = train_df2.as_matrix()

#モデル作成
model = Sequential()
model.add(Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0', input_shape = (100, 100, 3)))
model.add(BatchNormalization(axis = 3, name = 'bn0'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), name='max_pool'))
model.add(Conv2D(64, (3, 3), strides = (1,1), name="conv1"))
model.add(Activation('relu'))
model.add(AveragePooling2D((3, 3), name='avg_pool'))
model.add(Flatten())
model.add(Dense(500, activation="relu", name='rl'))
model.add(Dropout(0.8))
model.add(Dense(y_train.shape[1], activation='softmax', name='sm'))

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=100, batch_size=100, verbose=1)
gc.collect()                  
plt.plot(history.history['acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()