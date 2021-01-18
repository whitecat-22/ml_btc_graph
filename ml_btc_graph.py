#Keras
from __future__ import print_function
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.models import Sequential
from keras.optimizers import Adam

#TensorFlow
import tensorflow as tf

#LSTM
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import poloniex
import time


polo = poloniex.Poloniex()

# データ取得
chart_data = polo.returnChartData('USDT_BTC',
                                  period=7200,
                                  start=time.time()-polo.DAY*500,
                                  end=time.time())
df = pd.DataFrame(chart_data)  # dataframe化

data = df["close"].astype("float32")

#グラフ設定
plt.rcParams['figure.figsize'] = 15, 6
plt.title("BTC_Chart")
plt.xlabel("date")
plt.ylabel("price")
plt.grid(True)

#プロット
plt.plot(data)
plt.show()

#移動平均
data1 = df["close"].rolling(6).mean()
data1 = data1.fillna(data1[5])

#プロット
plt.plot(data)
plt.plot(data1)

plt.grid(True)
plt.show()


#データから変数とラベルを生成し、訓練データとテストデータに分割する
def data_split(data, v_size=30, train_split_latio=0.7):
    data = data.astype("float32")  # データをfloat32型に変換
    x, t = [], []
    data_len = len(data)  # 総データ数

    #変数とラベルの生成
    for i in range(data_len - v_size):
        x_valu = data[i: i+v_size]  # 連続したmax_len個の値
        t_valu = data[i+v_size]  # x_valuの次の値

        x.append(x_valu)  # 入力変数ベクトル
        t.append(t_valu)  # 出力変数ベクトル

    #ndarray型に変換し形を直す
    x = np.array(x).reshape(data_len-v_size, v_size, 1)
    t = np.array(t).reshape(data_len-v_size, 1)

    #訓練データとテストデータに分割
    border = int(data_len * train_split_latio)  # 分割境界値
    x_train, x_test = x[: border], x[border:]  # 訓練データ
    t_train, t_test = t[: border], t[border:]  # テストデータ

    return x_train, x_test, t_train, t_test

#LSTMモデルの生成
def create_LSTM(v_size, in_size, out_size, hidden_size):
    tf.set_random_seed = (20180822)
    model = Sequential()
    model.add(LSTM(hidden_size, batch_input_shape=(None, v_size, in_size),
                   recurrent_dropout=0.5))
    model.add(Dense(out_size))
    model.add(Activation("linear"))

    return model


#各パラメータの定義
now_data = data  # 扱う元データ
v_size = 30  # 入力データ幅
train_split_ratio = 0.7  # 訓練データとテストデータの分割割合
x_train, x_test, t_train, t_test = data_split(
    now_data, v_size, train_split_ratio)  # データの分割

mean = np.mean(x_train)  # 平均値の保存
std = np.std(x_train)  # 標準偏差の保存
x_train = (x_train - mean) / std  # 標準化
x_test = (x_test - mean) / std

tmean = np.mean(t_train)
tstd = np.std(t_train)
t_train = (t_train - tmean) / tstd  # 出力変数も同じように標準化
t_test = (t_test - tmean) / tstd

in_size = x_train.shape[2]  # 入力数
out_size = t_train.shape[1]  # 出力数
hidden_size = 300  # 隠れ層の数
epochs = 100  # エポック数
batch_size = 30  # バッチサイズ


# 学習
early_stopping = EarlyStopping(patience=10)  # ストップカウント
model = create_LSTM(v_size, in_size, out_size, hidden_size)  # インスタンス生成
model.compile(loss="mean_squared_error", optimizer=Adam(0.0001))  # 損失関数定義
model.fit(x_train, t_train, batch_size=batch_size, epochs=epochs,
          shuffle=True, callbacks=[early_stopping], validation_split=0.1)  # 学習


#予測
pred_train = model.predict(x_train)     #訓練データ予測
pred_train = pred_train * tstd + tmean  #標準化したデータを戻す

a = np.zeros((v_size, 1))
b = pred_train
pred_train = np.vstack((a, b))   #データ長を合わせるため0ベクトルと結合


#プロット
plt.figure()
plt.plot(pred_train, color="r", label="predict")
plt.plot(now_data, color="b", label="real")
plt.grid(True)
plt.legend()
plt.show()


pred_test = model.predict(x_test)     #テストデータ予測
pred_test = pred_test * tstd + tmean  #標準化したデータを戻す

a = np.zeros((v_size + x_train.shape[0], 1))
b = pred_test
pred_test = np.vstack((a, b))   #データ長を合わせるため0ベクトルと結合


#プロット
plt.figure()
plt.plot(pred_test, color="r", label="predict")
plt.plot(now_data, color="b", label="real")
plt.grid(True)
plt.legend()
plt.show()


#trainデータ騰落正答確率測定
data_score = 0
for i in range(pred_train.shape[0] - v_size - 1):
    if pred_train[i + v_size + 1] - pred_train[i + v_size] > 0:
        if data[i + v_size + 1]-data[i + v_size] > 0:
            data_score += 1
    if pred_train[i + v_size + 1] - pred_train[i + v_size] < 0:
        if data[i + v_size + 1] - data[i + v_size] < 0:
            data_score += 1
print("train score: {}".format(
    data_score / (pred_train.shape[0] - v_size - 1)))

#testデータ騰落正答確率測定
N = x_train.shape[0] + v_size
data_score = 0
for i in range(pred_test.shape[0] - N - 1):
    if pred_test[i + N + 1] - pred_test[i + N] > 0:
        if data[i + N + 1] - data[i + N] > 0:
            data_score += 1
    if pred_test[i + N + 1] - pred_test[i + N] < 0:
        if data[i + N + 1] - data[i + N] < 0:
            data_score += 1
print("test score: {}".format(data_score / (pred_test.shape[0] - N - 1)))
