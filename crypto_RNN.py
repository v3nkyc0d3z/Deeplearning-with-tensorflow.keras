import pandas as pd
import numpy as np
from collections import deque
import random
from sklearn import preprocessing
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,LSTM,BatchNormalization,LSTM
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tqdm import tqdm

time_frame = 60 #time difference between subsequent records(seconds)
future_ = 3 #see every 3 minutes in future to classify a buy or sell
coin_to_predict = 'BTC-USD'
EPOCHS = 10
BATCH_SIZE = 64
NAME = "{}time-{}future-{}".format(time_frame,future_,int(time.time()))


def labeling(future,present):
    if float(future) > float(present):
        return 1
    else:
        return 0

def preprocessing_df(df):
    df = df.drop("future",1)
    for col in df.columns:
        if col != "label":
            df[col] = df[col].pct_change()
            df.dropna(inplace = True)
            df[col]= preprocessing.scale(df[col].values)
    df.dropna(inplace= True)
    sequences = []
    prev_data = deque(maxlen= time_frame)
    for i in tqdm(df.values):
        prev_data.append([n for n in i[:-1]])
        if len(prev_data) == time_frame:
            sequences.append([np.array(prev_data),i[-1]])
    random.shuffle(sequences)
    print("Balancing the data frame....")

    buy=[]
    sell=[]
    for seq,target in sequences:
        if target == 0:
            sell.append([seq,target])
        else:
            buy.append([seq,target])
    min_count = min(len(buy),len(sell))
    random.shuffle(buy)
    random.shuffle(sell)
    buy = buy[:min_count]
    sell = sell[:min_count]
    sequences = buy+sell
    random.shuffle(sequences)
    X=[]
    y=[]

    for sequence,target in tqdm(sequences):
        X.append(sequence)
        y.append(target)
    print("returning values....")
    return np.array(X),np.array(y)

main_df = pd.DataFrame()
coins_ = ['BCH-USD','BTC-USD','ETH-USD','LTC-USD']
for coin in coins_:
    f_name = "crypto_data/{}.csv".format(coin)
    df = pd.read_csv(f_name, names= ["time","low","High","open","close","volume"])
    df.rename(columns={'close':'{}_close'.format(coin),'volume':'{}_volume'.format(coin)},inplace = True)
    df.set_index("time", inplace=True)
    df = df[["{}_close".format(coin),"{}_volume".format(coin)]]
    if len(main_df)== 0:
        main_df = df
    else:
        main_df = main_df.join(df)

main_df.fillna(method = "ffill",inplace = True)
main_df.dropna(inplace = True)


main_df['future'] = main_df["{}_close".format(coin_to_predict)].shift(-future_)
main_df["label"] = list(map(labeling,main_df["{}_close".format(coin_to_predict)],main_df['future']))

time = sorted(main_df.index.values)
Val_size = sorted(main_df.index.values)[-int(len(main_df)*0.05)]
validation_df = main_df[(main_df.index)>= Val_size]
main_df = main_df[(main_df.index)<Val_size]

print("Beginning PreProcessing.....")

train_x,train_y = preprocessing_df(main_df)
test_x,test_y = preprocessing_df(validation_df)
print(f"train data: {len(train_x)} validation: {len(test_x)}")
# print(f"Dont buys: {train_y.0)}, buys: {train_y.count(1)}")
# print(f"VALIDATION Dont buys: {test_y.count(0)}, buys: {test_y.count(1)}")

model = Sequential()

model.add(LSTM(128,input_shape = (train_x.shape[1:]),return_sequences = True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128,return_sequences = True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(BatchNormalization())


model.add(Dense(32,activation= 'relu'))
model.add(Dropout(0.2))

model.add(Dense(2,activation = 'softmax'))

opt = tf.keras.optimizers.Adam(lr=0.001,decay = 1e-6)
model.compile(loss="sparse_categorical_crossentropy", optimizer = opt,
              metrics=["accuracy"])

tensorboard= TensorBoard(log_dir = 'RNN_logs\\{}'.format(NAME))
filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"
checkpoint = ModelCheckpoint("Crypto_models\\{}.model".format(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max'))

history = model.fit(
    train_x,train_y,
    batch_size = BATCH_SIZE,
    epochs=EPOCHS,
    validation_data = (test_x,test_y),
    callbacks= [tensorboard])

score = model.evaluate(test_x, test_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])