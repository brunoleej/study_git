# tf.train.Checkpoint
# - 모델의 가중치를 저장하는 파일
# latest_checkpoint
# - checkpoint_dir에서 체크포인트 상태를 가져옴, 최신 가중치가 가장 최적의 로스 값을 갖고 있다.

import tensorflow as tf
import numpy as np
from tensorflow.train import Checkpoint, CheckpointManager, latest_checkpoint
from tensorflow.keras.callbacks import ModelCheckpoint
import os
 
#1. 데이터 준비
x = np.array([1,2,3,4,5,6,7,8])
y = np.array([1,2,3,4,5,6,7,8])

print(x.shape)
print(x.dtype) 
 
#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
 
model = Sequential() 
model.add(Dense(5, input_dim=1, activation='linear')) 
model.add(Dense(3, activation='linear'))
model.add(Dense(4))
model.add(Dense(1))
 
cp_path = './DACON/cp/training-{epoch:04d}.ckpt'    # 가중치를 저장할 경로를 지정한다.
cp_dir = os.path.dirname(cp_path)                   # 가중치가 저장되어 있는 폴더 루트
cp = ModelCheckpoint(filepath=cp_path,            
    save_weights_only=True, verbose=0, period=5)    # callback에 들어갈 ModelCheckPoint

#3. 컴파일 & 훈련
model.compile(loss = 'mse', optimizer=Adam(0.01)) 
model.fit(x, y, epochs= 100, batch_size=1, callbacks=[cp])  

# checkpoint = Checkpoint(model)                      # 모델 가중치 저장
# save_path = checkpoint.save('./DACON/cp')         # 지정한 경로에 저장
# checkpoint.restore(save_path)                       # 모델로 복구시킴

latest = latest_checkpoint(cp_dir)                  # 가중치가 저장된 곳에서 가장 최근에 저장된 가중치를 불러온다.
print(">>>>>> latest : ", latest)
#      >>>>>> latest :  ./DACON/cp\training-0100.ckpt
model.load_weights(latest)                          # 최신 가중치를 load_weights 한다.

#4. 평가, 예측
loss = model.evaluate(x, y, batch_size=1) 
print('loss : ', loss)
 
result = model.predict([9])
print('result : ', result)

########################### 결과 비교 ###########################
# [1] 그냥 모델을 돌렸을 때,
# loss :  0.0
# result :  [[9.]]

# [2] checkpoint로 evaluate 했을 때, 
'''
[사용법]
checkpoint = Checkpoint(model)                      # 모델 가중치 저장
save_path = checkpoint.save('./DACON/cp')           # 지정한 경로에 저장
checkpoint.restore(save_path)                       # 모델로 복구시킴

[생성파일]
cp-1.data-0000-of-00001
cp-1.index

[result]
# loss :  9.947598300641403e-14
# result :  [[9.]]
'''

# [3] latest_checkpoint 로 evaluate 했을 때,
'''
[사용법]
cp_path = './DACON/cp/training-{epoch:04d}.ckpt'    # 가중치를 저장할 경로를 지정한다.
cp_dir = os.path.dirname(cp_path)                   # 가중치가 저장되어 있는 폴더 루트
cp = ModelCheckpoint(filepath=cp_path,            
    save_weights_only=True, verbose=0, period=5)    # callback에 들어갈 ModelCheckPoint

model.compile(loss = 'mse', optimizer=Adam(0.01)) 
model.fit(x, y, epochs= 100, batch_size=1, callbacks=[cp])  

latest = latest_checkpoint(cp_dir)                  # 가중치가 저장된 곳에서 가장 최근에 저장된 가중치를 불러온다.
print(">>>>>> latest : ", latest)
model.load_weights(latest)                          # 최신 가중치를 load_weights 한다.

[생성파일]
./cp/checkpoint
./cp/training-0005.ckpt.data-00000-of-00001
./cp/training-0005.ckpt.index
.
.
./cp/training-100번까지

[result]
# loss :  2.1316282072803006e-13
# result :  [[8.999998]]
'''