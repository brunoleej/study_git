import numpy as np
import tensorflow as tf

EPOCHS = 100
batch_size = 1

# Network Architecture
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel,self).__init__()
        self.dense1 = tf.keras.layers.Dense(1024,activation = 'relu')
        self.dense2 = tf.keras.layers.Dense(512,activation = 'relu')
        self.dense3 = tf.keras.layers.Dense(216,activation = 'relu')
        self.dense4 = tf.keras.layers.Dense(112,activation = 'relu')
        self.dense5 = tf.keras.layers.Dense(1)
        

    def call(self,x,training = False,mask= None):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        return self.dense5(x)
# Data
x = np.arange(1,11)
y = np.arange(1,11)

# Optimizer
optimizer = tf.keras.optimizers.Adadelta(lr=0.01)

# Model
model = MyModel()
model.compile(loss='mse',optimizer=optimizer,metrics = ['mae'])
model.fit(x,y,epochs = EPOCHS,batch_size=batch_size)

# Predict
loss,mae = model.evaluate(x,y,batch_size=batch_size)
y_pred = model.predict([11])
print('loss:',loss)      
print('mae:',mae)                  
print('prediction: ',y_pred)

# Adam(lr=0.1)
# loss: 8.942119598388672
# mae: 2.566387414932251
# prediction:  [[6.3319373]]

# Adam(lr=0.01)
# loss: 5.208562470215838e-06
# mae: 0.0019437909359112382
# prediction: [[10.996722]]

# Adam(lr=0.001)
# loss: 0.0023726222570985556
# mae: 0.04442998021841049
# prediction: [[10.917852]]

# Adam(lr=0.0001)
# loss: 0.00016421967302449048
# mae: 0.010614776983857155
# prediction: [[10.986686]]

# Adam(lr=0.00001)
# loss: 0.0049919793382287025
# mae: 0.059057533740997314
# prediction: [[10.911114]]

# Adadelta(lr=0.1)
# loss: 0.004177263006567955
# mae: 0.05680052191019058
# prediction:  [[10.883793]]

# Adadelta(lr=0.01)
# loss: 0.0032607335597276688
# mae: 0.04751534387469292
# prediction:  [[10.937571]]

# RMSprop(lr=0.1)
# loss: 8.250371932983398
# mae: 2.5
# prediction:  [[5.480712]]