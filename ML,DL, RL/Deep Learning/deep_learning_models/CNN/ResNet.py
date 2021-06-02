# Residual Neural Network
import numpy as np
import tensorflow as tf

# Hyperparameter
EPOCHS = 10

# Residual Unit avatar
class ResidualUnit(tf.keras.Model):
    def __init__(self,fiter_in, filter_out, kernel_size):
        super(ResidualUnit,self).__init__()
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv2D(filter_out,kernel_size,padding = "SAME")
        
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filter_out,kernel_size,padding = "SAME")
    
        if filter_in == filter_out:
            self.ientity = lambda x: x
        else:
            self.ientity = tf.keras.layers.Conv2D(filter_out,(1,1),padding = 'same')

    def call(self,x,training = False,mask=None):
        h = self.bn1(x,traning,training)
        h = tf.nn.relu(h)
        h = self.conv1(h)

        h = self.bn2(h,traning,training)
        h = tf.nn.relu(h)
        h = self.conv2(h)
        return self.identity(x) + h

# Residual acvatar
class ResnetLayer(tf.keras.Model):
    def __init__(self,filter_in,filters,kernel_size):
        super(ResnetLayer,self).__init__()
        self.sequence = list()
        for f_in,f_out in zip([filter_in] + list(filters),filters):
            # [16] + [32,32,32]
            # zip([16,32,32,32],[32,32,32])
            self.sequence.append(ResidualUnit(f_in,f_out,kernel_size))
    def call(self,x,training = False,mask = None):
        for unit in self.sequence:
            x = unit(x,training)
        return x
# Model Definition
class ResNet(tf.keras.Model):
    def __init__(self):
        super(ResNet,self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(8,(3,3),padding = "same",activation = 'relu') # 28 x 28 x 8
        
        self.res2 = ResnetLayer(8,(16,16),(3,3)) # 28x28x16
        self.pool2 = tf.keras.layers.MaxPool2D((2,2)) # 14x14x16

        self.res3 = ResnetLayer(8,(32,32),(3,3)) # 14x14x32
        self.pool3 = tf.keras.layers.MaxPool2D((2,2)) # 7x7x32

        self.res3 = ResnetLayer(8,(64,64),(3,3)) # 7x7x64
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128,activation = 'relu')
        self.dense2 = tf.keras.layers.Dense(10,activation = 'relu')

    def call(self,x,training = False, mask = None):
        x = self.conv1(x)

        x = self.res1(x,training = training)
        x = self.pool1(x)
        x = self.res2(x,training = training)
        x = self.pool2(x)
        x = self.res3(x,training = training)

        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)
        
# Train, Test Loop Definition
# Implement training loop
@tf.function
def train_step(model,images,labels,loss_object,optimizer,train_loss,train_accuracy):
    with tf.GradientTape() as tape:
        predictions = model(images,training = True)
        loss = loss_object(labels,predictions)
    gradients = tape.gradient(loss,model.trainable_variables)

    optimizer.apply_gradients(zip(gradients,model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)

# Implement algorithm test
@tf.function
def test_step(model, images, labels, loss_object, test_loss, test_accuracy):
    predictions = model(images,training = False)

    t_loss = loss_object(labels,predictions)
    test_loss(t_loss)
    test_accuracy(labels,predictions)

# Prepare Datasets
mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train,x_test = x_train / 255.0, x_test / 255.0

# x_train : (NUM_SAMPLE,28,28) -> (NUM_SAMPLE,28,28,1)
x_train = x_train[...,tf.newaxis].astype(np.float32)
x_test = x_test[...,tf.newaxis].astype(np.float32)

train_ds = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(32)

# Train environment Definition
# Model Generate, Loss Function, Optimizer, Accuracy
# Create Model
model = ConvNet()

# Define loss and optimizer
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# Define performance metrics
train_loss = tf.keras.metrics.Mean(name = 'train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = 'train_accuracy')

test_loss = tf.keras.metrics.Mean(name = 'test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = 'test_accuracy')

# train loop start
for epoch in range(EPOCHS):
    for images, labels in train_ds:
        train_step(model, images, labels, loss_object,optimizer,train_loss,train_accuracy)

    for test_images, test_labels in test_ds:
        test_step(model,test_images,test_labels,loss_object,test_loss,test_accuracy)
    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1, 
    train_loss.result(),
    train_accuracy.result()*100,
    test_loss.result(),
    test_accuracy.result() * 100))

# Epoch 1, Loss: 0.1108165830373764, Accuracy: 96.586669921875, Test Loss: 0.07494485378265381, Test Accuracy: 97.48999786376953
# Epoch 2, Loss: 0.07826791703701019, Accuracy: 97.5816650390625, Test Loss: 0.056176669895648956, Test Accuracy: 98.14500427246094
# Epoch 3, Loss: 0.06288620084524155, Accuracy: 98.06222534179688, Test Loss: 0.04927448183298111, Test Accuracy: 98.39666748046875
# Epoch 4, Loss: 0.0542098730802536, Accuracy: 98.34249877929688, Test Loss: 0.04704597219824791, Test Accuracy: 98.48249816894531
# Epoch 5, Loss: 0.048128753900527954, Accuracy: 98.5296630859375, Test Loss: 0.04334200546145439, Test Accuracy: 98.62200164794922
# Epoch 6, Loss: 0.043288614600896835, Accuracy: 98.67888641357422, Test Loss: 0.04143429920077324, Test Accuracy: 98.69499969482422
# Epoch 7, Loss: 0.03989275544881821, Accuracy: 98.78309631347656, Test Loss: 0.04095447435975075, Test Accuracy: 98.72285461425781
# Epoch 8, Loss: 0.03691549971699715, Accuracy: 98.87603759765625, Test Loss: 0.04065656289458275, Test Accuracy: 98.7562484741211
# Epoch 9, Loss: 0.034648969769477844, Accuracy: 98.94721984863281, Test Loss: 0.039980120956897736, Test Accuracy: 98.78555297851562
# Epoch 10, Loss: 0.032605528831481934, Accuracy: 99.01083374023438, Test Loss: 0.0390433669090271, Test Accuracy: 98.82099914550781