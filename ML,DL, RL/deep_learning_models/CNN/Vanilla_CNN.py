# Basic Convolution NN avatar
import numpy as np
import tensorflow as tf

# Hyperparameter
EPOCHS = 10

# Model definition
class ConvNet(tf.keras.Model):
    def __init__(self):
        super(ConvNet,self).__init__()
        conv2d = tf.keras.layers.Conv2D
        maxpool = tf.keras.layers.MaxPool2D
        self.sequence = list()
        self.sequence.append(conv2d(16,(3,3),padding = "same", activation='relu')) # 28x28x16
        self.sequence.append(conv2d(16,(3,3),padding = "same", activation='relu')) # 28x28x16
        self.sequence.append(maxpool((2,2)))    # 14x14x16
        self.sequence.append(conv2d(32,(3,3),padding = "same", activation='relu')) # 14x14x32
        self.sequence.append(conv2d(32,(3,3),padding = "same", activation='relu')) # 14x14x32
        self.sequence.append(maxpool((2,2)))    # 7x7x32
        self.sequence.append(conv2d(64,(3,3),padding = "same", activation='relu')) # 7x7x64
        self.sequence.append(conv2d(64,(3,3),padding = "same", activation='relu')) # 7x7x64
        self.sequence.append(tf.keras.layers.Flatten()) # 1568
        self.sequence.append(tf.keras.layers.Dense(2048,activation = 'relu'))
        self.sequence.append(tf.keras.layers.Dense(10,activation = 'softmax'))

    def call(self,x,training = False,mask = None):
        for layer in self.sequence:
            x = layer(x)
        return x

# Train, Test Loop Definition
# Implement training loop
@tf.function
def train_step(model,images,labels,loss_object,optimizer,train_loss,train_accuracy):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels,predictions)
    gradients = tape.gradient(loss,model.trainable_variables)

    optimizer.apply_gradients(zip(gradients,model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)

# Implement algorithm test
@tf.function
def test_step(model, images, labels, loss_object, test_loss, test_accuracy):
    predictions = model(images)

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

# Epoch 1, Loss: 0.11067452281713486, Accuracy: 96.45833587646484, Test Loss: 0.05084584280848503, Test Accuracy: 98.5999984741211
# Epoch 2, Loss: 0.07744288444519043, Accuracy: 97.54916381835938, Test Loss: 0.04424263536930084, Test Accuracy: 98.78500366210938
# Epoch 3, Loss: 0.062123361974954605, Accuracy: 98.03666687011719, Test Loss: 0.04550863802433014, Test Accuracy: 98.76333618164062
# Epoch 4, Loss: 0.05343606323003769, Accuracy: 98.32416534423828, Test Loss: 0.04148157313466072, Test Accuracy: 98.8324966430664
# Epoch 5, Loss: 0.047125861048698425, Accuracy: 98.52466583251953, Test Loss: 0.038856275379657745, Test Accuracy: 98.87999725341797
# Epoch 6, Loss: 0.04253499582409859, Accuracy: 98.67694091796875, Test Loss: 0.037641946226358414, Test Accuracy: 98.92666625976562
# Epoch 7, Loss: 0.038901377469301224, Accuracy: 98.7933349609375, Test Loss: 0.037640493363142014, Test Accuracy: 98.95143127441406
# Epoch 8, Loss: 0.035987578332424164, Accuracy: 98.88416290283203, Test Loss: 0.03756728395819664, Test Accuracy: 98.97249603271484
# Epoch 9, Loss: 0.03365529328584671, Accuracy: 98.96148681640625, Test Loss: 0.03710494562983513, Test Accuracy: 98.99555206298828
# Epoch 10, Loss: 0.031660955399274826, Accuracy: 99.02300262451172, Test Loss: 0.03964250162243843, Test Accuracy: 98.98200225830078