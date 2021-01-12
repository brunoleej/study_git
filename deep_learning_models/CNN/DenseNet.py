# DenseNet
import numpy as np
import tensorflow as tf

# Hyperparameter
EPOCHS = 10

# DenseNet Unit avatar
class DenseUnit(tf.keras.Model):
    def __init__(self,filter_out,kernel_size):
        super(DenseUnit,self).__init__()
        self.bn = tf.keras.layers.BatchNormalization()
        self.conv = tf.keras.layers.Conv2D(filter_out,kernel_size, padding = 'SAME')
        self.concat = tf.keras.layers.Concatenate()
    
    def call(self,x,training = False,mask = None):  # x: (Batch,H,W,Ch_in)
        h = slef.bn(x,training = training)
        h = tf.nn.relu(h)
        h = self.conv(h)    # h: (Batch, H, W, filter_output)
        return self.concat([x,h])   # (Batch,H,W,(Ch_in + filter_output))

# Dense Layer avatar
class DenseLayer(tf.keras.Model):
    def __init__(self,num_unit,growth_rate,kernel_size):
        super(DenseLayer,self).__init__()
        self.sequence = list()
        for idx in range(num_unit):
            self.sequence.append(DenseUnit(growth_rate,kernel_size))

    def call(self,training = Fale, mask=None):
        for unit in self.sequence:
            x = unit(x,training = training)
        return x

# Transition Layer avatar
class TransitionLayer(tf.keras.Model):
    def __init__(self,filters,kernel_size):
        super(TransitionLayer,self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters,kernel_size, padding = 'SAME')
        self.pool = tf.keras.layers.MaxPool2D(2,2)
        

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
