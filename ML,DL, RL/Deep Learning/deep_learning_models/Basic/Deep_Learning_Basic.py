import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Flatten

# Hyperparameter
EPOCHS = 10

# Define Network architecture
class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.flatten = Flatten(input_shape = (28,28))
        self.d1 = Dense(128, activation = 'sigmoid')
        self.d2 = Dense(10, activation = 'softmax')
        
    def call(self,x):
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

# Definition Training Loop
@tf.function
def train_step(model,images,labels,loss_object, optimizer, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)

# Definition Test Loop
@tf.function
def test_step(model, images, labels, loss_object, test_loss, test_accuracy):
    predictions = model(images)
    
    t_loss = loss_object(labels,predictions)
    test_loss(t_loss)
    test_accuracy(labels,predictions)

# Organize Datasets

(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train[...,tf.newaxis]
x_test = x_test[...,tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(32)

# Generate Network
model = MyModel()

# Define loss and optimizer
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# Define perfonce metrics
train_loss = tf.keras.metrics.Mean(name = 'train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = 'train_accuracy')

test_loss = tf.keras.metrics.Mean(name = 'train_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = 'test_accuracy')

# Do training loop and test
for epoch in range(EPOCHS):
    for images, labels in train_ds:
        train_step(model,images ,labels, loss_object, optimizer, train_loss, train_accuracy)
        
    for test_images, test_labels in test_ds:
        test_step(model, test_images, test_labels, loss_object, test_loss, test_accuracy)
        
    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                         train_loss.result(),
                         train_accuracy.result() * 100,
                         test_loss.result(),
                         test_accuracy.result() * 100))

# Epoch 1, Loss: 0.3964761793613434, Accuracy: 89.74166870117188, Test Loss: 0.2317996770143509, Test Accuracy: 93.06999969482422
# Epoch 2, Loss: 0.2964911162853241, Accuracy: 92.01166534423828, Test Loss: 0.20004825294017792, Test Accuracy: 94.08000183105469
# Epoch 3, Loss: 0.24571916460990906, Accuracy: 93.2822265625, Test Loss: 0.17670167982578278, Test Accuracy: 94.79000091552734
# Epoch 4, Loss: 0.21238625049591064, Accuracy: 94.15374755859375, Test Loss: 0.1599213033914566, Test Accuracy: 95.25499725341797
# Epoch 5, Loss: 0.1880597025156021, Accuracy: 94.79299926757812, Test Loss: 0.14687398076057434, Test Accuracy: 95.6240005493164
# Epoch 6, Loss: 0.16912376880645752, Accuracy: 95.29972076416016, Test Loss: 0.13727198541164398, Test Accuracy: 95.91000366210938
# Epoch 7, Loss: 0.1538468599319458, Accuracy: 95.72000122070312, Test Loss: 0.1295967549085617, Test Accuracy: 96.11714172363281
# Epoch 8, Loss: 0.141142338514328, Accuracy: 96.07749938964844, Test Loss: 0.12370961904525757, Test Accuracy: 96.27749633789062
# Epoch 9, Loss: 0.13032162189483643, Accuracy: 96.38111114501953, Test Loss: 0.11825594305992126, Test Accuracy: 96.43111419677734
# Epoch 10, Loss: 0.12099989503622055, Accuracy: 96.64783477783203, Test Loss: 0.11414281278848648, Test Accuracy: 96.5510025024414