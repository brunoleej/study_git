import tensorflow as tf

# Hyperparameter
EPOCHS = 10
NUM_WORDS = 10000   # 총 10000개의 단어만 사용함

# Definition Model
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.emb = tf.keras.layers.Embedding(NUM_WORDS, 16)
        self.rnn = tf.keras.layers.LSTM(32)
        self.dense = tf.keras.layers.Dense(2, activation='softmax')
        
    def call(self, x, training=None, mask=None):
        x = self.emb(x)
        x = self.rnn(x)
        return self.dense(x)

# Implement training loop
@tf.function
def train_step(model, inputs, labels, loss_object, optimizer, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)

# Implement algorithm test
@tf.function
def test_step(model, images, labels, loss_object, test_loss, test_accuracy):
    predictions = model(images, training=False)

    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)

imdb = tf.keras.datasets.imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=NUM_WORDS)

x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train,value=0,padding='pre',maxlen=32)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test,value=0,padding='pre', maxlen=32)
                                                      
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# Create model
model = MyModel()

# Define loss and optimizer
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# Define performance metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

for epoch in range(EPOCHS):
    for seqs, labels in train_ds:
        train_step(model, seqs, labels, loss_object, optimizer, train_loss, train_accuracy)

    for test_seqs, test_labels in test_ds:
        test_step(model, test_seqs, test_labels, loss_object, test_loss, test_accuracy)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100))
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

# Epoch 1, Loss: 0.4974110722541809, Accuracy: 74.93600463867188, Test Loss: 0.4401787519454956, Test Accuracy: 79.24400329589844
# Epoch 2, Loss: 0.36480146646499634, Accuracy: 83.71599578857422, Test Loss: 0.44863906502723694, Test Accuracy: 79.31200408935547
# Epoch 3, Loss: 0.29543331265449524, Accuracy: 87.48400115966797, Test Loss: 0.4975162148475647, Test Accuracy: 77.62799835205078
# Epoch 4, Loss: 0.23738345503807068, Accuracy: 90.44000244140625, Test Loss: 0.5495217442512512, Test Accuracy: 77.7040023803711
# Epoch 5, Loss: 0.18833665549755096, Accuracy: 92.73200225830078, Test Loss: 0.7544450163841248, Test Accuracy: 76.47200012207031
# Epoch 6, Loss: 0.1454068124294281, Accuracy: 94.45999908447266, Test Loss: 0.7594044208526611, Test Accuracy: 76.36000061035156
# Epoch 7, Loss: 0.11686915159225464, Accuracy: 95.56000518798828, Test Loss: 0.7879559993743896, Test Accuracy: 75.97600555419922
# Epoch 8, Loss: 0.08858171850442886, Accuracy: 96.66400146484375, Test Loss: 1.0141077041625977, Test Accuracy: 75.11199951171875
# Epoch 9, Loss: 0.07332263886928558, Accuracy: 97.36800384521484, Test Loss: 1.1462262868881226, Test Accuracy: 75.40399932861328
# Epoch 10, Loss: 0.05459706857800484, Accuracy: 97.97999572753906, Test Loss: 1.3181949853897095, Test Accuracy: 74.54399871826172