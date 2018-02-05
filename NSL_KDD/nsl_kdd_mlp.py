import keras
from keras.layers import *
from keras.models import Sequential

batch_size = 100
epochs = 3

print('Loading data...')
with np.load("./NSL_KDD.npz") as f:
    x_train, y_train, x_test, y_test = f["x_train"], f["y_train"], f["x_test"], f["y_test"]
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

print('Building model...')
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(41,)))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
verify_with_test_data = False # 是否使用验证集进行验证
if verify_with_test_data:
    print("verify with test data")
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_data=(x_test, y_test))
else:
    print("verify with train self data")
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_split=0.8)
