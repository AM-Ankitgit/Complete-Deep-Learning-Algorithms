# %%
from keras.datasets.cifar10 import load_data
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:

    tf.config.experimental.set_memory_growth(gpu, True)





gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled for:", gpus)
    except RuntimeError as e:
        print("RuntimeError:", e)

# %%
import pandas as pd


# %%
import matplotlib.pyplot as plt

# %%
(trainX,trainy),(testX,testy) = load_data()

# %%
trainX[0].shape

# %%
for i in range(10):
    plt.subplot(5,2,i+1)
    plt.axis("off")
    plt.imshow(trainX[i])
plt.show()

# %%
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LeakyReLU
# from keras.utils.vis_utils import plot_model
from keras.utils import plot_model
import numpy as np



# %%
def define_discrimintor(in_shape=(32, 32, 3)):
    model  = Sequential()

    model.add(Conv2D(64,(3,3),padding='same',input_shape = in_shape)) # total 64 filter with size 3 * 3
    model.add(LeakyReLU(alpha=0.2))
    
    # downsample

    model.add(Conv2D(128,(3,3),strides=(2,2),padding='same'))
    model.add(LeakyReLU(alpha=0.2))


     # downsample

    model.add(Conv2D(128,(3,3),strides=(2,2),padding='same'))
    model.add(LeakyReLU(alpha=0.2))


    # downsample

    model.add(Conv2D(128,(3,3),strides=(2,2),padding='same'))
    model.add(LeakyReLU(alpha=0.2))


    # classifier

    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1,activation='sigmoid'))

    opt = Adam(learning_rate=0.0002,beta_1 =0.5)

    model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
    return model
# %%
model = define_discrimintor()

# %%
# model.train_on_batch(X_real,Y_real)

# %%
model.summary()

# %%
plot_model(model,to_file ='descriminator_arch.png',show_shapes=True,show_layer_names=True)
# %%
def load_real_sample():
    (trainX,_),(_,_) = load_data()

    x = trainX.astype('float32')
    x = (x-127.5)/127.5

    return x

# %%

x_real_load = load_real_sample()
x_real_load[0]
plt.imshow(x_real_load[0])
# %%
def generate_real_samples(datasets,n_samples):
    ix = np.random.randint(0,datasets.shape[0],n_samples)

    X = datasets[ix]
    y = np.ones((n_samples,1))

    return X,y



x_real,y_real = generate_real_samples(x_real_load,64)

# model.train_on_batch(x_real,y_real) # if you get error : DNN library is not found.” Problem with tensorflow then you need to install
# python3 -m pip install 'tensorflow[and-cuda]'
# print(x_real.shape)
# plt.imshow(x_real[0])





# %%

def generate_fake_samples(n_samples):
    X = np.random.rand(32 * 32 *3 *n_samples)
    X = -1 + X *2

    X = X.reshape(n_samples,32,32,3)

    Y = np.zeros((n_samples,1))
    return X,Y


# %%
x_fake,y_fake = generate_fake_samples(64)


print(x_fake.shape)
print(y_fake)
# %%

plt.imshow(x_fake[0])
# %%
plt.imshow(x_fake[1])


# %%
plt.imshow(x_fake[2])
# %%
def train_discriminator(model,dataset,n_iter=20,_n_batch=128):
    half_batch = int(_n_batch/2)

    for i in range(n_iter):

        # get randomly selected real sample
        X_real,Y_real = generate_real_samples(datasets=dataset,n_samples=half_batch)

        # update discriminator on real samples

        _,real_acc = model.train_on_batch(X_real,Y_real)

        # generate_fake_samples
        X_fake,Y_fake = generate_fake_samples(half_batch)

        _,fake_acc = model.train_on_batch(X_fake,Y_fake)

        print("%d real=%.0f%% fake=%.0f%%" %(i+1,real_acc*100,fake_acc*100))


# %%
model = define_discrimintor()

dataset = load_real_sample()

train_discriminator(model,dataset)
# %%
train_discriminator(model,dataset)

# %%
train_discriminator(model,dataset)
# %%
