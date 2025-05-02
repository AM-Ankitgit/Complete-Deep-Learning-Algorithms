# %%
from keras.datasets.cifar10 import load_data

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
model

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

x = load_real_sample()
x[0]

# %%
def generate_real_samples(datasets,n_samples):
    ix = np.random.randint(0,datasets.shape[0],n_samples)

    X = datasets[ix]
    y = np.ones((n_samples,1))

    return X,y



x,y = generate_real_samples(x,64)
print(x.shape)
# %%

def generate_fake_samples():
    pass

# %%
