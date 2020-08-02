from keras.models import *
from keras.layers import *
import keras.backend as K
import tensorflow as tf

class Kernel_layer(Layer):
    def __init__(self, kernel_type, kernel_shape):
        super(Kernel_layer, self).__init__()
        self.w = self.add_weight(name='kernel',
                                 shape=kernel_shape,
                                 initializer='uniform',
                                 trainable=True)
    def call(self,input):
        x, y = input[0], input[1]
        if kernel_type == 'mat':
            temp = Lambda(lambda z: K.sum(z,axis=-1))(K.expand_dims(x, axis=1) * self.w)
            temp = Permute((2,1))(temp)
            return Lambda(lambda z: K.sum(z,axis=-1))(temp * y)
        else:
            return Lambda(lambda z: K.sum(z,axis=-1))(x * y * K.expand_dims(self.w, axis=0))

class PNN():
    def __init__(self):
        self.cat_vars = ["Cat1", "Cat2", "Cat3", "Cat4", "Cat5"] # List of All Categorical Variables Names
        self.cat_levels =  [2, 5, 6, 3, 50] # List of All Categorical Variables Levels
        self.embd_dim = 8 # Size for Embedding Output
        self.inputs = [Input(shape=(1,), name='Input_'+c) for c in self.cat_vars]
        self.embeddings_1st = [Embedding(c,self.embd_dim)(inputs[i]) for i,c in enumerate(self.cat_levels)]
        self.embeddings_2nd = []
        self.kernel_type = 'mat'
        self.mul_type = 'inner'

    def Linear(self):
        temp = Concatenate()(embeddings_1st)
        out_1st = Lambda(lambda x: K.sum(x, axis=1))(temp)
        out_1st = Dense(len(self.cat_vars)*self.embd_dim, activation='linear')(out_1st)
        return out_1st

    def Inner_prod(self):
        I, J = [], []
        for i in range(len(self.cat_vars)):
            for j in range(i+1, len(self.cat_vars)):
                I.append(i)
                J.append(j)

        embeddings_1st_I = [embeddings_1st[i] for i in I]
        embeddings_1st_J = [embeddings_1st[j] for j in J]
        e_IJ = []
        for e_I, e_J in zip(embeddings_1st_I, embeddings_1st_J):
            e_IJ.append(Lambda(lambda x: K.sum(x,axis=2))(Multiply()([e_I, e_J])))
        out = Concatenate()(e_IJ)
        return out

    def Outter_prod(self):
        if kernel_type == 'mat':
           kernel_shape = self.embd_dim, len(self.cat_vars)*(len(self.cat_vars)-1)//2, self.embd_dim
        elif kernel_type == 'vec':
           kernel_shape = len(self.cat_vars)*(len(self.cat_vars)-1)//2, self.embd_dim
        elif kernel_type == 'num':
           kernel_shape = len(self.cat_vars)*(len(self.cat_vars)-1)//2, 1

        I, J = [], []
        for i in range(len(self.cat_vars)):
            for j in range(i+1, len(self.cat_vars)):
                I.append(i)
                J.append(j)
        embeddings_1st_I = Concatenate(axis=1)([embeddings_1st[i] for i in I])
        embeddings_1st_J = Concatenate(axis=1)([embeddings_1st[j] for j in J])

        kn = Kernel_layer(kernel_type, kernel_shape)
        out = kn([embeddings_1st_I,embeddings_1st_J])
        return out

    def combine(self):
        out_1st = self.Linear()
        if self.mul_type == 'inner':
           out_2nd = self.Inner_prod()
        else:
           out_2nd = self.Outter_prod()
        out_temp = Concatenate()([out_1st, out_2nd])
        dense = BatchNormalization()(out_temp)
        dense = Dense(K.int_shape(out_temp)[1], activation='relu')(dense)
        dense = Dense(K.int_shape(out_temp)[1]//2, activation='relu')(dense)
        out = Dense(1, activation='sigmoid')(dense)
        return out
