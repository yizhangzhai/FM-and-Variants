from keras.models import *
from keras.layers import *
import keras.backend as K

class Cross(Layer):
    def __init__(self, x0):
        super(Cross, self).__init__()
        self.x0 = x0

    def build(self, input_shape):
        self.w = self.add_weight(name='kernel',
                                 shape=(1, input_shape[1], input_shape[2]),
                                 initializer='uniform',
                                 trainable=True)

        self.b = self.add_weight(name='kernel',
                                 shape=(1,),
                                 initializer='uniform',
                                 trainable=True)

    def call(self, xL):
        temp = Dot(axes=2)([self.x0, xL])
        temp = Dot(axes=1)([temp, self.w])
        return temp + self.b + xL


class Cross_Deep():
    def __init__(self):
        self.cat_vars = ["Cat1", "Cat2", "Cat3", "Cat4", "Cat5"] # List of All Categorical Variables Names
        self.cat_levels =  [2, 5, 6, 3, 50] # List of All Categorical Variables Levels
        self.embd_dim = 8 # Size for Embedding Output
        self.inputs = [Input(shape=(1,), name='Input_'+c) for c in cat_vars]
        self.embeddings_1st = [Embedding(c, embd_dim)(inputs[i]) for i,c in enumerate(cat_levels)]
        self.cross_depth = 10
        self.hidden = [30,20,10]

    X0 = Concatenate(axis=1)(embeddings_1st)

    def cross(self):
        c = Cross(X0)
        X = c(X0)
        for i in range(self.cross_depth):
            X = c(X)
        return Flatten()(X)

    def deep(self):
        dense = Flatten()(X0)
        for j in self.hidden:
            dense = Dense(j, activation='relu')(dense)
        return dense

    def combine(self):
        res_cross = self.cross()
        res_deep  = self.deep()
        res_stack = Concatenate()([res_cross, res_deep])
        res = Dense(1, activation='sigmoid')(res_stack)

        return res
