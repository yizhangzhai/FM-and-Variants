from keras.layers import *
from keras.models import *
import keras.backend as K
import keras.activations
import tensorflow as tf

class first_order_term(Layer):
    def __init__(self):
        super(first_order_term, self).__init__()

    def build(self, input_shape):
        self.w = self.add_weight(name='w_1st',
                                 shape=(1, 1,input_shape[2]),
                                 initializer='uniform',
                                 trainable=True)

        self.b = self.add_weight(name='b_1st',
                                 shape=(1,),
                                 initializer='uniform',
                                 trainable=True)
    def call(self, x):
        return Lambda(lambda x: K.sum(x, keepdims=True))(self.b + K.squeeze((Dot(axes=2)([x, self.w])), axis=2))


class Attention_layer(Layer):
    def __init__(self, att_factor):
        self.att_factor = att_factor
        super(Attention_layer, self).__init__()

    def build(self, input_shape):
        self.w = self.add_weight(name='w_att',
                                 shape=(input_shape[2],self.att_factor),
                                 initializer='uniform',
                                 trainable=True)

        self.b = self.add_weight(name='b_att',
                                 shape=(self.att_factor,),
                                 initializer='uniform',
                                 trainable=True)

        self.h = self.add_weight(name='h_att',
                                 shape=(self.att_factor,1),
                                 initializer='uniform',
                                 trainable=True)

    def call(self, e):
        temp = keras.activations.relu(K.squeeze(self.b + Multiply()([e, self.w]), axis=1))
        return Dot(axes=1)([temp, self.h])

class Attention_FM():
    def __init__(self):
        self.cat_vars = ["Cat1", "Cat2", "Cat3", "Cat4", "Cat5"] # List of All Categorical Variables Names
        self.cat_levels =  [2, 5, 6, 3, 50] # List of All Categorical Variables Levels
        self.embd_dim = 8 # Size for Embedding Output
        self.inputs = [Input(shape=(1,), name='Input_'+c) for c in self.cat_vars]
        self.embeddings_1st = [Embedding(c, 1)(self.inputs[i]) for i,c in enumerate(self.cat_levels)]
        self.embeddings_2nd = [Embedding(c, self.embd_dim)(self.inputs[i]) for i,c in enumerate(self.cat_levels)]
        self.att_factor = 11

    def att_scored(self):
        I, J = [], []
        for i in range(len(self.cat_vars)):
            for j in range(i+1, len(self.cat_vars)):
                I.append(i), J.append(j)

        M_IJ = []
        for i, j in zip(I, J):
            M_IJ.append(Multiply()([self.embeddings_2nd[i], self.embeddings_2nd[j]]))

        att_scores = []
        att = Attention_layer(self.att_factor)
        for Z in M_IJ:
            att_scores.append(K.exp(att(Z)))
        att_scores = Lambda(lambda x: x/K.sum(att_scores, keepdims=True))(att_scores)
        att_scores = Reshape((len(self.cat_vars)*(len(self.cat_vars)-1)//2,1))(att_scores)

        res2 = Multiply()([att_scores, Concatenate(axis=1)(M_IJ)])
        res2 = Lambda(lambda x: K.sum(x, axis=1))(res2)
        res2 = Dense(1, activation='linear')(res2)

        return res2

    def combine(self):
        first = first_order_term()
        res1 = first(Concatenate(axis=1)(embeddings_1st))
        res2 = self.att_scored()
        return keras.activations.sigmoid(res1 + res2)
