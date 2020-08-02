from keras.layers import *
from keras.models import *
import keras.backend as K
from keras import activations

class Factorization_Machine():
    def __init__(self):
        self.cat_vars = ["Cat1", "Cat2", "Cat3", "Cat4", "Cat5"] # List of All Categorical Variables Names
        self.cat_levels =  [2, 5, 6, 3, 50] # List of All Categorical Variables Levels
        self.embd_dim = 8 # Size for Embedding Output
        self.inputs = [Input(shape=(1,), name='Input_'+c) for c in self.cat_vars]
        self.embeddings_1st = []
        self.embeddings_2nd = []

    """
    FM - 1st order
    """
    def FM_1st_order(self):
        for input, c, i in zip(self.inputs, self.cat_vars, self.cat_levels):
            embedding = Embedding(i, 1, name='Embedding_1st_'+c)(input)
            self.embeddings_1st.append(embedding)
        res1 = Add()([Reshape((1,))(x) for x in self.embeddings_1st])
        return res1, self.embeddings_1st

    """
    FM - 2nd order
    """
    def FM_2nd_order(self):
        for input, c, i in zip(self.inputs, self.cat_vars, self.cat_levels):
            embedding = Embedding(i, self.embd_dim, name='Embedding_2nd_'+c)(input)
            self.embeddings_2nd.append(embedding)

        concat = Concatenate(axis=1)(self.embeddings_2nd)
        square_sum = Lambda(lambda x: K.square(K.sum(x, axis=1)))(concat)
        sum_square = Lambda(lambda x: K.sum(x**2, axis=1))(concat)
        diff = Subtract()([square_sum, sum_square])
        res2 = Lambda(lambda x: 0.5 * K.sum(x, axis=1, keepdims=True))(diff)
        return res2, self.embeddings_2nd

    """
    Final Output
    """
    def FM_modeling(self):
        res1, _ = self.FM_1st_order()
        res2, _ = self.FM_2nd_order()
        y = Add()([res1, res2])
        y = Lambda(lambda x: activations.sigmoid(x))(y)

        model_FM = Model(inputs=self.inputs, outputs=y)
        model_FM.summary()

        return model_FM
