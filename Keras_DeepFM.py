from keras.layers import *
from keras.models import *
import keras.backend as K

class DeepFM():
    def __init__(self):
        self.cat_vars = ["Cat1", "Cat2", "Cat3", "Cat4", "Cat5"] # List of All Categorical Variables Names
        self.cat_levels =  [2, 5, 6, 3, 50] # List of All Categorical Variables Levels
        self.embd_dim = 8 # Size for Embedding Output
        self.hidden_layers = [500,100,20] # List of MLP hidden layers
        self.inputs = [Input(shape=(1,), name='Input_'+c) for c in self.cat_vars]
        self.embeddings_1st = []
        self.embeddings_2nd = []

    """
    FM - 1st order
    """
    def FM_1st_order(self):
        for input, c, i in zip(self.inputs, self.cat_vars, self.cat_levels):
            embedding = Embedding(i, 1)(input)
            self.embeddings_1st.append(embedding)
        res1 = Add()([Reshape((1,))(x) for x in self.embeddings_1st])
        return res1, self.embeddings_1st

    """
    FM - 2nd order
    """
    def FM_2nd_order(self):
        for input, c, i in zip(self.inputs, self.cat_vars, self.cat_levels):
            embedding = Embedding(i, self.embd_dim)(input)
            self.embeddings_2nd.append(embedding)

        concat = Concatenate(axis=1)(self.embeddings_2nd)
        square_sum = Lambda(lambda x: K.square(K.sum(x, axis=1)))(concat)
        sum_square = Lambda(lambda x: K.sum(x**2, axis=1))(concat)
        diff = Subtract()([square_sum, sum_square])
        res2 = Lambda(lambda x: 0.5 * K.sum(x, axis=1, keepdims=True))(diff)
        return res2, self.embeddings_2nd

    """
    MLP layer
    """
    def MLP(self):
        _, embeddings2 = self.FM_2nd_order()
        embeddings2 = Flatten()(Concatenate()(embeddings2))
        dense = Dense(self.hidden_layers[0], activation='relu')(embeddings2)
        for k in self.hidden_layers[1:]:
            dense = Dense(k, activation='relu')(dense)
        res3 = Dense(1, activation='relu')(dense)
        return res3

    """
    Final Output
    """
    def DeepFM_modeling(self):
        res1, _ = self.FM_1st_order()
        res2, _ = self.FM_2nd_order()
        res3 = self.MLP()

        y = Concatenate()([res1, res2, res3])
        y = Dense(1, activation='sigmoid')(y)

        model_DeepFM = Model(inputs=self.inputs, outputs=y)
        model_DeepFM.summary()

        return model_DeepFM
