from keras.layers import *
from keras.models import *
import keras.backend as K

class Wide_Deep():
    def __init__(self):
        self.cat_vars = ["Cat1", "Cat2", "Cat3", "Cat4", "Cat5"] # List of All Categorical Variables Names
        self.cat_levels =  [2, 5, 6, 3, 50] # List of All Categorical Variables Levels
        self.embd_dim = 8 # Size for Embedding Output
        self.hidden_layers = [500,100,20] # List of MLP hidden layers
        self.inputs = [Input(shape=(1,), name='Input_'+c) for c in self.cat_vars]
        self.embeddings_1st = []
        self.embeddings_2nd = []

    """
    WD - Wide
    """
    def Wide(self):
        for input, c, i in zip(self.inputs, self.cat_vars, self.cat_levels):
            embedding = Embedding(i, 1)(input)
            self.embeddings_1st.append(embedding)
        res1 = Add()([Reshape((1,))(x) for x in self.embeddings_1st])
        return res1, self.embeddings_1st

    """
    WD - Deep
    """
    def Deep(self):
        for input, c, i in zip(self.inputs, self.cat_vars, self.cat_levels):
            embedding = Embedding(i, self.embd_dim)(input)
            self.embeddings_2nd.append(embedding)

        embeddings2 = Flatten()(Concatenate()(self.embeddings_2nd))
        dense = Dense(self.hidden_layers[0], activation='relu')(embeddings2)
        for k in self.hidden_layers[1:]:
            dense = Dense(k, activation='relu')(dense)
        res2 = Dense(1, activation='relu')(dense)
        return res2, self.embeddings_2nd

    """
    Final Output
    """
    def WD_modeling(self):
        res1, _ = self.Wide()
        res2, _ = self.Deep()
        y = Concatenate()([res1, res2])
        y = Dense(1, activation='sigmoid')(y)

        model_WD = Model(inputs=self.inputs, outputs=y)
        model_WD.summary()

        return model_WD
