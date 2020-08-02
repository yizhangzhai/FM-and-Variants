from keras.layers import *
from keras.models import *
import keras.backend as K

class FiledAware_Factorization_Machine():
    def __init__(self):
        self.cat_vars = ["Cat1", "Cat2", "Cat3", "Cat4", "Cat5"] # List of All Categorical Variables Names
        self.cat_levels =  [2, 5, 6, 3, 50] # List of All Categorical Variables Levels
        self.embd_dim = 8 # Size for Embedding Output
        self.inputs = [Input(shape=(1,), name='Input_'+c) for c in self.cat_vars]
        self.embeddings_1st = []
        self.embeddings_2nd_dic = {}

    """
    FFM - 1st order
    """
    def FFM_1st_order(self):
        for input, c, i in zip(self.inputs, self.cat_vars, self.cat_levels):
            embedding = Embedding(i, 1, name='Embedding_1st_'+c)(input)
            self.embeddings_1st.append(embedding)
        res1 = Add()([Reshape((1,))(x) for x in self.embeddings_1st])
        return res1, self.embeddings_1st

    """
    FFM - 2nd order
    """
    def FFM_2nd_order(self):
        I, J = [], []
        for i in range(len(self.cat_vars)):
            for j in range(len(self.cat_vars)):
                if i!=j:
                    I.append(i)
                    J.append(j)

        for p in I:
            self.embeddings_2nd_dic[p]={}
            for q in J:
                self.embeddings_2nd_dic[p][q]=Embedding(self.cat_levels[p], self.embd_dim)(self.inputs[p])

        cross_mul = []
        for i, j in zip(I, J):
            cross_mul.append(Multiply()([self.embeddings_2nd_dic[i][j], self.embeddings_2nd_dic[j][i]]))
        cross_mul = Concatenate(axis=1)(cross_mul)
        res2 = Lambda(lambda x: K.sum(x, axis=1))(cross_mul)
        res2 = Lambda(lambda x: K.sum(x, axis=1, keepdims=True))(res2)

        return res2, self.embeddings_2nd_dic

    """
    Final Output
    """
    def FFM_modeling(self):
        res1, _ = self.FFM_1st_order()
        res2, _ = self.FFM_2nd_order()
        y = Concatenate()([res1, res2])
        y = Dense(1, activation='sigmoid')(y)

        model_FFM = Model(inputs=self.inputs, outputs=y)
        model_FFM.summary()

        return model_FFM

ffm = FiledAware_Factorization_Machine()
ffm.FFM_modeling()
