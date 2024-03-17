# %% libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

# %% Organizando o dataframe

Dataframe = pd.read_csv('data/Dataframe2.csv')

Dataframe = Dataframe.loc[:, ~Dataframe.columns.isin(['ID', 'Zclass'])]
Dataframe = Dataframe.loc[:, ~Dataframe.columns.isin(['RA', 'DEC'])]

colnames = list(Dataframe.columns)

z = ['Z']
info = ['PhotoFlagDet', 'nDet_PStotal']

erros = list(filter(lambda i: i[0] == 'e', colnames))
diferencas = list(filter(lambda i: '.' in i, colnames))

for i in diferencas:
    Dataframe['e_' + i] = np.sqrt(
        Dataframe['e_' + i.split('.')[0]] ** 2 + Dataframe['e_' + i.split('.')[1]] ** 2
    )

del colnames, diferencas, erros, i, info, z

Dataframe.to_csv('data/Dataframe_m.csv')


# %% index dictionary


class frame:
    def __init__(self, df,
                 set_size=[0.5, 0.25, 0.25], seed=2023, sample_size=1.00):

        df = df.sample(frac=sample_size)

        if 0 == 0:
            df = df[df.PhotoFlagDet == 0]

        df_train, df_val = train_test_split(df, test_size=(1 - set_size[0]),
                                            random_state=seed)
        df_test = df_val

        if len(set_size) == 3:
            df_val, df_test = train_test_split(df_val, test_size=set_size[2] / (1 - set_size[0]),
                                               random_state=seed)

        self.train = df_train

        self.val = df_val

        self.test = df_test

        self.df = df

        colnames = list(df.columns)
        colnames.remove('PhotoFlagDet')
        colnames.remove('nDet_PStotal')
        colnames.remove('Z')

        self.index = {
            "z": ['Z'],

            "info": ['PhotoFlagDet', 'nDet_PStotal'],

            "g": list(filter(lambda i: '.' not in i and
                                       'e' not in i and
                                       'e_' not in i and
                                       'mag' in i, colnames)),

            "e_g": list(filter(lambda i: '.' not in i and
                                         'e' in i and
                                         'e_' in i and
                                         'mag' in i, colnames)),

            "j": list(filter(lambda i: '.' not in i and
                                       'e' not in i and
                                       'e_' not in i and
                                       'PS' in i, colnames)),

            "e_j": list(filter(lambda i: '.' not in i and
                                         'e' in i and
                                         'e_' in i and
                                         'PS' in i, colnames)),

            "u": list(filter(lambda i: '.' not in i and
                                       'e' not in i and
                                       'e_' not in i and
                                       'MAG' in i, colnames)),

            "e_u": list(filter(lambda i: '.' not in i and
                                         'e' in i and
                                         'e_' in i and
                                         'MAG' in i, colnames)),

            "f": list(filter(lambda i: '.' not in i and
                                       'e' not in i and
                                       'e_' not in i, colnames)),

            "e_f": list(filter(lambda i: '.' not in i and
                                         'e' in i and
                                         'e_' in i, colnames)),

            "d": list(filter(lambda i: '.' in i and
                                       'e' not in i and
                                       'e_' not in i, colnames)),

            "e_d": list(filter(lambda i: '.' in i and
                                         'e' in i and
                                         'e_' in i, colnames))
        }

    def features(self, features):

        columns = []

        for i in features:
            columns += self.index[i]

        return {
            "complete": self.df.loc[:, self.df.columns.isin(columns)],
            "train": self.train.loc[:, self.train.columns.isin(columns)],
            "val": self.val.loc[:, self.val.columns.isin(columns)],
            "test": self.test.loc[:, self.test.columns.isin(columns)]
        }
