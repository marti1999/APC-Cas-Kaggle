import copy
import os
import pickle
import sys
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np

def main():

    dir = './models'
    files = []
    for file in os.listdir(dir):
        if file.endswith(".sav"):

            files.append(os.path.join(dir, file))

    x_test, y_test = getProcessedData()

    x_test_bak = copy.deepcopy(x_test)

    for file in files:

        if 'PCA' in str(file):
            num =  int(''.join(filter(str.isdigit, str(file))))
            x_test = PCA(n_components=num).fit_transform(x_test)


        loaded_model = pickle.load(open(file, 'rb'))
        y_pred = loaded_model.predict(x_test)
        print('\n', file)
        print('RMSE: ', np.sqrt(mean_squared_error(y_test, y_pred))/2)
        x_test = x_test_bak


def detectOutliers(df, atributs, maxOutliers):
    # maxOutliers és el nombre màxim d'outliers permesos per mostra

    indexsOutliers = []

    # iterem sobre tots els atributs
    for atr in atributs:
        # primer quartil
        Q1 = np.percentile(df[atr], 25)
        # tercer quartil
        Q3 = np.percentile(df[atr], 75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1

        # zona de tall
        cutOff = 1.5 * IQR

        # Busquem els indexs dels registres fora de la zona de tall
        indexsOutliersAtr = df[(df[atr] < Q1 - cutOff) | (df[atr] > Q3 + cutOff)].index

        # Els guardem a la llista general
        indexsOutliers.extend(indexsOutliersAtr)

    # contem quantes vegades ha aparegut cada índex i seleccinem els que sobrepasen el límit especificat
    indexsOutliers = Counter(indexsOutliers)
    indexsDrop = list(k for k, v in indexsOutliers.items() if v > maxOutliers)

    return indexsDrop


def deleteRowsByIndex(df, indexs):
    rows = df.index[indexs]
    df.drop(rows, inplace=True)
    return df


def getProcessedData():
    db = pd.read_csv("./data/student-por.csv")
    db = db.sample(random_state=0, frac=1).reset_index(drop=True)
    db = db.head(40)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for c in db.columns:
        if db.dtypes[c] == object:
            le.fit(db[c].astype(str))
            db[c] = le.transform(db[c].astype(str))

    db = deleteRowsByIndex(db, detectOutliers(db, ["age", "Dalc", "Walc", "absences"], 0))
    y = db['G3']
    x = db.drop(columns='G3')
    return x, y



if __name__ == '__main__':
    main()




