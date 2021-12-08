import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np


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
    db = pd.read_csv("../data/student-por.csv")
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for c in db.columns:
        if db.dtypes[c] == object:
            le.fit(db[c].astype(str))
            db[c] = le.transform(db[c].astype(str))

    db.loc[detectOutliers(db, ["age", "Dalc", "Walc", "absences"], 0)]
    db = deleteRowsByIndex(db, detectOutliers(db, ["age", "Dalc", "Walc", "absences"], 0))
    y = db['G3']
    x = db.drop(columns='G3')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=9)
    return x_train, x_test, y_train, y_test


def main():
    getProcessedData()


if __name__ == '__main__':
    main()