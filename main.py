import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sk



def main():
    db = pd.read_csv("./student-mat.csv")
    db.info()
    print(db.head())
    print(db.describe().T)

    sns.boxplot(data=db)
    #plt.show()
    plt.figure(figsize=(13, 13))
    sns.heatmap(db.corr(), fmt = '.2f', annot=True, cmap='rocket')
    plt.show()



if __name__ == "__main__":
    main()