import os
import pickle

def main():
    dir = '../models'
    files = []
    for file in os.listdir("../models"):
        if file.endswith(".sav"):
            files.append(os.path.join(dir, file))
            print(os.path.join(dir, file))

    for file in files:
        loaded_model = pickle.load(open(file, 'rb'))


if __name__ == '__main__':
    main()