import numpy as np
from sklearn.datasets import load_svmlight_file
import os

def preprocess_data():
    if os.path.exists('sorted.libsvm'):
        pass
    else:
        origin_data = open("20news.libsvm", "r")
        doc_data = origin_data.readlines()

        write_list = []
        for i, doc in enumerate(doc_data):
            doc = doc.split("\t")[1]
            word_list = []
            count_list = []
            for word in doc.split(" "):
                word = word.split(":")
                if len(word)==2:
                    word_list.append(int(word[0]))
                    count_list.append(int(word[-1]))
            index = np.array(word_list).argsort()
            word_array = np.array(word_list)[index]
            count_array = np.array(count_list)[index]
            line_sorted = [f"{word_array[i]}:{count_array[i]}" for i in range(word_array.shape[0])]
            line_sorted_write = f"{i}"+ "\t" + " ".join(line_sorted) + "\n"
            write_list.append(line_sorted_write)

        file = open('sorted.libsvm','w')
        for write_line in write_list:
            file.write(write_line)
        file.close()

def preprocess_dict():
    origin_data = open("20news.vocab", "r")
    vocab_data = origin_data.readlines()
    vocab_name_list = []
    vocab_count_list = []
    for word in vocab_data:
        word = word.split("\n")[0]
        _, name, count = word.split("\t")
        vocab_name_list.append(name)
        vocab_count_list.append(count)
    return vocab_name_list, vocab_count_list

def read_data():
    data = load_svmlight_file("sorted.libsvm")
    return data[0].toarray(), data[1]



import scipy.special as sc
def cal_coeff(X, Nd):
    if os.path.exists("Coeff.txt"):
        Coeff = np.loadtxt("Coeff.txt")
    else:
        Coeff = np.sum(sc.gammaln(X+1), axis=1) - sc.gammaln(Nd+1)
        np.savetxt('Coeff.txt', Coeff)
    Coeff = Coeff[:,np.newaxis]
    return Coeff