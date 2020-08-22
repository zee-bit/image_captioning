import numpy as np
from numpy import array
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from pickle import dump

# %%

# Function to initialize a generator for training and optimizing the weights


def data_generator(
    descriptions, photos, wtix, max_length, num_photos_per_batch, v_size
):
    X1, X2, y = list(), list(), list()
    n = 0

    while 1:
        for key, desc_list in descriptions.items():
            n += 1
            photo = photos[key + ".jpg"]
            for desc in desc_list:
                seq = [wtix[word] for word in desc.split(" ") if word in wtix]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]

                    out_seq = to_categorical([out_seq], num_classes=v_size)[0]

                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)
            if n == num_photos_per_batch:
                yield [[array(X1), array(X2)], array(y)]
                X1, X2, y = list(), list(), list()
                n = 0


# %%

# Function to create and save locally the mapping dictionaries


def get_mapping_dicts(vocab):
    ixtoword = {}
    wordtoix = {}
    ix = 1
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1
    with open("./resources/ixtoword.pkl", "wb") as f1:
        dump(ixtoword, f1)
        f1.close()
    with open("./resources/wordtoix.pkl", "wb") as f2:
        dump(wordtoix, f2)
        f2.close()
    return (ixtoword, wordtoix)


# %%

# Function to import the 'glove' word-set and storing it in 'embedding_index'


def get_glove_wordset():
    glove_dir = "./resources/glove.6B.200d.txt"
    embedding_index = {}

    f = open(glove_dir, encoding="utf-8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embedding_index[word] = coefs
    f.close()
    return embedding_index


# %%

# Function to make a matrix of common words in 'glove' and wordtoix dict


def get_embedding_matrix(embedding_dim, wordtoix, vocab_size):
    embedding_index = get_glove_wordset()
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for word, i in wordtoix.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
