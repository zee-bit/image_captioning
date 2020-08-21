import numpy as np
from keras.preprocessing.sequence import pad_sequences

#%%

# Caption creator function (using GreedySearch algorithm)
def greedySearch(photo, max_length, wordtoix, ixtoword, nlp_model):

    # Initializing caption sequence
    in_text = "startseq"

    # Searching for the predicted words and adding them into the 'in_text' sequence (Requires 'wordtoix.pkl' and 'ixtoword.pkl')
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = nlp_model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += " " + word
        if word == "endseq":
            break
    # Splitting the final sequence into individual words and removing the first and last word
    final = in_text.split()
    final = final[1:-1]

    # Rejoining the final sequence and returning it
    final = " ".join(final)
    return final
