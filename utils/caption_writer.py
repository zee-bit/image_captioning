import numpy as np
from keras.preprocessing.sequence import pad_sequences

# %%

# Caption creator function (using GreedySearch algorithm)


def greedySearch(photo, max_length, wordtoix, ixtoword, nlp_model):

    # Initializing caption sequence
    in_text = "startseq"

    # Searching for the predicted words and adding them into the 'in_text' seq
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = nlp_model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += " " + word
        if word == "endseq":
            break
    # Splitting the final sequence into individual words
    # Removing the first and last word
    final = in_text.split()
    final = final[1:-1]

    # Rejoining the final sequence and returning it
    final = " ".join(final)
    return final


# %%

# Caption Formatting function


def caption_format(caption):

    # Splitting the caption into words and resizing the length

    f_caption = caption.split()
    f_caption = f_caption[0:35]

    # Removing consecutive replications

    for i in range(35):
        if i + 1 < len(f_caption):
            if f_caption[i] == f_caption[i + 1]:
                del f_caption[i + 1]
    # Adding comma punctuation

    for i in range(35):
        if i + 2 < len(f_caption):
            if f_caption[i] == "and" and f_caption[i + 2] == "and":
                f_caption[i] = ","
    # Removing repitition of multi-word caption terms

    i = 0
    while i + 3 < len(f_caption):
        if f_caption[i] == f_caption[i + 2]:
            if f_caption[i + 1] == f_caption[i + 3]:
                f_caption[i + 2] = f_caption[i + 3] = "$"
        i += 1
    f_caption = list(filter(lambda a: a != "$", f_caption))

    i = 0
    while i + 5 < len(f_caption):
        if f_caption[i] == f_caption[i + 3]:
            if f_caption[i + 1] == f_caption[i + 4]:
                if f_caption[i + 2] == f_caption[i + 5]:
                    f_caption[i + 3] = f_caption[i + 4] = "$"
                    f_caption[i + 5] = "$"
                    f_caption = list(filter(lambda a: a != "$", f_caption))
                    i -= 3
        i += 1
    # Removing blank space adjacent to punctuations

    i = 0
    while i < len(f_caption):
        if f_caption[i] == ",":
            f_caption[i - 1] = f_caption[i - 1] + ","
            del f_caption[i]
            i -= 1
        i += 1
    # Removing the occurrence of auxiliary words as the last term

    aux_words = ["and", "or", "is", "was", "a", "an", "it", "of", "the"]
    end_word = f_caption[-1]
    if end_word in aux_words:
        del f_caption[-1]
    # Rejoining the caption and adding sentence formatting

    f_caption = " ".join(f_caption)
    f_caption = f_caption[0].upper() + f_caption[1:] + "."

    # Returning the formatted caption

    return f_caption
