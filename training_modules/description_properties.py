# %%

# Function to print the number of distinct words and return the frequent-words list


def description_vocabulary(all_train_captions):

    # Setting the threshold for word to be regarded as frequent
    word_count_threshold = 10

    # Creating a dictionary mapping words to their frequency

    word_counts = {}
    nsents = 0
    for sent in all_train_captions:
        nsents += 1
        for w in sent.split(" "):
            word_counts[w] = word_counts.get(w, 0) + 1
    # Creating the frequent-words dataset and printing it's ddetails
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print("No. of words = ", len(word_counts))
    print("Length of vocabulary = ", len(vocab))
    return vocab


# %%

# Function to make a list for the description-lines of each key


def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc


# %%

# Function to return the length of the longest description of image


def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)
