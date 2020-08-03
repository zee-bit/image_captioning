import string

# %%

# Function to load the captions dataset

def load_doc(filename):    
    file= open('./resources/results.csv','r',encoding='utf-8')
    text=file.read()
    file.close()
    return text

# %%

# Function to create a dictionary mapping training caption to training images
    
def load_descriptions():
    mapping=dict()
    text=load_doc('./resources/results.csv')
    for line in text.split('\n'):
        tokens=line.split('|')
        #print(tokens[0]," ",tokens[1:],end='\n')
        image_id,image_desc=tokens[0],tokens[-1]
        image_id=image_id.split('.')[0]
        
        if image_id not in mapping:
            mapping[image_id]=list()
        mapping[image_id].append(image_desc)
        
    return mapping

# %%

# Function to clean description strings
    
def clean_descriptions(descriptions):
    table=str.maketrans('','',string.punctuation)
    for key,desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc=desc_list[i]
            desc=desc.split()
            desc=[word.lower() for word in desc]
            desc=[w.translate(table) for w in desc]
            desc=[word for word in desc if len(word)>1]
            desc=[word for word in desc if word.isalpha()]
            desc_list[i]=' '.join(desc)
    return descriptions


# %%
    
# Function to load, process and return the processed descriptions
    
def load_final_descriptions():
    descriptions = load_descriptions()
    descriptions = clean_descriptions(descriptions)
    return descriptions

# %%

# Function to save local copy of cleaned description dictionary

def save_descriptions(descriptions,filename):
    lines=list()
    for key,desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key+' '+desc)
    data='\n'.join(lines)
    
    file=open(filename,'w')
    file.write(data)
    file.close()
    
# %%

# Function for loading dataset and appending all distinct lines to a set

def load_set(filename):
    doc=load_doc(filename)
    dataset=list()
    for line in doc.split('\n'):
        if len(line)<1:
            continue
        identifier=line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)

