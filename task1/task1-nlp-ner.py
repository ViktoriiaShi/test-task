#!/usr/bin/env python
# coding: utf-8

# In[28]:


import spacy
import pandas as pd
import random
from spacy.training.example import Example


# the dataset of mountains found on kaggle. the model architecture chosen for NER solving is spacy pre trained transformer based model en_core_web_trf

# In[29]:


# reading the csv file with mountains
def load_mountain_names_from_csv(csv_file):
    data = pd.read_csv("/Users/vikavika/Desktop/Top montains data.csv")
    mountain_names = data['Mountain name(s)'].tolist()  
    return mountain_names


# In[30]:


def create_train_data(mountain_names):
    train_data = []
    
    for name in mountain_names:
        # create a simple sentence with the mountain name
        text = f"The {name} is a famous mountain."
        
        # mark the mountain name as an entity (start, end, label)
        start = text.find(name)
        end = start + len(name)
        entities = [(start, end, "MOUNTAIN")]
        
        # append the sentence and entity annotation to the training data
        train_data.append((text, {"entities": entities}))
    
    return train_data


# In[31]:


def train_ner_model(csv_file, output_dir="mountain_ner_model", epochs=20):
    # Load the pre-trained model
    nlp = spacy.load("en_core_web_trf")
    
    # Create a blank NER pipe or use the existing one
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    mountain_names = load_mountain_names_from_csv(csv_file)
    train_data = create_train_data(mountain_names)

    # Add the new entity label to the NER pipe
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # Disable other pipes during training (to only update the NER)
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}")
            random.shuffle(train_data)  # shuffle data each epoch for better training
            losses = {}
            
            # training on each example
            for text, annotations in train_data:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update([example], drop=0.5, losses=losses)
            
            print(f"Losses at epoch {epoch + 1}: {losses}")

    # saving the trained model
    nlp.to_disk(output_dir)
    print(f"Model saved to '{output_dir}'.")


# In[32]:


# test
def test_model(model_path="mountain_ner_model"):
    nlp = spacy.load(model_path)
    
    text = "The Himalayas include Mount Everest and K2."
    doc = nlp(text)
    
    for ent in doc.ents:
        print(f"Entity: {ent.text}, Label: {ent.label_}")

train_ner_model("/Users/vikavika/Desktop/Top montains data.csv")
test_model()  


# tried to fix it, package is installed. didn't find a solution yet. need more time.

# # we also can create our own dataset using python and pandas. example 1:

# In[24]:


# Example data (list of dictionaries, each dictionary is a row)
data = [
    {"id": 1, "mountain": "Hoverla"},
    {"id": 2, "mountain": "Pikui"},
    {"id": 3, "mountain": "PipIvan"},
    {"id": 4, "mountain": "Everest"}
]

# Create a pandas DataFrame
df = pd.DataFrame(data)

# Display the dataset
print(df)

# Save the dataset to a CSV file
df.to_csv("mount_data.csv", index=False)


# # example 2:

# In[23]:


data = [
    ("Mount Everest is the highest mountain in the world.", [(0, 12, "MOUNTAIN")]),
    ("K2 is located in the Karakoram range.", [(0, 2, "MOUNTAIN")]),
    ("Mount Fuji is an iconic symbol of Japan.", [(0, 9, "MOUNTAIN")]),
    ("The Alps are famous for skiing.", [(4, 8, "MOUNTAIN")]),
    ("Mount Kilimanjaro is in Tanzania.", [(0, 17, "MOUNTAIN")]),
]

# Create the DataFrame
df = pd.DataFrame(data, columns=["text", "entities"])

# Display the dataset
print(df)

# Save the dataset to CSV
df.to_csv("mountain_ner_data.csv", index=False)


# In[ ]:




