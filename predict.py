import json
from keras.models import load_model
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
import collections


# Read the file tokens_clean.txt and store the cleaned captions in a dictionary
content = None
with open ("data/textFiles/tokens_clean.txt", 'r') as file:
    content = file.read()

json_acceptable_string = content.replace("'", "\"")
content = json.loads(json_acceptable_string)


total_words = []
for key in content.keys():
    for caption in content[key]:
        for i in caption.split():
            total_words.append(i)


# Compute the frequency of occurrence of each word
counter = collections.Counter(total_words)
freq_cnt = dict(counter)

# Sort the dictionary according to frequency of occurrence
sorted_freq_cnt = sorted(freq_cnt.items(), reverse=True, key=lambda x:x[1])

#Filter off those words which occur less than the threshold
threshold = 5
sorted_freq_cnt = [x for x in sorted_freq_cnt if x[1]>threshold]
total_words = [x[0] for x in sorted_freq_cnt]


print("Loading the model...")
model = load_model('model_checkpoints/model_19.h5')


test_encoding = {}
with open("encoded_test_features.pkl", "rb") as file:
    test_encoding = pd.read_pickle(file)


# Create the word-to-index and index-to-word mappings
word_to_index = {}
index_to_word = {}

for i, word in enumerate(total_words):
    word_to_index[word] = i+1
    index_to_word[i+1] = word

# Add startseq and endseq also to the mappings
index_to_word[2645] = 'startseq'
word_to_index['startseq'] = 2645

index_to_word[2646] = 'endseq'
word_to_index['endseq'] = 2646


# Generate Captions for a random image in test dataset
def predict_caption(photo):

    inp_text = "startseq"

    for i in range(38):
        sequence = [word_to_index[w] for w in inp_text.split() if w in word_to_index]
        sequence = pad_sequences([sequence], maxlen=38, padding='post')

        ypred = model.predict([photo, sequence])
        ypred = ypred.argmax()
        word = index_to_word[ypred]

        inp_text += (' ' + word)

        if word == 'endseq':
            break

    final_caption = inp_text.split()[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption



all_img_IDs = list(test_encoding.keys())

# Get a random image
number = np.random.randint(0, len(test_encoding))
img_ID = all_img_IDs[int(number)]
photo = test_encoding[img_ID].reshape((1, 2048))

print("Running model to genrate the caption...")
caption = predict_caption(photo)

img_data = plt.imread("data/Images/" + img_ID + ".jpg")
plt.imshow(img_data)
plt.axis("off")

plt.show()
print(caption)
