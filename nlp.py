import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
train_data = pd.read_csv('tamil_news_train.csv.xls')
test_data = pd.read_csv('tamil_news_test.csv.xls')
train_data
train_data['Category'].value_counts()
test_data['Category'].value_counts()
plt.figure(figsize=(8,6))
sns.countplot(train_data.Category)
plt.figure(figsize=(8,6))
sns.countplot(test_data.Category)
punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

def remove_punc(data):
    result = []

    for test_str in data.NewsInTamil:
        for ele in test_str:
            if ele in punc:
                test_str = test_str.replace(ele, "")
        result.append(test_str)
    
    return result
train_res = remove_punc(train_data)
train_data['WithoutPunc'] = train_res
train_data
test_res = remove_punc(test_data)
test_data['WithoutPunc'] = test_res
test_data
def get_category_as_vector(data):
    tn, ind, cin, sp, pol, wo = [], [], [], [], [], []
    for i in data['Category']:
        if i == 'tamilnadu':
            tn.append(1)
            ind.append(0)
            cin.append(0)
            sp.append(0)
            pol.append(0)
            wo.append(0)
        elif i == 'india':
            tn.append(0)
            ind.append(1)
            cin.append(0)
            sp.append(0)
            pol.append(0)
            wo.append(0)
        elif i == 'cinema':
            tn.append(0)
            ind.append(0)
            cin.append(1)
            sp.append(0)
            pol.append(0)
            wo.append(0)
        elif i == 'sports':
            tn.append(0)
            ind.append(0)
            cin.append(0)
            sp.append(1)
            pol.append(0)
            wo.append(0)
        elif i == 'politics':
            tn.append(0)
            ind.append(0)
            cin.append(0)
            sp.append(0)
            pol.append(1)
            wo.append(0)
        else:
            tn.append(0)
            ind.append(0)
            cin.append(0)
            sp.append(0)
            pol.append(0)
            wo.append(1)
    
    return tn, ind, cin, sp, pol, wo
tn, ind, cin, sp, pol, wo = get_category_as_vector(train_data)
train_data['tn'] = tn
train_data['ind'] = ind
train_data['cin'] = cin
train_data['sp'] = sp
train_data['pol'] = pol
train_data['wo'] = wo

train_data
tn, ind, cin, sp, pol, wo = get_category_as_vector(test_data)
test_data['tn'] = tn
test_data['ind'] = ind
test_data['cin'] = cin
test_data['sp'] = sp
test_data['pol'] = pol
test_data['wo'] = wo

test_data
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout, LSTM, Bidirectional
max_len = 50 
trunc_type = "post" 
padding_type = "post" 
from keras.preprocessing.text import Tokenizer
import pickle

tokenize = Tokenizer(num_words = 5000, char_level=False, oov_token = "<OOV>")
tokenize.fit_on_texts(train_data.WithoutPunc)

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenize, handle, protocol=pickle.HIGHEST_PROTOCOL)

word_index = tokenize.word_index
word_index
training_sequenc = tokenize.texts_to_sequences(train_data.WithoutPunc)
training_padd = pad_sequences(training_sequenc, maxlen = max_len, padding = padding_type, truncating = trunc_type)
testing_sequenc = tokenize.texts_to_sequences(test_data.WithoutPunc)
testing_padd = pad_sequences(testing_sequenc, maxlen = max_len, padding = padding_type, truncating = trunc_type)
training_padd
training_padd.shape
testing_padd.shape
predict_msg1=['சென்னையில் பெண் கொல்லப்பட்டார்']
predict_msg2=['டெல்லியில் பூகம்பம்']
predict_msg3=['நடிகர் போதைப்பொருள் பயன்படுத்தி பிடிபட்டார்']
predict_msg4=['கால்பந்து ரசிகர்களை இழந்து வருகிறது']
predict_msg5=['சுயேச்சை அமைச்சரை டிஸ்மிஸ்']
predict_msg6=['மலேசிய விமானம் காணவில்லை']
n_lstm = 20
drop_lstm =0.2
vocab_s = 5000
embed_dim = 16
drp_vale = 0.2 
n_dense = 24
tn_lstm = Sequential()
tn_lstm.add(Embedding(vocab_s, embed_dim, input_length=max_len))
tn_lstm.add(LSTM(n_lstm, dropout=drop_lstm, return_sequences=True))
tn_lstm.add(LSTM(n_lstm, dropout=drop_lstm, return_sequences=True))
tn_lstm.add(Dense(1, activation='sigmoid'))
tn_lstm.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])

num_epochs = 50
#early_stop = EarlyStopping(monitor='val_loss', patience=2)
history = tn_lstm.fit(training_padd, train_data['tn'], epochs=num_epochs, validation_data=(testing_padd, test_data['tn']), verbose=2)
metrics = pd.DataFrame(history.history)
# Rename column
metrics.rename(columns = {'loss': 'Training_Loss', 'accuracy': 'Training_Accuracy',
                         'val_loss': 'Validation_Loss', 'val_accuracy': 'Validation_Accuracy'}, inplace = True)
def plot_graphs1(var1, var2, string):
    metrics[[var1, var2]].plot()
    plt.title('LSTM Model: Training and Validation ' + string)
    plt.xlabel ('Number of epochs')
    plt.ylabel(string)
    plt.legend([var1, var2])
plot_graphs1('Training_Loss', 'Validation_Loss', 'loss')
plot_graphs1('Training_Accuracy', 'Validation_Accuracy', 'accuracy')
ind_lstm = Sequential()
ind_lstm.add(Embedding(vocab_s, embed_dim, input_length=max_len))
ind_lstm.add(LSTM(n_lstm, dropout=drop_lstm, return_sequences=True))
ind_lstm.add(LSTM(n_lstm, dropout=drop_lstm, return_sequences=True))
ind_lstm.add(Dense(1, activation='sigmoid'))
ind_lstm.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])

num_epochs = 50
#early_stop = EarlyStopping(monitor='val_loss', patience=2)
history = ind_lstm.fit(training_padd, train_data['ind'], epochs=num_epochs, validation_data=(testing_padd, test_data['ind']), verbose=2)

metrics = pd.DataFrame(history.history)
# Rename column
metrics.rename(columns = {'loss': 'Training_Loss', 'accuracy': 'Training_Accuracy',
                         'val_loss': 'Validation_Loss', 'val_accuracy': 'Validation_Accuracy'}, inplace = True)
def plot_graphs1(var1, var2, string):
    metrics[[var1, var2]].plot()
    plt.title('LSTM Model: Training and Validation ' + string)
    plt.xlabel ('Number of epochs')
    plt.ylabel(string)
    plt.legend([var1, var2])

plot_graphs1('Training_Loss', 'Validation_Loss', 'loss')
plot_graphs1('Training_Accuracy', 'Validation_Accuracy', 'accuracy')
wo_lstm = Sequential()
wo_lstm.add(Embedding(vocab_s, embed_dim, input_length=max_len))
wo_lstm.add(LSTM(n_lstm, dropout=drop_lstm, return_sequences=True))
wo_lstm.add(LSTM(n_lstm, dropout=drop_lstm, return_sequences=True))
wo_lstm.add(Dense(1, activation='sigmoid'))
wo_lstm.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])

num_epochs = 50
#early_stop = EarlyStopping(monitor='val_loss', patience=2)
history = wo_lstm.fit(training_padd, train_data['wo'], epochs=num_epochs, validation_data=(testing_padd, test_data['wo']), verbose=2)
metrics = pd.DataFrame(history.history)
# Rename column
metrics.rename(columns = {'loss': 'Training_Loss', 'accuracy': 'Training_Accuracy',
                         'val_loss': 'Validation_Loss', 'val_accuracy': 'Validation_Accuracy'}, inplace = True)
def plot_graphs1(var1, var2, string):
    metrics[[var1, var2]].plot()
    plt.title('LSTM Model: Training and Validation ' + string)
    plt.xlabel ('Number of epochs')
    plt.ylabel(string)
    plt.legend([var1, var2])

plot_graphs1('Training_Loss', 'Validation_Loss', 'loss')
plot_graphs1('Training_Accuracy', 'Validation_Accuracy', 'accuracy')
sp_lstm = Sequential()
sp_lstm.add(Embedding(vocab_s, embed_dim, input_length=max_len))
sp_lstm.add(LSTM(n_lstm, dropout=drop_lstm, return_sequences=True))
sp_lstm.add(LSTM(n_lstm, dropout=drop_lstm, return_sequences=True))
sp_lstm.add(Dense(1, activation='sigmoid'))
sp_lstm.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])

num_epochs = 50
#early_stop = EarlyStopping(monitor='val_loss', patience=2)
history = sp_lstm.fit(training_padd, train_data['sp'], epochs=num_epochs, validation_data=(testing_padd, test_data['sp']), verbose=2)
metrics = pd.DataFrame(history.history)
# Rename column
metrics.rename(columns = {'loss': 'Training_Loss', 'accuracy': 'Training_Accuracy',
                         'val_loss': 'Validation_Loss', 'val_accuracy': 'Validation_Accuracy'}, inplace = True)
def plot_graphs1(var1, var2, string):
    metrics[[var1, var2]].plot()
    plt.title('LSTM Model: Training and Validation ' + string)
    plt.xlabel ('Number of epochs')
    plt.ylabel(string)
    plt.legend([var1, var2])

plot_graphs1('Training_Loss', 'Validation_Loss', 'loss')
plot_graphs1('Training_Accuracy', 'Validation_Accuracy', 'accuracy')
pol_lstm = Sequential()
pol_lstm.add(Embedding(vocab_s, embed_dim, input_length=max_len))
pol_lstm.add(LSTM(n_lstm, dropout=drop_lstm, return_sequences=True))
pol_lstm.add(LSTM(n_lstm, dropout=drop_lstm, return_sequences=True))
pol_lstm.add(Dense(1, activation='sigmoid'))
pol_lstm.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])

num_epochs = 50
#early_stop = EarlyStopping(monitor='val_loss', patience=2)
history = pol_lstm.fit(training_padd, train_data['pol'], epochs=num_epochs, validation_data=(testing_padd, test_data['pol']), verbose=2)
metrics = pd.DataFrame(history.history)
# Rename column
metrics.rename(columns = {'loss': 'Training_Loss', 'accuracy': 'Training_Accuracy',
                         'val_loss': 'Validation_Loss', 'val_accuracy': 'Validation_Accuracy'}, inplace = True)
def plot_graphs1(var1, var2, string):
    metrics[[var1, var2]].plot()
    plt.title('LSTM Model: Training and Validation ' + string)
    plt.xlabel ('Number of epochs')
    plt.ylabel(string)
    plt.legend([var1, var2])

plot_graphs1('Training_Loss', 'Validation_Loss', 'loss')
plot_graphs1('Training_Accuracy', 'Validation_Accuracy', 'accuracy')
cin_lstm = Sequential()
cin_lstm.add(Embedding(vocab_s, embed_dim, input_length=max_len))
cin_lstm.add(LSTM(n_lstm, dropout=drop_lstm, return_sequences=True))
cin_lstm.add(LSTM(n_lstm, dropout=drop_lstm, return_sequences=True))
cin_lstm.add(Dense(1, activation='sigmoid'))
cin_lstm.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])

num_epochs = 50
#early_stop = EarlyStopping(monitor='val_loss', patience=2)
history = cin_lstm.fit(training_padd, train_data['cin'], epochs=num_epochs, validation_data=(testing_padd, test_data['cin']), verbose=2)
metrics = pd.DataFrame(history.history)
# Rename column
metrics.rename(columns = {'loss': 'Training_Loss', 'accuracy': 'Training_Accuracy',
                         'val_loss': 'Validation_Loss', 'val_accuracy': 'Validation_Accuracy'}, inplace = True)
def plot_graphs1(var1, var2, string):
    metrics[[var1, var2]].plot()
    plt.title('LSTM Model: Training and Validation ' + string)
    plt.xlabel ('Number of epochs')
    plt.ylabel(string)
    plt.legend([var1, var2])

plot_graphs1('Training_Loss', 'Validation_Loss', 'loss')
plot_graphs1('Training_Accuracy', 'Validation_Accuracy', 'accuracy')
cin_lstm.save('cin_lstm.h5')
sp_lstm.save('sp_lstm.h5')
pol_lstm.save('pol_lstm.h5')
wo_lstm.save('wo_lstm.h5')
ind_lstm.save('ind_lstm.h5')
tn_lstm.save('tn_lstm.h5')
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
max_len = 50
trunc_type = "post"
padding_type = "post"

wo_lstm_path = 'wo_lstm.h5'
ind_lstm_path = 'ind_lstm.h5'
cin_lstm_path = 'cin_lstm.h5'
pol_lstm_path = 'pol_lstm.h5'
sp_lstm_path = 'sp_lstm.h5'
tn_lstm_path = 'tn_lstm.h5'

handle = open('tokenizer.pickle', 'rb')
tokenize = pickle.load(handle)
# with open('models/tokenizer.pickle', 'rb') as handle:
    # tokenize = pickle.load(handle)

wo_lstm = keras.models.load_model(wo_lstm_path)
ind_lstm = keras.models.load_model(ind_lstm_path)
cin_lstm = keras.models.load_model(cin_lstm_path)
pol_lstm = keras.models.load_model(pol_lstm_path)
sp_lstm = keras.models.load_model(sp_lstm_path)
tn_lstm = keras.models.load_model(tn_lstm_path)
def predict_wo(predict_msg):
    new_seq = tokenize.texts_to_sequences(predict_msg)
    padded = pad_sequences(new_seq, maxlen =max_len,
                      padding = padding_type,
                      truncating=trunc_type)
    res = wo_lstm.predict(padded)
    return np.average(res)

def predict_ind(predict_msg):
    new_seq = tokenize.texts_to_sequences(predict_msg)
    padded = pad_sequences(new_seq, maxlen =max_len,
                      padding = padding_type,
                      truncating=trunc_type)
    # return (ind_lstm.predict(padded))
    res = ind_lstm.predict(padded)
    return np.average(res)

def predict_tn(predict_msg):
    new_seq = tokenize.texts_to_sequences(predict_msg)
    padded = pad_sequences(new_seq, maxlen =max_len,
                      padding = padding_type,
                      truncating=trunc_type)
    # return (tn_lstm.predict(padded))
    res = tn_lstm.predict(padded)
    return np.average(res)

def predict_sp(predict_msg):
    new_seq = tokenize.texts_to_sequences(predict_msg)
    padded = pad_sequences(new_seq, maxlen =max_len,
                      padding = padding_type,
                      truncating=trunc_type)
    # return (tn_lstm.predict(padded))
    res = sp_lstm.predict(padded)
    return np.average(res)

def predict_pol(predict_msg):
    new_seq = tokenize.texts_to_sequences(predict_msg)
    padded = pad_sequences(new_seq, maxlen =max_len,
                      padding = padding_type,
                      truncating=trunc_type)
    # return (tn_lstm.predict(padded))
    res = pol_lstm.predict(padded)
    return np.average(res)

def predict_cin(predict_msg):
    new_seq = tokenize.texts_to_sequences(predict_msg)
    padded = pad_sequences(new_seq, maxlen =max_len,
                      padding = padding_type,
                      truncating=trunc_type)
    # return (tn_lstm.predict(padded))
    res = cin_lstm.predict(padded)
    return np.average(res)

predict_msg1=['சென்னையில் பெண் கொல்லப்பட்டார்'] # TN News
predict_msg2=['டெல்லியில் பூகம்பம்'] # India News
predict_msg3=['நடிகர் போதைப்பொருள் பயன்படுத்தி பிடிபட்டார்'] # Cinema News
predict_msg4=['கால்பந்து ரசிகர்களை இழந்து வருகிறது'] # Sports News
predict_msg5=['சுயேச்சை அமைச்சரை டிஸ்மிஸ்'] # Politics News
predict_msg6=['மலேசிய விமானம் காணவில்லை'] # World News
print(predict_wo(predict_msg1))
print(predict_wo(predict_msg2))
print(predict_wo(predict_msg3))
print(predict_wo(predict_msg4))
print(predict_wo(predict_msg5))
print(predict_wo(predict_msg6))
print(predict_ind(predict_msg1))
print(predict_ind(predict_msg2))
print(predict_ind(predict_msg3))
print(predict_ind(predict_msg4))
print(predict_ind(predict_msg5))
print(predict_ind(predict_msg6))
print(predict_tn(predict_msg1))
print(predict_tn(predict_msg2))
print(predict_tn(predict_msg3))
print(predict_tn(predict_msg4))
print(predict_tn(predict_msg5))
print(predict_tn(predict_msg6))
print(predict_sp(predict_msg1))
print(predict_sp(predict_msg2))
print(predict_sp(predict_msg3))
print(predict_sp(predict_msg4))
print(predict_sp(predict_msg5))
print(predict_sp(predict_msg6))
print(predict_pol(predict_msg1))
print(predict_pol(predict_msg2))
print(predict_pol(predict_msg3))
print(predict_pol(predict_msg4))
print(predict_pol(predict_msg5))
print(predict_pol(predict_msg6))
print(predict_cin(predict_msg1))
print(predict_cin(predict_msg2))
print(predict_cin(predict_msg3))
print(predict_cin(predict_msg4))
print(predict_cin(predict_msg5))
print(predict_cin(predict_msg6))
category_list = ["World news", "India News", "Cinema News", "Politics News", "Sports News", "TN News"]

def show(input_msg):

    wo_res = predict_wo(input_msg)
    ind_res = predict_ind(input_msg)
    cin_res = predict_cin(input_msg)
    pol_res = predict_pol(input_msg)
    sp_res = predict_sp(input_msg)
    tn_res = predict_tn(input_msg)
    res = [wo_res, ind_res, cin_res, pol_res, sp_res, tn_res]
    # for path in lstm_list:
        # val = predict(input_msg, path)
        # res.append(val)

    idx = 0
    max_val = -1
    for i in range(len(res)):
        if res[i] > max_val:
            max_val = res[i]
            idx = i

    print(res)
    print(res[idx])
    category = category_list[idx]
    print(category)
    show(predict_msg1)
show(predict_msg2)
show(predict_msg3)
show(predict_msg4)
show(predict_msg5)
show(predict_msg6)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.layers import SpatialDropout1D
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.feature_selection import RFE
import re
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
train = pd.read_csv('tamil_news_train.csv.xls')
test = pd.read_csv('tamil_news_test.csv.xls')
# fix random seed for reproducibility
np.random.seed(7)
train = train.drop_duplicates().reset_index(drop=True)
test = test.drop_duplicates().reset_index(drop=True)
train.NewsInTamil = train.NewsInTamil.str.replace('\d+', ' ')

test.NewsInTamil = test.NewsInTamil.str.replace('\d+', ' ')

    
train = train.append(test)
df = train
df.head()
df.shape
df.Category.unique()
df.Category = df.Category.replace('world', 1)
df.Category = df.Category.replace('cinema', 2)
df.Category = df.Category.replace('tamilnadu', 3)
df.Category = df.Category.replace('india', 4)
df.Category = df.Category.replace('politics', 5)
df.Category = df.Category.replace('sports', 6)


df.Category.head()
# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 32000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 120
# This is fixed.
EMBEDDING_DIM = 100
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=False)
tokenizer.fit_on_texts(df.NewsInTamil.values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
X = tokenizer.texts_to_sequences(df.NewsInTamil.values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)
Y = pd.get_dummies(df.Category).values
print('Shape of label tensor:', Y.shape)
train_features, test_features, train_labels, test_labels = train_test_split(X, Y, test_size=.10)
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(6, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(train_features, train_labels, epochs=5, batch_size=32,validation_split=0.2)
# Final evaluation of the model
model_pred_train = model.predict(train_features)
model_pred_test = model.predict(test_features)
# print(classification_report(test_labels,model_pred_test))
print('LSTM Recurrent Neural Network baseline: ' + str(roc_auc_score(train_labels, model_pred_train)))
print('LSTM Recurrent Neural Network: ' + str(roc_auc_score(test_labels, model_pred_test)))
model.summary()
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()
news = ['இயற்கையை நேசிப்பதுதானே கொண்டாட்டம்.. இது ஒரு புது முயற்சி..!']
seq = tokenizer.texts_to_sequences(news)
padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
pred = model.predict(padded)
labels = ['world', 'cinema', 'tamilnadu', 'india', 'politics', 'sports']
label = pred, labels[np.argmax(pred)]
print("News Category is: ")
print(label[1])
predict_msg1=['சென்னையில் பெண் கொல்லப்பட்டார்']
predict_msg2=['டெல்லியில் பூகம்பம்']
predict_msg3=['நடிகர் போதைப்பொருள் பயன்படுத்தி பிடிபட்டார்']
predict_msg4=['கால்பந்து ரசிகர்களை இழந்து வருகிறது']
predict_msg5=['சுயேச்சை அமைச்சரை டிஸ்மிஸ்']
predict_msg6=['மலேசிய விமானம் காணவில்லை']
def find_predict(news):
    seq = tokenizer.texts_to_sequences(news)
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    pred = model.predict(padded)
    labels = ['world', 'cinema', 'tamilnadu', 'india', 'politics', 'sports']
    label = pred, labels[np.argmax(pred)]
    print(label[1])

find_predict(predict_msg1)
find_predict(predict_msg2)
find_predict(predict_msg3)
find_predict(predict_msg4)
find_predict(predict_msg5)
find_predict(predict_msg6)

model.save('lstm_baseline.h5')
from tensorflow import keras

classifier = keras.models.load_model('lstm_baseline.h5')
def find_predict(news):
    seq = tokenizer.texts_to_sequences(news)
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    pred = classifier.predict(padded)
    labels = ['world', 'cinema', 'tamilnadu', 'india', 'politics', 'sports']
    label = pred, labels[np.argmax(pred)]
    print(label[1])

predict_msg1=['சென்னையில் பெண் கொல்லப்பட்டார்']
predict_msg2=['டெல்லியில் பூகம்பம்']
predict_msg3=['நடிகர் போதைப்பொருள் பயன்படுத்தி பிடிபட்டார்']
predict_msg4=['கால்பந்து ரசிகர்களை இழந்து வருகிறது']
predict_msg5=['சுயேச்சை அமைச்சரை டிஸ்மிஸ்']
predict_msg6=['மலேசிய விமானம் காணவில்லை']

find_predict(predict_msg1)
find_predict(predict_msg2)
find_predict(predict_msg3)
find_predict(predict_msg4)
find_predict(predict_msg5)
find_predict(predict_msg6)

import gradio as gr
def find_predic(news):
    l=[news]
    seq = tokenizer.texts_to_sequences(l)
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    pred = classifier.predict(padded)
    labels = ['world', 'cinema', 'tamilnadu', 'india', 'politics', 'sports']
    label = pred, labels[np.argmax(pred)]
    return label[1]
iface=gr.Interface(fn=find_predic,inputs="text",outputs="text")
iface.launch()
import googletrans
from googletrans import Translator
translator=Translator()
out=translator.translate("சென்னையில் பெண் கொல்லப்பட்டார்",dest="en")
print(out.text)