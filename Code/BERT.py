
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout
from keras.regularizers import l2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping 

import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils


from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)
import torch
from transformers import DistilBertTokenizer, DistilBertModel, BertTokenizer, BertModel, BertForMaskedLM

# Load pre-trained model tokenizer (vocabulary)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

import pandas as pd
df = pd.read_csv('commit_only.csv', encoding='iso-8859-1')

df.head()

labels = df['3_labels']

print(labels.value_counts())

tokenized_text = df['comment'].apply((lambda x: tokenizer.encode(x, max_length=128, add_special_tokens=True, truncation=True)))

import numpy as np
max_len = 0
for i in tokenized_text.values:
    if len(i) > max_len:
        max_len = len(i)

padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized_text.values])

np.array(padded).shape

attention_mask = np.where(padded != 0, 1, 0)
attention_mask.shape

model = DistilBertModel.from_pretrained('distilbert-base-uncased')
input_ids = torch.tensor(padded)  
attention_mask = torch.tensor(attention_mask)

with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)
    
    
features = last_hidden_states[0][:,0,:]

features[0].shape

features = last_hidden_states[0][:,0,:].numpy() 

from sklearn.preprocessing import LabelBinarizer, StandardScaler
encoder = LabelBinarizer()
labels = encoder.fit_transform(labels)
print(labels)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
#train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=42)

#stratified train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels) #80% training and 20% testing

#split the dataset into train and validations sets
#adjust validation size
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=42) # 0.125 * 0.8 = 0.10 --> #10% validation and 70% training    



#Define the DNN Model 

def create_model():
  model = Sequential()
  #Layers
  model.add(Dense(300,activation='relu',input_dim=768,kernel_regularizer=l2(0.01)))
  model.add(Dropout(0.1, noise_shape=None, seed=None))
  
 
  model.add(Dense(300,activation = 'relu',kernel_regularizer=l2(0.01)))
  model.add(Dropout(0.1, noise_shape=None, seed=None))
  
  model.add(Dense(300,activation = 'relu',kernel_regularizer=l2(0.01)))
  model.add(Dropout(0.1, noise_shape=None, seed=None))   

  #model.add(Dense(300,activation = 'relu',kernel_regularizer=l2(0.01)))
  #model.add(Dropout(0.1, noise_shape=None, seed=None)) 
  
  #Output layer
  model.add(Dense(3,activation='softmax'))
#from keras import optimizers
  #sgd = optimizers.SGD(lr=0.02, decay=1e-9, momentum=0.9, nesterov=True)
  #compile the model
  model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
  #Check the Model summary
  model.summary()
  return model


#Training the DNN
model=create_model()
history= model.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=100,batch_size=32, verbose=1) 


# Use score method to get validation accuracy of model
score = model.evaluate(X_val, y_val)
print(score)

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()


#accuracy figure
plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
plt.legend()
plt.show();



# Test Evaluation

#Predict the response for test dataset   
predictions =  np.argmax(model.predict(X_test), axis=1) 


from sklearn.metrics import accuracy_score
#score=accuracy_score(y_test, predictions)
test_score=accuracy_score( np.argmax(y_test, axis=1), predictions)
print('Accuracy:',test_score)


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

cm = metrics.confusion_matrix(np.argmax(y_test, axis=1), predictions)
print(cm)

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
#all_sample_title = 'Accuracy Score: {0}'.format(score)
#plt.title(all_sample_title, size = 15);

from sklearn.metrics import classification_report 

report = metrics.classification_report (np.argmax(y_test, axis=1), predictions)
print(report)


