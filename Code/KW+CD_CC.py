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

#load the dataset
df = pd.read_csv("kw+levin_cc.csv")
df.replace(np.nan,0)

#our inputs will contain 87 features
X = df.drop(columns=['3_labels'], axis=1).replace(np.nan,0)
#the labels are the following
Y = df['3_labels']


#One Hot Encode our Y:
from sklearn.preprocessing import LabelBinarizer, StandardScaler
encoder = LabelBinarizer()
Y = encoder.fit_transform(Y)
print(Y)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X= sc.fit_transform(X)
print(X)

#stratified train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y) #80% training

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=42) # 0.125 * 0.8 = 0.10 --> #10% validation and 70% training


#Define the DNN Model 

def create_model():
  model = Sequential()
  #Layers
  model.add(Dense(76,activation='relu',input_dim=66,kernel_regularizer=l2(0.01)))
  model.add(Dropout(0.1, noise_shape=None, seed=None))
  
  
  model.add(Dense(76,activation = 'relu',kernel_regularizer=l2(0.01)))
  model.add(Dropout(0.1, noise_shape=None, seed=None))
 
  #model.add(Dense(100,activation = 'relu',kernel_regularizer=l2(0.01)))
  #model.add(Dropout(0.1, noise_shape=None, seed=None))   

  #model.add(Dense(100,activation = 'relu',kernel_regularizer=l2(0.01)))
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
history= model.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=100,batch_size=32, verbose=1) #

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

#Predict the response for test dataset   
predictions =  np.argmax(model.predict(X_test), axis=1) 
predictions



from sklearn.metrics import accuracy_score

test_score=accuracy_score( np.argmax(y_test, axis=1), predictions)
print('Accuracy:',test_score)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

#from sklearn.metrics import confusion_matrix
cm = metrics.confusion_matrix(np.argmax(y_test, axis=1), predictions)
print(cm)

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');

from sklearn.metrics import classification_report 

report = metrics.classification_report (np.argmax(y_test, axis=1), predictions)
print(report)