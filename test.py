import pandas as pd
import numpy as np
from keras.models import load_model

test_dataset = pd.read_csv('data/test.csv')
# print(test_dataset.head())
# print(test_dataset.shape)
X_test = test_dataset.iloc[:,:].values

X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

X_test /= 255
# print(X_test.shape)


model = load_model('mnist_cnn_18-04-12-18-47.h5')
y_pred = model.predict_proba(X_test)

y_conv = []
for i in range(len(y_pred)):
    max_val = max(y_pred[i])
    y_arr = list(y_pred[i])
    max_index = y_arr.index(max_val)
    y_conv.append(max_index)

with open('submission.csv','w') as f:
    f.write('ImageId,Label\n')
    for i,pred in enumerate(y_conv):
        f.write(str(i+1)+','+str(pred)+'\n')
