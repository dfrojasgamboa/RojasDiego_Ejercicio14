import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split # Split arrays or matrices into random train and test subsets
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

images = glob.glob("./train/*.jpg")

data=[]
for i in range(len(images)):
    d=np.float_(plt.imread(images[i]).flatten())
    data.append(d)

data = np.array(data)

#si e<s hombre es 1, si es mujer es 0
target=[]
for i in range(len(data)):
    if((int(images[i][8:-4])+1)%2)==0:
        target.append(1)
    else:
        target.append(0)

target = np.array(target)


# Split in train/test
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.7) # train_size

# Reescalado de los datos
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

c_array = np.logspace(-3,5)
f1_array = []
count = 0

for c in c_array:    
    #Create a svm Classifier
    svm = SVC( C = c  , kernel='linear' ) # Linear Kernel

    #Train the model using the training sets
    svm.fit(x_train, y_train)

    # predigo los valores para test
    y_predict = svm.predict(x_test)

    f1_array.append( f1_score(y_test, y_predict ) )

F1 = np.array(f1_array)
ii = np.argmax(f1_array)
C_max = c_array[ii]

svm = SVC( C=C_max, kernel='linear')
svm.fit(x_train, y_train)
y_pred = svm.predict(x_test)

files_test = glob.glob("test/*.jpg")
n_test = len(files_test)
predict_test = y_pred

out = open("test/predict_test.csv", "w")
out.write("Name,Target\n")
for f, p in zip(files_test, predict_test):
    print(f.split("/")[-1], p)
    out.write("{},{}\n".format(f.split("/")[-1],p))

out.close()