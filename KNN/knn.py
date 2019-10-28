import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.neighbors.nca import NeighborhoodComponentsAnalysis

import pickle
file_dir = "D:\\SDN Project\\Data\\"

names = ['#flow_id', 'protocol', 'src_ip', 'src_port', 'dst_ip', 'dst_port', 'ndpi_proto_num', 'src2dst_packets',
         'src2dst_bytes', 'dst2src_packets', 'dst2src_bytes', 'ndpi_proto', 'class']
print("Loading Dataset total_class.csv")
df = pd.read_csv(file_dir + 'total_class.csv', names=names)

X = np.asarray(
    df[['protocol', 'src_port', 'dst_port', 'src2dst_packets', 'src2dst_bytes', 'dst2src_packets', 'dst2src_bytes']][
    1:])
y = []

my_tags = []
classes = open("knn.txt", "w+")
for i in df['class'][1:]:
    if i not in my_tags:
        my_tags.append(i)
for i in df['class'][1:]:
    classes.write(i + " " + str(my_tags.index(i)) + "\n")
    y.append(my_tags.index(i))
y = np.asarray(y)
print("Splitting dataset")
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True)
max = 0
for i in range(2, 15):
    knn = KNeighborsClassifier(n_neighbors=i)

    #print("Training Started")
    knn.fit(x_train, y_train)

    #print("Testing the classifier")
    y_pred = knn.predict(x_test)
    acc = accuracy_score(y_pred, y_test)
    print(f'K={i} accuracy %s' % acc)
    if acc >= max:
        print("Saving the model")
        filename = 'knn_'+str(i)+'_model.sav'
        pickle.dump(knn, open(filename, 'wb'))
        max = acc
        print('saved acc: ', max)


#print(classification_report(y_test, y_pred, target_names=my_tags, labels=range(len(my_tags))))