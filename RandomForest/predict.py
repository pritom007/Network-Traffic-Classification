import matplotlib.pyplot as plt
import pandas as pd
# Helper libraries
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils.multiclass import unique_labels
import seaborn as sns
from dictionary import name_convert


def cm_analysis(y_true, y_pred, labels, ymap=None, title="", figsize=(10, 10)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args:
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    l = []
    for i in labels:
        l.append(labels.index(i))
    cm = confusion_matrix(y_true, y_pred, labels=l)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    ax.set(title=title)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)
    plt.savefig("cm.png")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


file_dir = "D:\\SDN Project\\Data\\Processed\\"

names = ['#flow_id', 'protocol', 'src_ip', 'src_port', 'dst_ip', 'dst_port', 'ndpi_proto_num', 'src2dst_packets',
         'src2dst_bytes', 'dst2src_packets', 'dst2src_bytes', 'ndpi_proto']
df = pd.read_csv(file_dir + '26sep19.csv', names=names)
X = np.asarray(
    df[['protocol', 'src_port', 'dst_port', 'src2dst_packets', 'src2dst_bytes', 'dst2src_packets', 'dst2src_bytes']][
    1:])
y = df['ndpi_proto'][1:]

# read the dictionary

read_file = open("randomforest.txt", 'r')
dict_class = []
for i in read_file.readlines():
    name, index = str(i).split(" ")[0], str(i).split(" ")[1]
    if name not in dict_class:
        dict_class.append(name)
# Recreate the exact same model
count = 1
loaded_model = pickle.load(open('randomforest_model.sav', 'rb'))
result = loaded_model.predict(X)
original = []
for i in result:
    print(f'{count} {i} {dict_class[i]} {name_convert(y[count])}')
    if name_convert(y[count]) not in dict_class:
        original.append(7)
    else:
        original.append(dict_class.index(name_convert(y[count])))
    count += 1
print(accuracy_score(original, result))
print(confusion_matrix(original, result))
cm_analysis(original, result, dict_class, ymap=None, title="Confusion Matrix", figsize=(10, 10))
plot_confusion_matrix(original, result, classes= np.asarray(dict_class), normalize=False, title="Confusion Matrix",cmap=plt.cm.Reds)
plt.show()
plt.savefig("cm_1.png")