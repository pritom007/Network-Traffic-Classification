# Network-Traffic-Classification

This is a research project for classifying network traffic. We collected more than 300000 flows from the network. After that we used nDPI to analze the flows. We got more than 100 types of application. Then we put those application into 7 classes. After that we tried different ML algorithms to classify them.

Our current results-

Decesion tree 95.8% accurecy 

(I have added a new file https://github.com/pritom007/Network-Traffic-Classification/blob/master/DecisionTree/DecisionTree.ipynb  with clean code. You just follow this code and implement for KNN, RF)

Randomforest 96.69% accuracy

KNN 97.24% accuracy

PAA 99.29% accuracy

To get the dataset checout the instructions in dataset folder.

# How Did we collect Data

We used wireshark to collect the packets.Since for the project we wanted to use lab environment data, we first redirected out labnetwork to one pc and in that pc we used wireshark. After collecting the packets (as .pcap file), we used ndpi to analysis the the packets and get the flow info and then we export that data as excel file. The `data.csv` contains information of all parameters. however, for our project we only used top 7 most important parameter as feature.

Please read the following paper to know more: https://doi.org/10.1080/09540091.2020.1870437
