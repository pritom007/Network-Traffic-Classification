# Network-Traffic-Classification

This is a research project for classifying network traffic. We collected more than 300000 flows from the network. After that we used nDPI to analze the flows. We got more than 100 types of application. Then we group those application into 10 classes. After that we tried different ML algorithms to classify them.

Our current results-

Decesion tree 95.8% accurecy 

(I have added a new file https://github.com/pritom007/Network-Traffic-Classification/blob/master/DecisionTree/DecisionTree.ipynb  with clean code. You just follow this code and implement for KNN, RF)

Randomforest 96.69% accuracy

KNN 97.24% accuracy

PAA 99.29% accuracy (read the paper to know more)

To get the dataset checout the instructions in dataset folder.

# How Did we collect Data

We used wireshark to collect the packets. Since for the project we wanted to use lab environment data, we first redirected out lab-network to one personal computer(pc) and in that pc we used wireshark. After collecting the packets (as .pcap file), we used ndpi to analysis the the packets and get extract flow info and then we export that data as excel file. The `data.csv` contains information of all parameters. However, for our project we only used top 7 most important parameter as feature.

Github has limited the download so I am sharing a gdrive link for downloading the raw data: https://drive.google.com/file/d/1lcQmYyZutjsW_yJoHgx3Vles8eCgwQeD/view?usp=sharing

After you download it, you have to pre-process, in the paper we showed in a table that how did we grouped the applications in 10 classes. 

Please read the following paper to know more: https://doi.org/10.1080/09540091.2020.1870437

## To cite the paper and code:

article{mondal2021dynamic,<br>
  title={A dynamic network traffic classifier using supervised ML for a Docker-based SDN network},<br>
  author={Mondal, Pritom Kumar and Aguirre Sanchez, Lizeth P and Benedetto, Emmanuele and Shen, Yao and Guo, Minyi},<br>
  journal={Connection Science},<br>
  pages={1--26},<br>
  year={2021},<br>
  publisher={Taylor \& Francis}<br>
}


