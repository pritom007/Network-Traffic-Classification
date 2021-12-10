# Network-Traffic-Classification

This is a research project for classifying network traffic. We collected more than 300000 flows from the network. After that, we used nDPI to analyze the flows. We got more than 100 types of applications. Then we group that application into 10 classes. After that, we tried different ML algorithms to classify them.

Our current results-

Decision tree 95.8% accuracy

(I have added a new file https://github.com/pritom007/Network-Traffic-Classification/blob/master/DecisionTree/DecisionTree.ipynb with clean code. You just follow this code and implement it for KNN, RF)

Random forest 96.69% accuracy

KNN 97.24% accuracy

PAA 99.29% accuracy (read the paper to know more)

To get the dataset check out the instructions in the dataset folder.

# How Did we collect Data

We used Wireshark to collect the packets. Since for the project we wanted to use lab environment data, we first redirected our lab-network to one personal computer(pc) and in that pc we used Wireshark. After collecting the packets (as a .pcap file), we used ndpi to analyze the packets and get extract flow info and then we export that data as an excel file. The `data.csv` contains information on all parameters. However, for our project, we only used the top 7 most important parameters as features.

Github has limited the download so I am sharing a gdrive link for downloading the raw data: https://drive.google.com/file/d/1lcQmYyZutjsW_yJoHgx3Vles8eCgwQeD/view?usp=sharing

After you download it, you have to pre-process, in the paper, we showed in a table that how did we group the applications in 10 classes. 

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

# 中文版
# 网络流量分类

这是一个对网络流量进行分类的研究项目。我们从网络中收集了超过 300000 个流。之后，我们使用 nDPI 来分析流量。我们收到了 100 多种类型的应用程序。然后我们将该应用程序分为 10 个类。之后，我们尝试了不同的 ML 算法来对它们进行分类。

我们目前的结果——

Decision tree 95.8% 准确率

（我用干净的代码添加了一个新文件 https://github.com/pritom007/Network-Traffic-Classification/blob/master/DecisionTree/DecisionTree.ipynb。您只需按照此代码并为 KNN、RF 实现它）

Random forest 96.69% 准确率

KNN 97.24% 准确率

PAA 99.29% 准确率（阅读论文了解更多）

要获取数据集，请查看数据集文件夹中的说明。

# 如何收集数据的

我们使用 Wireshark 来收集数据包。由于对于我们想要使用实验室环境数据的项目，我们首先将我们的实验室网络重定向到一台个人计算机（PC），并在该 PC 中使用 Wireshark。收集数据包（作为 .pcap 文件）后，我们使用 ndpi 分析数据包并获取提取流信息，然后将该数据导出为 excel 文件。 `data.csv` 包含有关所有参数的信息。然而，对于我们的项目，我们只使用了前 7 个最重要的参数作为特征。

Github 限制了下载，所以我分享了一个用于下载原始数据的 gdrive 链接：https://drive.google.com/file/d/1lcQmYyZutjsW_yJoHgx3Vles8eCgwQeD/view?usp=sharing

下载后，您必须进行预处理，在论文中，我们在表格中展示了我们如何将应用程序分为 10 个类。

请阅读以下论文了解更多信息：https://doi.org/10.1080/09540091.2020.1870437

## 引用论文和代码：

article{mondal2021dynamic,<br>
  title={A dynamic network traffic classifier using supervised ML for a Docker-based SDN network},<br>
  author={Mondal, Pritom Kumar and Aguirre Sanchez, Lizeth P and Benedetto, Emmanuele and Shen, Yao and Guo, Minyi},<br>
  journal={Connection Science},<br>
  pages={1--26},<br>
  year={2021},<br>
  publisher={Taylor \& Francis}<br>
}
