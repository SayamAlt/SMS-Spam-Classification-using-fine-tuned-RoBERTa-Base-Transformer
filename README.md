# SMS-Spam-Classification-using-fine-tuned-RoBERTa-Base-Transformer

## Objective

This project focuses on developing a deep learning-based transformer model which can accurately classify SMS messages as spam or legitimate. The primary goal of this study is to examine the characteristics of SMS messages and establish a software product/service which can automatically identify and block spam messages. Moreover, it emphasizes on the development of appropriate strategies and measures to identify and eradicate spam SMS messages from the domain of digital communication.

In this project, I've trained and fine-tuned a RoBERTa-Base Spam Detection Classifier by augmenting additional neural network layers to the base RoBERTa transformer model to make it suitable enough for classifying spam SMS messages. 

## Dataset Description

The SMS Spam Collection v.1 is a public set of SMS messages that have been collected and labeled as either spam or not spam. This dataset contains 5574 English, real, and non-encoded messages, tagged as being legitimate(ham) or spam. The SMS messages are thought-provoking and eye-catching. The dataset is useful for facilitating mobile phone spam research. The dataset has been collected from various sources and is released under the CC BY-SA 4.0 license by Kaggle user Almeida et al.

<table>
  <tr>
    <th><b><em><strong>Column Name</strong></em></b></th>
    <th><b><em><strong>Description</strong></em></b></th>
  </tr>
  <tr>
    <td>sms</td>
    <td>The text of the SMS message. (String)</td>
  </tr>
  <tr>
    <td>label</td>
    <td>The label for the SMS message, indicating whether it is ham or spam. (String)</td>
  </tr>
</table>

## Detailed API Reference

The libraries used for implementing this project are as follows:

<ol type='a'>
  <li>wget: It is a free software package for retrieving files using HTTP, HTTPS, FTP and FTPS, the most widely used Internet protocols. It is a non-interactive commandline tool, so it may easily be called from scripts, cron jobs, terminals without X-Windows support, etc.</li>
  <li>transformers: The Hugging Face transformers package is an immensely popular Python library providing pretrained models that are extraordinarily useful for a variety of natural language processing (NLP) tasks.</li>
  <li>tensorflow: TensorFlow is a Python library for fast numerical computing created and released by Google. It is a foundation library that can be used to create Deep Learning models directly or by using wrapper libraries that simplify the process built on top of TensorFlow.</li>
  <li>matplotlib: It is a comprehensive library for creating static, animated, and interactive visualizations in Python. Matplotlib makes simple things easy and hard things possible. It can be used to create publication quality plots and make interactive figures that can zoom, pan, update.</li>
  <li>seaborn: It is a library that uses Matplotlib underneath to plot graphs. It can be used to visualize random distributions.</li>
  <li>numpy: NumPy is a Python library used for working with arrays. It also has functions for working in domain of linear algebra, fourier transform, and matrices. NumPy was created in 2005 by Travis Oliphant. It is an open source project and you can use it freely.</li>
  <li>pandas: Pandas is an open source Python package that is most widely used for data science/data analysis and machine learning tasks. It is built on top of another package named Numpy, which provides support for multi-dimensional arrays.</li>
  <li>sklearn: It is probably the most useful library for machine learning in Python. The sklearn library contains a lot of efficient tools for machine learning and statistical modeling including classification, regression, clustering and dimensionality reduction.</li>
  <li>wordcloud: Word Cloud is a data visualization technique used for representing text data in which the size of each word indicates its frequency or importance. Significant textual data points can be highlighted using a word cloud. Word clouds are widely used for analyzing data from social network websites.</li>
</ol>

For installing the above mentioned libraries, execute the following command:

```bash

 pip install wget transformers numpy pandas matplotlib seaborn scikit-learn wordcloud
 
```
The dataset can be accessed through the following 2 links:

<ul>
  <li><a href="https://www.kaggle.com/datasets/thedevastator/sms-spam-collection-a-more-diverse-dataset">Kaggle Link</a></li>
  <li><a href="https://huggingface.co/datasets/sms_spam">Hugging Face Link</a></li>
</ul>


 
  
