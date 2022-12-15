# SMS-Spam-Classification-using-fine-tuned-RoBERTa-Base-Transformer

![SMS Spam Detection](https://miro.medium.com/max/1400/0*mbFBPcPUJD-53v3h.png)
![A Spam Transformer Model](https://d3i71xaburhd42.cloudfront.net/7e9e45a9a495fb090551dbc3c3932fec3e2735a8/5-Figure3-1.png)
![Deep Learning Architecture to filter SMS spam](https://ars.els-cdn.com/content/image/1-s2.0-S0167739X19306879-gr2.jpg)
![Pretrained Roberta Transformer](https://thumbnails.huggingface.co/social-thumbnails/models/mariagrandury/roberta-base-finetuned-sms-spam-detection.png)

## Overview

Spam is electronic communication that is unsolicited, undesirable, and potentially malicious. While SMS spam is often transmitted via a mobile network, email spam is sent and received through the Internet. Users who send spam will be referred to as "spammers." Since sending SMS texts is typically fairly inexpensive (if not free) for the user, it is enticing for unfair exploitation. This is exacerbated by the fact that users typically view SMS as a more secure and reliable method of communication than other sources, such as emails. 

The risks associated with spam communications for consumers are numerous: unwanted advertising, the disclosure of personal information, falling prey to fraud or financial schemes, getting seduced into malware and phishing websites, unintentional exposure to offensive content, etc. Spam messages raise operational costs for the network operator. 

This project focuses on developing a deep learning-based transformer model which can accurately classify SMS messages as spam or legitimate. The primary goal of this study is to examine the characteristics of SMS messages and establish a software product/service which can automatically identify and block spam messages. Moreover, it emphasizes on the development of appropriate strategies and measures to identify and eradicate spam SMS messages from the domain of digital communication.

In this project, I implemented text preprocessing by applying various functions such as cleaning HTML, removing stopwords, digits, lowercase and punctuation characters, email addresses, etc. Subsequently, I trained and fine-tuned a RoBERTa-Base Spam Detection Classifier by augmenting additional neural network layers to the base roberta transformer model to make it suitable enough for classifying the spam SMS messages. The model evaluation metrics that have been adopted are typical classification metrics such as confusion matrix and classification report. The fine-tuned roberta-base transformer model produced an excellent accuracy score of more than 99%.

## Dataset Description

<p>The SMS Spam Collection v.1 is a public set of SMS messages that have been collected and labeled as either spam or not spam. This dataset contains 5574 English, real, and non-encoded messages, tagged as being legitimate(ham) or spam. The SMS messages are thought-provoking and eye-catching. The dataset is useful for facilitating mobile phone spam research.</p>

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

## Model Architecture Explanation

The transformer model used here is a fine-tuned version of roberta-base on the SMS spam collection dataset. It achieves the following results on the test set:

<ul>
  <li>Accuracy: 0.9923</li>
  <li>Loss: 0.0486</li>
</ul>

The overall architecture of the fine-tuned RoBERTa-Base spam detection transformer model is plotted below:

![Model Architecture](https://www.kaggleusercontent.com/kf/113846181/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..wpilX0T2DB1c8YgTllxCdQ.QJtJ5oSYBVaiRz8unsbFDF_S2S48teQqtRLrCSYQlcif6DgAlBjnAORlJw1QlhBqaY7kuG8MpvckXHQ1iNdJIMm4H2dEGkLInNx91kE6iTjxSnIZD3grdk_X1TO-sUsTvKNS3o4KPAEbLe1QwNClsLcm8hYF4M_f4h7oMTl9cv_aGSWpNzqHaqjBKO-4DpWboxE2-_rZYfMxfh_NVp9GomSVxmDx6wPFTZRM49fL4owF-qVSJ7bWtIXc8v2p6bLX6As2sF9-qXj7QOtju8eIQUtzIuV0vypm-Jy4dcyIA_yjM49MhJLQRJnLGhR4AWetHST73KjY51MsSXrFxmyYluVtV6-q-CqLv5OizjdKfKWOelA3JwFvTfXf8RzUnQeHcHEf90bgwho-Ov73OG4k8uaPZ3ps9vmSAmMNtOcSS103gfG4nANa6VYT_NwUaF0n5iivChk1r2TjQxMZ9vSbrJhr99igzIycg3MFGcsEx1k7k18FYuBfpzcEwyh6hftRjRQQYynJM_Pp3MJU3wCQarQfqqohEyjy3JCTln7-lik44-xrl4G9w2bhPCYkD0bv8DJXkFFoS0RQ_tfkCoZZ4vpIylSeyZXVbgEAxIcXI18m7vFLsya0S-VkKtvikHjzhTs-uWrKBLUbk-gn1dAv77SJec7hNyBoqbSueiDeswo.r1XQoU9ZCAxF1ttSzlBf4Q/ROBERTA-BASE-FINETUNED-SMS-SPAM-DETECTION.png)

You can access the pretrained roberta-base transformer model used in this project through the following link: ![mariagrandury/roberta-base-finetuned-sms-spam-detection](https://huggingface.co/mariagrandury/roberta-base-finetuned-sms-spam-detection)

Numerous dense fully-connected layers have been added to the roberta-base transformer as the base model which serves as the text embeddings alongside the input parameters, like input ids and attention masks, resulting in an optimized, customized and fine-tuned transformer architecture which has a dense final output layer for generating the binary predictions pertaining to spam detection.

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

## Use Cases

Nowadays, several SMS spam detection systems are regularly developed and maintained to detect and filter malicious messages and identify whether or not a message is spam or ham. We can enhance user experience by recognizing undesirable and unsolicited emails and preventing them from entering the user's mailbox. Spam detection and elimination are significantly enormous problems for email and IoT service providers these days.

## Future Scope

The model performance can be improved further by tuning the hyperparameters of the fine-tuned roberta-base transformer model such as raising the fully-connected dense layers, tweaking the parameters of the Adam optimizer or simply using other powerful optimizers, for instance, RMSProp, and increasing the number of epochs for training the model. The model can be used in any kind of SMS spam detection and filtering related software products/services which can potentially be great assets from a cyber security perspective. Furthermore, such softwares can lead to enhanced social media security services such as emails, SMS messages, etc.

## Acknowledgements

    This dataset is used to train a machine learning or deep learning model for classifying SMS messages as either spam or not spam.

    The SMS Spam Collection v.1 is a set of SMS labeled messages that have been collected for conducting mobile phone spam research. This dataset contains 5574 English, real, and non-encoded messages, tagged as being either legitimate (ham) or spam. The dataset has been collected from various sources and is released under the CC BY-SA 4.0 license by Kaggle user Almeida et al.

## License

    License: CC0 1.0 Universal (CC0 1.0) - Public Domain Dedication
    No Copyright - You can copy, modify, distribute and perform the work, even for commercial purposes, all without asking permission. See Other Information.


 
  
