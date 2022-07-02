# Fake-Real-News-Classification

Advances in technology and social medias have made access to sources of information easier compared to decades ago. In the past, reporters gathered information about an event, then the news was published by a publishing company., By contrast, these days people have the ability to post and publish any news they are exposed to. This has advantages and disadvantages. On one hand, social media companies such as Twitter have made it easier to post and publish news regarding an event much faster than ever; this makes many people more informed. On the other hand, this easy way of publishing and posting news resulted in the existence of an enormous volume of fake news. Therefore, it is important for these platforms to be able to filter out fake news by using different methods.

One main approach to filtering out fake news is using machine learning models such as logistic regression, decision trees, and neural networks. To use these models, it is important to convert text to numerical data. To convert words to numbers, first we need to clean the data and remove the words or information that may not help us categorize the data. After that, we must convert cleaned sentences to individual words called “tokens.” This allows us to design a map that transforms tokenized data into numbers. There are several libraries that we may introduce for these purposes such as NLTK, Spacy, Textbloob etc. In this work, however, we will use NLTK to tokenize the data.
 
The data that we are using are from different online sources which are:

1. [Fake and real news dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset). The data sets we use are:

    
2. [Source based Fake News Classification](https://www.kaggle.com/datasets/ruchi798/source-based-news-classification?select=news_articles.csv). The data sets we use are:


3. [REAL and FAKE news dataset](https://www.kaggle.com/datasets/nopdev/real-and-fake-news-dataset?select=news.csv). The data sets we use are:

    
4. [GitHub Repo](https://github.com/KaiDMML/FakeNewsNet). The data sets we use are:
    
    
The structure of this project is as follows:

1. **EDA-part-1-Cleaning-Tokenization-Lammatization**. In this notebook we clean the data, and then we tokenize and lemattize the data. This notebook is located in the “EDA” folder.

2. **EDA-part-2-Visualization**. In this notebook, we use the cleaned data and will visualize the top-10 words used in the whole dataset as well as top-1o words in fake news and true news. We also perform some statistical tests to see whether or not some columns of the datasets are coming from a same population. This notebook is located in the “EDA” folder.

3. **Modeling**. In this notebook we first calculate the term frequency–inverse document frequency (TF-IDF) for about 10,000 tokens and then we will train several machine learning models, namely, [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), [XGBoost](https://xgboost.readthedocs.io/en/stable/python/python_intro.html), [LightGBM](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier) on training data and we evaluate their performances on the test sets. This notebook is located in the “Modeling” folder.

4. **Modeling_GloVe**.In this notebook, we use a vector representation of words called  “Global Vectors for Word Representation” (for short GloVe) prepared by [Stanford University](https://nlp.stanford.edu/projects/glove/) to train our machine learning models and then test the performance of the model on testing set. This notebook is located in the “Modeling” folder. 

5. **Modeling_NN**. In this notebook, we use TensorFlow to design neural networks to be trained on training data and then we evaluate their performance on testing data. This notebook is located in the “Modeling” folder. 

6. **Modeling_NN_Transfer_Learning**. This is the last series of modeling notebooks. In this notebook, we use available embedding layers from [TensorFlow Hub](tfhub.dev) and will use them in a neural network to train the rest of the neural network and test its performance on the testing data. 
    

The model we introduce as a final model is the **3rd finetuned model** from the last notebook (Modeling_NN_Transfer_Learning) and the result of the model on the training and test sets are shown below.


![model_trained_3_df](./model_trained_3_df.JPEG)

