import os

import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from data_processing import get_data
import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm
from nltk.corpus import stopwords
from sklearn.tree import DecisionTreeClassifier
import pickle


# data preprocessing
def clean_text(df, field) -> DataFrame:
    df[field] = df[field].str.replace(r"@", "at")
    df[field] = df[field].str.replace("#[A-Za-z0-9_]+", ' ')
    df[field] = df[field].str.replace(r"[^A-Za-z(),!?@\'\"_\n]", " ")
    df[field] = df[field].str.lower()
    return df


lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer("english")


# lemmatisation and stemming
def preprocess_text(text):
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = re.sub(r'[^\x00-\x7f]', '', text)
    text = " ".join([stemmer.stem(word) for word in text.split()])
    # getting rid of stopwords (like "the", "a", "an", "in")
    text = [lemmatizer.lemmatize(word) for word in text.split() if not word in set(stopwords.words('english'))]
    text = ' '.join(text)
    return text


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


# for simplicity,
# I assumed that score in [4.0, 5.0] means feedback is positive
# score in [3.0] is neutral
# score in [1.0, 2.0] is negative
def prepare_data():
    data = get_data()

    data = clean_text(data, "review/text")
    data['review/text'] = data['review/text'].apply(preprocess_text)
    data['feedback'] = np.nan
    for i in range(len(data['review/score'])):
        if data['review/score'][i] == 5.0 or data['review/score'][i] == 4.0:
            data['feedback'][i] = 'positive'
        elif data['review/score'][i] == 3.0:
            data['feedback'][i] = 'neutral'
        else:
            data['feedback'][i] = 'negative'

    data.to_csv("prepared_data.csv", index=False)


with open('prepared_data.csv', 'r') as file:
    data = pd.read_csv(file)

# creating train and test sets (75%, 25%)
X_train, X_test, y_train, y_test = train_test_split(np.array(data["review/text"]),
                                                    np.array(data["feedback"]),
                                                    test_size=0.25, random_state=42)

rf = RandomForestClassifier(random_state=42)
gb = GradientBoostingClassifier(random_state=42)
ada = AdaBoostClassifier(random_state=42)
dt = DecisionTreeClassifier(random_state=42)
svc = SVC(random_state=42)
nb = MultinomialNB()
mlp = MLPClassifier(random_state=42)

# algorithm can be developed further by adding more models
# I have chosen few of the most popular ones
clfs = {
    "Random Forest": rf,
    "Gradient Boosting": gb,
    "AdaBoost": ada,
    "Decision Tree": dt,
    "Support Vector Machine": svc,
    "Naive Bayes": nb,
    "Multilayer Perceptron": mlp
}

# fitting models and saving them to pickle files
# this can take up to 40 minutes
"""
If the data was bigger and don't have so much time,
you can test the models on smaller datasets (e.g. 10% of full dataset)
"""


def test_models():
    def fit_model(clf, x_train, y_train):
        clf.fit(x_train, y_train)

    # Term Frequency Inverse Document Frequency
    # algorithm for transforming text into meaningful representation of numbers
    # it gives number of frequency
    tfidf = TfidfVectorizer(use_idf=True, tokenizer=word_tokenize, min_df=0.00002, max_df=0.70)
    X_train_tf = tfidf.fit_transform(X_train.astype('U'))
    X_test_tf = tfidf.transform(X_test.astype('U'))

    for name, clf in tqdm(clfs.items()):
        fit_model(clf, X_train_tf, y_train, X_test_tf, y_test)
        with open(f'{name}.pickle', 'wb') as file:
            pickle.dump(clf, file)


# when models are saved to .pickle
# create accuracy to see which one was the best
def create_accuracy():
    accuracy = []
    tfidf = TfidfVectorizer(use_idf=True, tokenizer=word_tokenize, min_df=0.00002, max_df=0.70)
    X_train_tf = tfidf.fit_transform(X_train.astype('U'))
    X_test_tf = tfidf.transform(X_test.astype('U'))
    for name, clf in tqdm(clfs.items()):
        with open(f'{name}.pickle', 'rb') as file:
            clf = pickle.load(file)
        accuracy.append(accuracy_score(clf.predict(X_test_tf), y_test))

    models_df = pd.DataFrame({"Models": clfs.keys(), "Accuracy Scores": accuracy}).sort_values('Accuracy Scores',
                                                                                               ascending=False)
    models_df.to_csv("accuracy.csv", index=False)


def create_model_chart():
    models_df = pd.read_csv("accuracy.csv")
    # Figure Size
    fig, ax = plt.subplots(figsize=(16, 9))

    # Horizontal Bar Plot
    ax.barh(models_df["Models"], models_df["Accuracy Scores"])

    # Remove x, y Ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    # Add padding between axes and labels
    ax.xaxis.set_tick_params(pad=5)
    ax.yaxis.set_tick_params(pad=10)

    # Add x, y gridlines
    ax.grid(color='grey',
            linestyle='-.', linewidth=0.5,
            alpha=0.2)

    # Show top values
    ax.invert_yaxis()

    # Add annotation to bars
    for i in ax.patches:
        plt.text(i.get_width() + 0.01, i.get_y() + 0.5,
                 str(round((i.get_width()), 2)),
                 fontsize=10, fontweight='bold',
                 color='grey')

    # Add Plot Title
    ax.set_title('Accuracy of different models',
                 loc='center', )
    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Models')

    # Show Plot
    plt.savefig('models_accuracy_chart.png')
    plt.show()


create_model_chart()


# As we can see, the best model is SVC with 0.79 accuracy score
# Now we can tune the hyperparameters of the model to get the best accuracy score
# This can take up to 10 minutes
def tune_model():
    tfidf = TfidfVectorizer(use_idf=True, tokenizer=word_tokenize, min_df=0.00002, max_df=0.70)
    X_train_tf = tfidf.fit_transform(X_train.astype('U'))
    X_test_tf = tfidf.transform(X_test.astype('U'))

    svc_model = SVC(random_state=42)

    # testing for linear, gaussian and polynomial kernel
    """
    if you need to save time, you could use LinearSVC instead of SVC with linear kernel
    LinearSVC is based on LIBLINEAR library which contain optimized algorithm for linear SVM
    """
    param_grid = [
        {'C': [0.9, 0.96, 1], 'kernel': ['linear']},
        {'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
        {'C': [0.1, 1, 10, 100], 'degree': [2, 3, 4], 'kernel': ['poly']},
    ]
    grid_search = GridSearchCV(svc_model, param_grid, cv=5)
    grid_search.fit(X_train_tf[:3000], y_train[:3000])
    print(grid_search.best_params_)
    print(grid_search.best_score_)

# It looks like the SVC with C=0.96 and kernel = linear gets the best results
# Now we can train it on full dataset

def final_model():
    tfidf = TfidfVectorizer(use_idf=True, tokenizer=word_tokenize, min_df=0.00002, max_df=0.70)
    X_train_tf = tfidf.fit_transform(X_train.astype('U'))
    X_test_tf = tfidf.transform(X_test.astype('U'))

    # For saving time, I am using LinearSVC instead os standard SVC with kernel = linear
    # As explained before, this algorithm is much faster with time complexity of O(m*n)
    svm_clf = LinearSVC(C=0.96, loss='hinge', random_state=42)
    svm_clf.fit(X_train_tf, y_train)
    with open('final_model.pickle', 'wb') as file:
        pickle.dump(svm_clf, file)
    print(accuracy_score(svm_clf.predict(X_test_tf), y_test))

final_model()

"""
Final model gets accuracy of almost 80%. It did not improve previous best model,
which means that the default hyper parameters were good.
It is also good to see that the model is not overfitting,
because the accuracy score on test set is almost the same as on train set
To improve we could use e.g. RoBERTa model or try with other models not used in this analysis
"""

