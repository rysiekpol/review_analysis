import nltk
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import RegexpTokenizer, word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from data_processing import get_data
import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm
from nltk.corpus import stopwords


def clean_text(df, field) -> DataFrame:
    df[field] = df[field].str.replace(r"http\S+", " ")
    df[field] = df[field].str.replace(r"http", " ")
    df[field] = df[field].str.replace(r"@", "at")
    df[field] = df[field].str.replace("#[A-Za-z0-9_]+", ' ')
    df[field] = df[field].str.replace(r"[^A-Za-z(),!?@\'\"_\n]", " ")
    df[field] = df[field].str.lower()
    return df


lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer("english")


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
    text = [lemmatizer.lemmatize(word) for word in text.split() if not word in set(stopwords.words('english'))]
    text = ' '.join(text)
    return text


data = pd.read_csv("Cell_Phones_&_Accessories.csv")[:1000]

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

X_train, X_test, y_train, y_test = train_test_split(np.array(data["review/text"]), np.array(data["feedback"]),
                                                    test_size=0.25, random_state=42)

svc = SVC(random_state=42)

clfs = {
    "Support Vector Machine": svc,
}


def fit_model(clf, x_train, y_train, x_test, y_test):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_pred, y_test)
    return accuracy


accuracys = []
tfidf = TfidfVectorizer(use_idf=True, tokenizer=word_tokenize, min_df=0.00002, max_df=0.70)
X_train_tf = tfidf.fit_transform(X_train.astype('U'))
X_test_tf = tfidf.transform(X_test.astype('U'))

for name, clf in tqdm(clfs.items()):
    curr_acc = fit_model(clf, X_train_tf, y_train, X_test_tf, y_test)
    accuracys.append(curr_acc)

models_df = pd.DataFrame({"Models": clfs.keys(), "Accuracy Scores": accuracys}).sort_values('Accuracy Scores',
                                                                                            ascending=False)
print(models_df)
