import re
from collections import defaultdict
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk import WordNetLemmatizer, SnowballStemmer, word_tokenize
from nltk.corpus import stopwords
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline

from data_processing import get_data

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


def prepare_data():
    data = get_data()

    data = clean_text(data, "review/summary")
    data['review/summary'] = data['review/summary'].apply(preprocess_text)
    data['num_letters'] = data['review/summary'].str.count('[a-zA-Z]')
    data = data[data.num_letters > 0]
    data = data[['review/summary', 'num_letters']]
    data.to_csv("data_clustering.csv", index=False)


def generate_ngrams(text, n_gram=1):
    token = [w.lower() for sent in nltk.sent_tokenize(text) for w in nltk.word_tokenize(sent)]
    token = [t for t in token if re.search('[a-zA-Z]', t) and t not in set(stopwords.words('english'))]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]


def summarize_data():
    data = pd.read_csv("data_clustering.csv")
    freq_dict = defaultdict(int)
    for sent in data["review/summary"]:
        for word in generate_ngrams(sent):
            freq_dict[word] += 1
    wdf = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1],
                       columns=["word", "word_count"])
    # Count Trigrams
    freq_dict = defaultdict(int)
    for sent in data["review/summary"]:
        for word in generate_ngrams(sent, 3):
            freq_dict[word] += 1
    tdf = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1],
                       columns=["trigram", "trigram_count"])

    # Sort and filter top 12
    tdf = tdf.sort_values(by='trigram_count', ascending=False).iloc[:12]
    wdf = wdf.sort_values(by='word_count', ascending=False).iloc[:12]

    create_chart(wdf.word.values, wdf.word_count.values, "Word")
    create_chart(tdf.trigram.values, tdf.trigram_count.values, "Trigram")


def create_chart(x, y, word_type):
    ig, ax = plt.subplots(figsize=(16, 9))

    # Horizontal Bar Plot
    ax.barh(x, y)

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
    ax.set_title('Number of ' + word_type,
                 loc='center', )
    ax.set_xlabel('Count')
    ax.set_ylabel(word_type)

    # Show Plot
    plt.savefig(word_type + '_chart.png')
    plt.show()


def save_top_words_for_topic(components, feature_names, num_words):
    """Save top words for each topic in components vector."""
    with open('top_word_for_topics.txt', 'wb') as f:
        for topic_idx, topic in enumerate(components):
            message = "Topic #%d: " % (topic_idx + 1)
            message += " ".join([feature_names[i] for i in topic.argsort()[:-num_words - 1:-1]])
            f.write(message.encode('utf-8'))
            f.write(b'message'+b"\n")


num_topics = 3


class Tokenize(BaseEstimator, TransformerMixin):
    """Class to tokenize comments."""

    def fit(self, x, y=None):
        return self

    def transform(self, comments):
        tokenized_comments = list()
        for text in comments:
            # Tokenize and lower
            tokens = [w.lower() for sent in nltk.sent_tokenize(text) for w in nltk.word_tokenize(sent)]
            # Filter tokens that doesn't have letters
            tokens = [t for t in tokens if re.search('[a-zA-Z]', t)]
            tokenized_comments.append(tokens)
        return tokenized_comments


def training_model():
    lda_pipeline = Pipeline([
        ('tokenize', Tokenize()),
        #('countvec', CountVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x,
        #                             analyzer='word', min_df=2, max_df=0.95)),
        ('countvec', TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, analyzer='word', min_df=2, max_df=0.95)),
        ('reduce', LatentDirichletAllocation(n_components=num_topics,
                                             random_state=42))
    ])

    # I use prepared data which is stemmed and lemmatized before
    data = pd.read_csv("data_clustering.csv")['review/summary']

    lda_pipeline.fit_transform(data)
    # Get words and components from pipeline transformers
    feat_names = lda_pipeline.get_params()['countvec'].get_feature_names_out()
    components = lda_pipeline.get_params()['reduce'].components_
    # Print the 8 most important words for each topic
    save_top_words_for_topic(components, feat_names, 8)


training_model()
"""
As we can see in the top_word_for_topics.txt file, the topics are:
Topic #1: mostly about the user feelings, like good, nice, to buy, etc.
Topic #2: mostly about the product type or features like name, utilities, etc.
Topic #3: mostly about the product quality, which is good or bad

The problem with this approach is that we have to manually check the topics and assign them to the right category.
There is also a problem with checking the unsupervised model accuracy, because we don't have the labels.
Of course the same words, may be in the different topics, because of the high probability of their appearance in the text.
"""

