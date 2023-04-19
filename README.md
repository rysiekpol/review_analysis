
# Machine Learning 

@@ author -> Maciej Szefler - ma.szefler@gmail.com

These scripts were made for the Nokia machine learning internship task. I've made 3 scripts. In sentiment_analysis.py, I have provided an analysis of user opinion about the product. The reviews are classified as ["positive", "neutral", "negative"]. The accuracy for the test set I got was almost 80%. I tried many approaches the best of which turned out to be the use of support vector machines. 

The second script in categories_clustering.py was unsupervised learning for clustering comments into certain categories. The results are not great, but they show an approach that could be developed in the future.

The third script (recommend_analysis.py) was a simple data analysis to see the distribution of scores for unique products. I've come to the conclusion, that it is best to recommend products with an average score of >=7.5 because it provides the best quality and contains the most products to be recommended.

To each script are attached the charts I used to analyze the data.

## Packages

On Windows:

```bash
$ python -m venv venv
$ venv\Scripts\activate
$ pip install -r requirements.txt

```

On macOS/Linux:

```bash
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```
