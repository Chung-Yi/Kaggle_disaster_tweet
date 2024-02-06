import matplotlib.pyplot as plt
import plotly.express as px
import os
import spacy
import random
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.gridspec as gridspec
from plotly.subplots import make_subplots
from matplotlib.ticker import MaxNLocator
from collections import Counter, defaultdict
from nltk.probability import FreqDist
from wordcloud import WordCloud, STOPWORDS
from PIL import Image

nlp = spacy.load("en_core_web_sm")

def draw(df, feature, chart_type='pie', marginal='histogram'):


    # df = px.data.tips()
    # print(df['target'].head())


    # df = px.data.tips()
    # print(df.head())
    # fig = px.pie(df, values='tip', names='day')
    # fig.show()

    
    

    if chart_type == 'pie':
        fig = px.pie(df, values=feature, names='target')
        
    else:
        fig = px.ecdf(df, x=feature, color='target', markers=True, lines=False, marginal=marginal)

        
    fig.show()


def plot_word_number_histogram(df):
    is_disaster_tweet = df[df['target'] == 1]['text']
    not_disaster_tweet = df[df['target'] == 0]['text']

    # print(is_disaster_tweet)

    is_disaster_tweet_count = is_disaster_tweet.map(lambda x: len(x.split(' ')))
    no_disaster_tweet_count = not_disaster_tweet.map(lambda x: len(x.split(' ')))


    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(
        go.Histogram(
            # x=is_disaster_tweet.str.split().map(lambda x: len(x))
            x=is_disaster_tweet_count
        ), row=1, col=1
    )

    fig.add_trace(
        go.Histogram(
            # x=not_disaster_tweet.str.split().map(lambda x: len(x))
            x=no_disaster_tweet_count
        ), row=1, col=2
    )

    fig.update_layout()
    fig.show()

   

    # fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(18, 6), sharey=True)
    # sns.distplot(is_disaster_tweet.str.split().map(lambda x: len(x)), ax=axes[0], color='#e74c3c')
    # sns.distplot(not_disaster_tweet.str.split().map(lambda x: len(x)), ax=axes[1], color='#e74c3c')
    
    # axes[0].set_xlabel('Word Count')
    # axes[0].set_ylabel('Frequency')
    # axes[0].set_title('Non Disaster Tweets')
    # axes[1].set_xlabel('Word Count')
    # axes[1].set_title('Disaster Tweets')
    
    # fig.suptitle('Words Per Tweet', fontsize=24, va='baseline')
    
    # fig.tight_layout()
    # plt.show()


def plot_word_len_histogram(df):
    is_disaster_tweet = df[df['target'] == 1]['text']
    not_disaster_tweet = df[df['target'] == 0]['text']

    
    

    is_disaster_tweet_mean_length = is_disaster_tweet.str.split().apply(lambda x: [len(word) for word in x]).map(lambda x: np.mean(x))
    not_disaster_tweet_mean_length = not_disaster_tweet.str.split().apply(lambda x: [len(i) for i in x]).map(
        lambda x: np.mean(x))

   
    # print(is_disaster_tweet_mean_length)
    # print(is_disaster_tweet.str.split().apply(lambda x: [len(i) for i in x]).map(
    #     lambda x: np.mean(x)))
    
    # print(is_disaster_tweet_mean_length == is_disaster_tweet.str.split().apply(lambda x: [len(i) for i in x]).map(
    #     lambda x: np.mean(x)))
    
    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(go.Histogram(x=is_disaster_tweet_mean_length), row=1, col=1)
    fig.add_trace(go.Histogram(x=not_disaster_tweet_mean_length), row=1, col=2)

    fig.update_layout()
    fig.show()


    # fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(18, 6), sharey=True)
    # sns.distplot(not_disaster_tweet.str.split().apply(lambda x: [len(i) for i in x]).map(
    #     lambda x: np.mean(x)),
    #              ax=axes[0], color='#e74c3c')
    # sns.distplot(is_disaster_tweet.str.split().apply(lambda x: [len(i) for i in x]).map(
    #     lambda x: np.mean(x)),
    #              ax=axes[1], color='#e74c3c')
    
    # axes[0].set_xlabel('Word Length')
    # axes[0].set_ylabel('Frequency')
    # axes[0].set_title('Non Disaster Tweets')
    # axes[1].set_xlabel('Word Length')
    # axes[1].set_title('Disaster Tweets')
    
    # fig.suptitle('Mean Word Lengths', fontsize=24, va='baseline')
    # fig.tight_layout()
    # plt.show()

def ngrams(df, stops):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    # axes = axes.flatten()

    is_disaster_tweet = df[df['target'] == 1]['lemma_str']
    not_disaster_tweet = df[df['target'] == 0]['lemma_str']

    new_is_disaster_tweet = is_disaster_tweet.str.split()
    new_not_disaster_tweet = not_disaster_tweet.str.split()
    # new_is_disaster_tweet_list = new_is_disaster_tweet.values.tolist()
    corpus_is_disaster_tweet = [word for i in new_is_disaster_tweet for word in i]
    corpus_not_disaster_tweet = [word for i in new_not_disaster_tweet for word in i]
    # corpus = new_is_disaster_tweet.apply(lambda x: [word.lower() for word in x])

    is_disaster_tweet_counter = Counter(corpus_is_disaster_tweet)
    most_is_disaster_tweet = is_disaster_tweet_counter.most_common(30)

    print(most_is_disaster_tweet)

    not_disaster_tweet_counter = Counter(corpus_not_disaster_tweet)
    most_not_disaster_tweet = not_disaster_tweet_counter.most_common(30)
    
    x1 = []
    y1 = []

    x2 = []
    y2 = []
    
    for word, count in most_is_disaster_tweet:
        if (word not in stops):
            x1.append(word)
            y1.append(count)

    for word, count in most_not_disaster_tweet:
        if (word not in stops):
            x2.append(word)
            y2.append(count)
    

    sns.barplot(x=y1, y=x1, palette='plasma', ax=axes[0])
    sns.barplot(x=y2, y=x2, palette='plasma', ax=axes[1])

    axes[0].set_title('Disaster Tweets')
    axes[1].set_title('Non Disaster Tweets')

    axes[0].set_xlabel('Count')
    axes[0].set_ylabel('Word')
    axes[1].set_xlabel('Count')
    axes[1].set_ylabel('Word')

    fig.suptitle('Most Common Unigrams', fontsize=24, va='baseline')
    plt.tight_layout()
    plt.show()


    # lis = [
    #     df[df['target'] == 0]['lemma_str'],
    #     df[df['target'] == 1]['lemma_str']
    # ]

    # fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    # axes = axes.flatten()

    # for i, j in zip(lis, axes):

    #     new = i.str.split()
    #     new = new.values.tolist()
    #     corpus = [word for i in new for word in i]

    #     counter = Counter(corpus)
    #     most = counter.most_common()
    #     x, y = [], []
    #     for word, count in most[:30]:
    #         if (word not in stops):
    #             x.append(word)
    #             y.append(count)

    #     sns.barplot(x=y, y=x, palette='plasma', ax=j)
    # axes[0].set_title('Non Disaster Tweets')

    # axes[1].set_title('Disaster Tweets')
    # axes[0].set_xlabel('Count')
    # axes[0].set_ylabel('Word')
    # axes[1].set_xlabel('Count')
    # axes[1].set_ylabel('Word')

    # fig.suptitle('Most Common Unigrams', fontsize=24, va='baseline')
    # plt.tight_layout()
    # plt.show()
    

    
def plot_wordcloud(text, title, title_size):
    words = text
    allwords = []

    for wordlist in words:
        allwords += wordlist
    
    mostcommon = FreqDist(allwords).most_common(140)
    mask = np.array(Image.open('twittermask.png'))
    mask[mask.sum(axis=2) == 0] = 255
    wordcloud = WordCloud(
        width=1200,
        height=800,
        background_color='black',
        stopwords=set(STOPWORDS),
        max_words=150,
        scale=3,
        mask=mask,
        contour_width=0.1,
        contour_color='grey',
    ).generate(str(mostcommon))
    

    def grey_color_func(word,
                        font_size,
                        position,
                        orientation,
                        random_state=None,
                        **kwargs):
        return "hsl(0, 0%%, %d%%)" % random.randint(60, 100)




    fig = plt.figure(figsize=(10, 10))

    plt.imshow(wordcloud.recolor(color_func=grey_color_func, random_state=42), interpolation="bilinear")

    plt.axis('off')
    plt.title(title,
              fontdict={
                  'size': title_size,
                  'verticalalignment': 'bottom'
              })
    # plt.tight_layout()
    plt.show()

def plot_name_entity_barchart(df):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    # axes = axes.flatten()

    is_disaster_tweet = df[df['target'] == 1]['lemma_str']
    not_disaster_tweet = df[df['target'] == 0]['lemma_str']
    
    def __get__ner(i):
        doc = nlp(i)

        return [x.label_ for x in doc.ents]
    
    ent1 = is_disaster_tweet.apply(lambda x: __get__ner(x))
    ent1 = [x for sub in ent1 for x in sub]

    ent2 = not_disaster_tweet.apply(lambda x: __get__ner(x))
    ent2 = [x for sub in ent2 for x in sub]

    counter1 = Counter(ent1)
    counter1 = counter1.most_common(15)

    counter2 = Counter(ent2)
    counter2 = counter2.most_common(15)

    x1, y1 = map(list, zip(*counter1))
    x2, y2 = map(list, zip(*counter2))

    sns.barplot(x=y1, y=x1, palette='plasma', ax=axes[0])
    sns.barplot(x=y2, y=x2, palette='plasma', ax=axes[1])

    axes[0].set_title("Disaster Tweets")
    axes[1].set_title("Non Disaster Tweets")

    axes[0].set_xlabel("Count")
    axes[0].set_ylabel("Named")

    axes[1].set_xlabel("Count")
    axes[1].set_ylabel("Named")

    fig.suptitle("Common Named-Entity Counts")
    plt.show()

    

   

