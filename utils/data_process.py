import pandas as pd
import re
import nltk
import string
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import RegexpTokenizer
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer

nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

wnl = WordNetLemmatizer()



stops = set(stopwords.words('english'))

def read_csv_file(file_name):
    return pd.read_csv(file_name)

def remove_url(text):
    pattern = re.compile('https?://\S+|www\.\S+')
    return re.sub(pattern, '', text)

def remove_emojis(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # 表情符號
                           u"\U0001F300-\U0001F5FF"  # 圖案符號
                           u"\U0001F680-\U0001F6FF"  # 運輸和地圖符號
                           u"\U0001F700-\U0001F77F"  # 運動符號
                           u"\U0001F780-\U0001F7FF"  # 圖畫符號
                           u"\U0001F800-\U0001F8FF"  # 表情符號補充
                        #    u"\U0001F900-\U0001F9FF"  # 裝飾符號補充
                        #    u"\U0001FA00-\U0001FA6F"  # 表情符號擴展
                           u"\U0001FA70-\U0001FAFF"  # 表情符號擴展-B
                           u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
                           u"\U00002702-\U000027B0"  # 對勾等符號
                           u"\U000024C2-\U0001F251" 
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub('', text)


def remove_punct(text):
    table = text.maketrans('', '', string.punctuation)
    return text.translate(table)

def remove_stopwords(text):
    exclusive_stopwords = [word for word in text if word not in stops]
    return exclusive_stopwords


def word_tokenize(text):
    tokenizer = RegexpTokenizer("\w+|\'+\w+", gaps = False)  
    clean_sentence = tokenizer.tokenize(text)
    return clean_sentence

def lower_case_process(text):
    lower_case = [word.lower() for word in text]
    return lower_case


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def remove_html(text):
    html = re.compile(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    return re.sub(html, '', text)
    
def lemmatized(text):
    return [wnl.lemmatize(word, tag) for word, tag in text]

def apply_data_process(df):
    df['text_clean'] = df['text'].apply(lambda x: remove_url(x))
    df['text_clean'] = df['text_clean'].apply(lambda x: remove_emojis(x))
    df['text_clean'] = df['text_clean'].apply(lambda x: remove_html(x))
    df['text_clean'] = df['text_clean'].apply(lambda x: remove_punct(x))

    df['tokenized'] = df['text_clean'].apply(lambda x: word_tokenize(x))
    # df['tokenized'] = df['text_clean'].apply(word_tokenize)

    df['lower'] = df['tokenized'].apply(lambda x:lower_case_process(x))
    df['stopwords_removed'] = df['lower'].apply(lambda x:remove_stopwords(x))

    df['pos_tag'] = df['stopwords_removed'].apply(pos_tag)


    df['wordnet_pos'] = df['pos_tag'].apply(lambda x: [(word, get_wordnet_pos(pos_tag)) for word, pos_tag in x])



    df['lemmatized'] = df['wordnet_pos'].apply(lambda x: lemmatized(x))
    df['lemmatized'] = df['lemmatized'].apply(lambda x:  [word for word in x if word not in stops])
    df['lemma_str'] = df['lemmatized'].apply(lambda x: ' '.join(x))
    # df['lemma_str'] = [' '.join(map(str, l)) for l in df['lemmatized']]


    df['Character Count'] = df['text_clean'].apply(lambda x: len(x))

    print(df.head())
    


