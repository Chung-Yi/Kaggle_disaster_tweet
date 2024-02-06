import nltk
from nltk.corpus import stopwords
from utils.data_process import read_csv_file, remove_url, remove_emojis, remove_punct, apply_data_process
from utils.draw_utils import draw, plot_word_number_histogram, plot_word_len_histogram, ngrams, plot_wordcloud, plot_name_entity_barchart

TRAIN_FILE = 'data/train.csv'
TEST_FILE = 'data/test.csv'


nltk = nltk.download('stopwords')
stops = set(stopwords.words('english'))

def main():
    train_data = read_csv_file(TRAIN_FILE)
    test_data = read_csv_file(TEST_FILE)

    # print(train_data.head())
    # print(stops)

    # new_sentence = remove_emojis("this 😄 的例句。")
    # new_sentence = remove_punct("this 😄 的例句。")

    apply_data_process(train_data)
    # draw(train_data[train_data['target']==1], 'Character Count', chart_type='', marginal='box')
    # plot_word_number_histogram(train_data)
    # plot_word_len_histogram(train_data)
    # ngrams(train_data, stops)
    # plot_wordcloud(train_data[train_data['target']==1]['lemmatized'], 'Most Common Words in Disaster Tweets', title_size=20)
    # plot_dist3(train_data[train_data['target']==0], 'Character Count', '')

    plot_name_entity_barchart(train_data)


    # print(test_data.head())
    
    




if __name__ == '__main__':
    main()