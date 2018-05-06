from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.utils import resample
from sklearn.svm import LinearSVC
from nltk.stem.snowball import SnowballStemmer
from imblearn.over_sampling import SMOTE
from xlrd import open_workbook
import pandas as pd
import numpy as np
import argparse
import csv
import re
import nltk

stemmer = SnowballStemmer("english")
mystop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ourselves', 'you', 'your', 'yourself', 'yourselves', 'he',
                'him', 'his', 'himself', 'she', 'her', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
                'themselves', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'if', 'or', 'as',
                'until', 'of', 'at', 'by', 'between', 'into', 'through', 'during', 'to', 'from', 'in', 'out', 'on',
                'off', 'then', 'once', 'here', 'there', 'all', 'any', 'both', 'each', 'few', 'more', 'other', 'some',
                'such', 'than', 'too', 'very', 's', 't', 'can', 'will', 'don', 'should', 'now', 'while', 'case',
                'switch', 'def', 'abstract', 'byte', 'continue', 'native', 'private', 'synchronized', 'if', 'do',
                'include', 'each', 'than', 'finally', 'class', 'double', 'float', 'int', 'else', 'instanceof',
                'long', 'super', 'import', 'short', 'default', 'catch', 'try', 'new', 'final', 'extends', 'implements',
                'public', 'protected', 'static', 'this', 'return', 'char', 'const', 'break', 'boolean', 'bool',
                'package', 'byte', 'assert', 'raise', 'global', 'with', 'or', 'yield', 'in', 'out', 'except', 'and',
                'enum', 'signed', 'void', 'virtual', 'union', 'goto', 'var', 'function', 'require', 'print', 'echo',
                'foreach', 'elseif', 'namespace', 'delegate', 'event', 'override', 'struct', 'readonly', 'explicit',
                'interface', 'get', 'set', 'elif', 'for', 'throw', 'throws', 'lambda', 'endfor', 'endforeach', 'endif',
                'endwhile', 'clone']


def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text


def stem_tokens(tokens):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize_and_stem(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens)
    return stems


# logging.basicConfig(level=logging.INFO,
#                    format='%(asctime)s %(levelname)s %(message)s')


emodict = []
contractions_dict = []

# Read in the words with sentiment from the dictionary
with open("lib\external\SentiCR\Contractions.txt", "r") as contractions, \
        open("lib\external\SentiCR\EmoticonLookupTable.txt", "r") as emotable:
    contractions_reader = csv.reader(contractions, delimiter='\t')
    emoticon_reader = csv.reader(emotable, delimiter='\t')

    # Hash words from dictionary with their values
    contractions_dict = {rows[0]: rows[1] for rows in contractions_reader}
    emodict = {rows[0]: rows[1] for rows in emoticon_reader}

    contractions.close()
    emotable.close()

grammar = r"""
NegP: {<VERB>?<ADV>+<VERB|ADJ>?<PRT|ADV><VERB>}
{<VERB>?<ADV>+<VERB|ADJ>*<ADP|DET>?<ADJ>?<NOUN>?<ADV>?}

"""
chunk_parser = nltk.RegexpParser(grammar)

contractions_regex = re.compile('(%s)' % '|'.join(contractions_dict.keys()))


def expand_contractions(s, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]

    return contractions_regex.sub(replace, s.lower())


url_regex = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')


def remove_url(s):
    return url_regex.sub(" ", s)


negation_words = ['not', 'never', 'none', 'nobody', 'nowhere', 'neither', 'barely', 'hardly',
                  'nothing', 'rarely', 'seldom', 'despite']

emoticon_words = ['PositiveSentiment', 'NegativeSentiment']


def negated(input_words):
    """
    Determine if input contains negation words
    """
    neg_words = []
    neg_words.extend(negation_words)
    for word in neg_words:
        if word in input_words:
            return True
    return False


def prepend_not(word):
    if word in emoticon_words:
        return word
    elif word in negation_words:
        return word
    return "NOT_" + word


def handle_negation(comments):
    sentences = nltk.sent_tokenize(comments)
    modified_st = []
    for st in sentences:
        allwords = nltk.word_tokenize(st)
        modified_words = []
        if negated(allwords):
            part_of_speech = nltk.tag.pos_tag(allwords, tagset='universal')
            chunked = chunk_parser.parse(part_of_speech)
            # print("---------------------------")
            # print(st)
            for n in chunked:
                if isinstance(n, nltk.tree.Tree):
                    words = [pair[0] for pair in n.leaves()]
                    # print(words)

                    if n.label() == 'NegP' and negated(words):
                        for i, (word, pos) in enumerate(n.leaves()):
                            if (pos == "ADV" or pos == "ADJ" or pos == "VERB") and (word != "not"):
                                modified_words.append(prepend_not(word))
                            else:
                                modified_words.append(word)
                    else:
                        modified_words.extend(words)
                else:
                    modified_words.append(n[0])
            newst = ' '.join(modified_words)
            # print(newst)
            modified_st.append(newst)
        else:
            modified_st.append(st)
    return ". ".join(modified_st)


def preprocess_text(text):
    # comments = text.encode('utf-8')
    comments = expand_contractions(text)
    comments = remove_url(comments)
    comments = replace_all(comments, emodict)
    comments = handle_negation(comments)

    return comments


class SentimentData:
    def __init__(self, text, rating):
        self.text = text
        self.rating = rating


class SentiCR:
    def __init__(self, algo="GBT", training_data=None):
        self.algo = algo
        if (training_data is None):
            self.training_data = self.read_data_from_oracle()
        else:
            self.training_data = training_data
        self.model = self.create_model_from_training_data()

    def get_classifier(self):
        algo = self.algo

        if algo == "GBT":
            return GradientBoostingClassifier()
        elif algo == "RF":
            return RandomForestClassifier()
        elif algo == "ADB":
            return AdaBoostClassifier()
        elif algo == "DT":
            return DecisionTreeClassifier()
        elif algo == "NB":
            return BernoulliNB()
        elif algo == "SGD":
            return SGDClassifier()
        elif algo == "SVC":
            return LinearSVC()
        elif algo == "MLPC":
            return MLPClassifier(activation='logistic', batch_size='auto',
                                 early_stopping=True, hidden_layer_sizes=(100,), learning_rate='adaptive',
                                 learning_rate_init=0.1, max_iter=5000, random_state=1,
                                 solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
                                 warm_start=False)
        return 0

    def create_model_from_training_data(self):
        training_comments = []
        training_ratings = []
        print("Training classifier model..")
        for sentidata in self.training_data:
            comments = preprocess_text(sentidata.text)
            training_comments.append(comments)
            training_ratings.append(sentidata.rating)

        # discard stopwords, apply stemming, and discard words present in less than 3 comments
        self.vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem, sublinear_tf=True, max_df=0.5,
                                          stop_words=mystop_words, min_df=3)
        X_train = self.vectorizer.fit_transform(training_comments).toarray()
        Y_train = np.array(training_ratings)
        # Apply SMOTE to improve ratio of the minority class
        smote_model = SMOTE(ratio='auto', random_state=None, k=None, k_neighbors=15, m=None, m_neighbors=15,
                            out_step=.0001,
                            kind='regular', svm_estimator=None, n_jobs=1)

        X_resampled, Y_resampled = smote_model.fit_sample(X_train, Y_train)

        model = self.get_classifier()
        model.fit(X_resampled, Y_resampled)

        return model

    def read_data_from_oracle(self):
        workbook = open_workbook("oracle.xlsx")
        sheet = workbook.sheet_by_index(0)
        oracle_data = []
        print("Reading data from oracle..")
        for cell_num in range(0, sheet.nrows):
            comments = SentimentData(sheet.cell(cell_num, 0).value, sheet.cell(cell_num, 1).value)
            oracle_data.append(comments)
        return oracle_data

    def get_sentiment_polarity(self, text):
        comment = preprocess_text(text)
        feature_vector = self.vectorizer.transform([comment]).toarray()
        sentiment_class = self.model.predict(feature_vector)
        return sentiment_class

    def get_sentiment_polarity_collection(self, texts):
        predictions = []
        for text in texts:
            comment = preprocess_text(text)
            feature_vector = self.vectorizer.transform([comment]).toarray()
            sentiment_class = self.model.predict(feature_vector)
            predictions.append(sentiment_class)

        return predictions


def cross_val(dataset, data_y, classifier, num_classes=2, n_splits=10):
    skf = StratifiedKFold(n_splits, shuffle=True,random_state=157)
    precision_list = [0 for i in range(num_classes)]
    recall_list = [0 for i in range(num_classes)]
    f1_list = [0 for i in range(num_classes)]
    mean_accuracy = 0
    count = 1
    for train, test in skf.split(dataset, data_y):
        print("Iteration %d of %d" % (count, n_splits))
        classifier_model = SentiCR(algo=classifier, training_data=dataset[train])
        test_comments = [comments.text for comments in dataset[test]]
        test_ratings = [comments.rating for comments in dataset[test]]
        pred = classifier_model.get_sentiment_polarity_collection(test_comments)
        metrics = {"individual": precision_recall_fscore_support(test_ratings, pred),
                   "micro-average": precision_recall_fscore_support(test_ratings, pred, average="micro")}
        precision_list = [x + y for x, y in zip(metrics['individual'][0], precision_list)]
        recall_list = [x + y for x, y in zip(metrics['individual'][1], recall_list)]
        mean_accuracy += metrics['micro-average'][0]
        count += 1
        print("Precision, Recall, F_Score, Support")
        print(metrics)

    print("Mean accuracy: %s Mean precision: %s, Mean recall: %s" % (mean_accuracy/n_splits, [precision/n_splits for precision in precision_list], [recall/n_splits for recall in recall_list]))
    return mean_accuracy, precision_list, recall_list


def bootstrap_trend(dataset, classifier, num_classes):
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=157)
    test_x = [comments.text for comments in test_dataset]
    test_y = [comments.rating for comments in test_dataset]
    precision_list = [0 for i in range(num_classes)]
    recall_list = [0 for i in range(num_classes)]
    f1_list = [0 for i in range(num_classes)]
    accuracy_list = list()

    for sample_rate in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        n_samples = int(sample_rate * len(train_dataset) + 1)
        train_xr = resample(train_dataset, n_samples=n_samples, random_state=157)
        classifier_model = SentiCR(algo=classifier, training_data=train_xr)
        predict_y = classifier_model.get_sentiment_polarity_collection(test_x)
        metrics = {"individual": precision_recall_fscore_support(test_y, predict_y),
                   "micro-average": precision_recall_fscore_support(test_y, predict_y, average="micro")}
        print("Accuracy: %s, Precision: %s, Recall: %s, F1: %s" % (metrics['micro-average'][0], metrics['individual'][0],
                                                                   metrics['individual'][1], metrics['individual'][2]))
        precision_list = [x + y for x, y in zip(metrics['individual'][0], precision_list)]
        recall_list = [x + y for x, y in zip(metrics['individual'][1], recall_list)]
        f1_list = [x + y for x, y in zip(metrics['individual'][2], f1_list)]
        accuracy_list.append(metrics['micro-average'][0])

    print("Accuracies:", accuracy_list)
    print("Dataset sizes:", [int(sample_rate * len(train_dataset) + 1) for sample_rate in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
    print("Mean accuracy: %s Mean precision: %s, Mean recall: %s, Mean F1: %s" % (sum(accuracy_list)/9, [precision/9 for precision in precision_list],
                                                                                  [recall/9 for recall in recall_list], [f1/9 for f1 in f1_list]))


def cross_dataset(data_list, classifier, num_classes):
    precision_list = [0 for i in range(num_classes)]
    recall_list = [0 for i in range(num_classes)]
    f1_list = [0 for i in range(num_classes)]
    accuracy_list = list()

    # Uncomment for resampling:
    # for i0 in range(len(data_list)):
    #     data_list[i0] = resample(data_list[i0], n_samples=1500, random_state=157, replace=False)

    for i0 in range(len(data_list)):
        train_dataset = data_list[i0]
        classifier_model = SentiCR(algo=classifier, training_data=train_dataset)
        for i1 in range(len(data_list)):
            if i1 != i0:
                test_dataset = data_list[i1]
                test_x = [comments.text for comments in test_dataset]
                test_y = [comments.rating for comments in test_dataset]
                predict_y = classifier_model.get_sentiment_polarity_collection(test_x)
                metrics = {"individual": precision_recall_fscore_support(test_y, predict_y),
                           "micro-average": precision_recall_fscore_support(test_y, predict_y, average="micro")}
                accuracy_list.append(metrics['micro-average'][0])
                precision_list = [x + y for x, y in zip(metrics['individual'][0], precision_list)]
                recall_list = [x + y for x, y in zip(metrics['individual'][1], recall_list)]
                f1_list = [x + y for x, y in zip(metrics['individual'][2], f1_list)]
                print(i0, i1, "Accuracy: %s, Precision: %s, Recall: %s, F1: %s" % (metrics['micro-average'][0], metrics['individual'][0],
                                                                                   metrics['individual'][1], metrics['individual'][2]))
    print("Mean accuracy: %s Mean precision: %s, Mean recall: %s, Mean F1: %s" % (sum(accuracy_list)/len(accuracy_list),
                                                                                  [precision/len(accuracy_list) for precision in precision_list],
                                                                                  [recall/len(accuracy_list) for recall in recall_list],
                                                                                  [f1/len(accuracy_list) for f1 in f1_list]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Supervised sentiment classifier')
    parser.add_argument('--algo', type=str,
                        help='Classification algorithm', default="GBT")
    parser.add_argument('--repeat', type=int,
                        help='Iteration count', default=1)

    args = parser.parse_args()
    classifier = args.algo

    # workbook = open_workbook("oracle.xlsx")
    # sheet = workbook.sheet_by_index(0)
    # for cell_num in range(0, sheet.nrows):
    #     comments = SentimentData(str(sheet.cell(cell_num, 0)), sheet.cell(cell_num, 1).value)
    #     oracle_data.append(comments)
    #     oracle_y.append(sheet.cell(cell_num, 1).value)

    data = list()
    data_y = list()
    # raw_data = pd.read_csv("data/labelled/Gerrit.csv").as_matrix()
    # raw_data = pd.read_csv("data/labelled/StackOverflow.csv", encoding='latin1').as_matrix()
    data_1 = pd.read_csv("data/labelled/Gerrit.csv")
    data_2 = pd.read_csv("data/labelled/JIRA.csv")
    data_3 = pd.read_csv("data/labelled/StackOverflow2.csv", encoding='latin1')
    # raw_data = pd.concat([data_1, data_2, data_3]).as_matrix()
    # for item in raw_data:
    #     comment = SentimentData(str(item[0]), item[1])
    #     data.append(comment)
    #     data_y.append(item[1])

    # data = np.array(data)
    # cross_val(data, data_y, classifier, num_classes=2)
    # bootstrap_trend(data, classifier, num_classes=2)