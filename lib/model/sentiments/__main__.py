import os
from copy import deepcopy

import pandas as pd
from sklearn.model_selection import train_test_split

from lib.embedding import word2vec
from lib.model.sentiments import HiCNNBiLSTM, HybridCNNBiLSTM
from lib.model.sentiments.args import get_args
from lib.util import preprocessing

if __name__ == '__main__':
    # Select GPU based on args
    args = get_args()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    model_config = deepcopy(args)

    # Set random seeds for Tensorflow and NumPy
    from tensorflow import set_random_seed
    from numpy.random import seed

    set_random_seed(args.seed)
    seed(args.seed)

    if args.dataset == 'IMDB':
        model_config.num_classes = 2
        raw_data = pd.read_csv('data/labelled/IMDB.tsv', delimiter='\t')
    elif args.dataset == 'Jira':
        model_config.num_classes = 2
        raw_data = pd.read_csv('data/labelled/JIRA.csv')
    elif args.dataset == 'Gerrit':
        model_config.num_classes = 2
        raw_data = pd.read_csv('data/labelled/Gerrit.csv')
    elif args.dataset == 'AppReviews':
        model_config.num_classes = 3
        raw_data = pd.read_csv('data/labelled/AppReviews.csv')
    elif args.dataset == 'SOJava':
        model_config.num_classes = 3
        raw_data = pd.read_csv('data/labelled/StackOverflowJavaLibraries.csv', encoding='latin1')
    elif args.dataset == 'SOSentiments':
        model_config.num_classes = 3
        raw_data = pd.read_csv('data/labelled/StackOverflowSentiments.csv', encoding='latin1')
    else:
        raise Exception('Unsupported dataset')

    if args.model == 'HiCNNBiLSTM':
        model = HiCNNBiLSTM
    elif args.model == 'HybridCNNBiLSTM':
        model = HybridCNNBiLSTM
    else:
        raise Exception('Unsupported model')

    if args.transfer_learn:
        # source_data = pd.read_csv('data/labelled/StackOverflowSentiments.csv', encoding='latin1')
        source_data = pd.read_csv('data/labelled/IMDB.tsv', delimiter='\t', names=raw_data.columns.values)
        _, _, tokenizer, max_sequence_len, max_sequences = preprocessing.make_hierarchical_network_ready(
            pd.concat([source_data, raw_data]).as_matrix(),
            model_config.num_classes,
            max_sequence_len=150,
            max_sequences=15,
            enforce_max_len=True)
        source_data_x, source_data_y_cat, tokenizer, max_sequence_len, max_sequences = preprocessing.make_hierarchical_network_ready(
            source_data.as_matrix(),
            model_config.num_classes,
            tokenizer=tokenizer,
            max_sequence_len=150,
            max_sequences=15,
            enforce_max_len=True)
        data_x, data_y_cat, tokenizer, max_sequence_len, max_sequences = preprocessing.make_hierarchical_network_ready(
            raw_data.as_matrix(),
            model_config.num_classes,
            tokenizer=tokenizer,
            max_sequence_len=150,
            max_sequences=15,
            enforce_max_len=True)
        embedding_map = word2vec.embedding_matrix(tokenizer.word_index,
                                                  model_path='data/embedding/word2vec/googlenews_size300.bin',
                                                  binary=True)
        model_config.max_sequences = max_sequences
        model_config.max_sequence_len = max_sequence_len
        model.transfer_learn(data_x, data_y_cat, source_data_x, source_data_y_cat, embedding_map, tokenizer, model_config, n_splits=args.k_fold)

    else:
        data_x, data_y_cat, tokenizer, max_sequence_len, max_sequences = preprocessing.make_hierarchical_network_ready(
            raw_data.as_matrix(),
            model_config.num_classes)
        embedding_map = word2vec.embedding_matrix(tokenizer.word_index,
                                                  model_path='data/embedding/word2vec/googlenews_size300.bin',
                                                  binary=True)
        model_config.max_sequences = max_sequences
        model_config.max_sequence_len = max_sequence_len
        print('Dataset loaded to memory. Size:', len(data_y_cat))

        model.cross_val(data_x, data_y_cat, embedding_map, tokenizer, model_config, n_splits=args.k_fold)
        # bootstrap_trend(data_x, data_y_cat, embedding_map, tokenizer, model_config)
