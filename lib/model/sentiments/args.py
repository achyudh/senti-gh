from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description="Machine learning models for Replica-GH.")
    parser.add_argument('--model', type=str, default='HiCNNBiLSTM',
                        choices=['HiCNNBiLSTM', 'HybridCNNBiLSTM'])
    parser.add_argument('--dataset', type=str, default='IMDB',
                        choices=['IMDB', 'Gerrit', 'Jira', 'AppReviews', 'SOJava', 'SOSentiments'])
    parser.add_argument('--gpu', type=int, default=0)  # If the value is -1, use CPU
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=3435)
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--load-model', action='store_true')
    parser.add_argument('--transfer-learn', action='store_true')
    parser.add_argument('--load-path', type=str, default='data/checkpoints/HiCNNBiLSTM-IMDB.hdf5')
    parser.add_argument('--checkpoint-path', type=str, default='data/checkpoints')
    parser.add_argument('--embedding-dim', type=int, default=300)
    parser.add_argument('--dropout-rate', type=float, default=0.5)
    parser.add_argument('--k-fold', type=int, default=10)

    args = parser.parse_args()
    return args