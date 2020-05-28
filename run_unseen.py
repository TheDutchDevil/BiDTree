import os
from data_utils import get_trimmed_glove_vectors, load_vocab, \
    get_processing_word, CoNLLDataset, DepsDataset, \
    get_processing_relation
from general_utils import get_logger
from model import DepsModel
from config import Config
from itertools import chain
from build_data import build_data
import argparse

'''
Modificatin of the code in main.py that uses a pre-trained model to provide
predictions on unseen data. 

First the script reads the existing preprocessing results (as we need to use the same vocab) 
and then it reads in the unseen data.  
'''

def parse_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help="path to the data on which predictions should be given. If files are under a/data.json and a/data.txt the value should be a/data")
    parser.add_argument('--trained_on', type=str, default="laptops_2014", help="dataset on which the model has been trained.")

    args, _ = parser.parse_known_args()
    return args

def config_from_args(args):
    args.data_sets = args.trained_on

    args.txt_file = "{}.txt".format(args.data)
    args.deps_file = "{}.json".format(args.data)
    
    config = Config()
    for key, value in vars(args).items():
        config.__dict__[key] = value

    
    config.auto_config()
    logger = get_logger(config.log_path)
    return config, logger

if __name__ == "__main__":
    args = parse_parameters()
    config, logger = config_from_args(args)

    # load vocabs
    vocab_words = load_vocab(config.words_filename)
    vocab_tags = load_vocab(config.tags_filename)
    vocab_chars = load_vocab(config.chars_filename)
    vocab_relations = load_vocab(config.relations_filename)

    # get processing functions
    processing_word = get_processing_word(vocab_words, vocab_chars, lowercase=config.lowercase, chars=config.chars)
    processing_tag = get_processing_word(vocab_tags, lowercase=False)
    processing_relation = get_processing_relation(vocab_relations)

    # get pre trained embeddings
    embeddings = get_trimmed_glove_vectors(config.trimmed_filename)

    # create dataset
    data_in_CoNLL = CoNLLDataset(config.txt_file, processing_word, processing_tag=processing_tag)

    dev = CoNLLDataset(config.dev_filename, processing_word, processing_tag=processing_tag)
    test = CoNLLDataset(config.test_filename, processing_word, processing_tag=processing_tag)
    train = CoNLLDataset(config.train_filename, processing_word, processing_tag=processing_tag)

    data = [dev, test, train]

    for row in data_in_CoNLL:
        pass

    _ = map(len, chain.from_iterable(w for w in (s for s in data)))
    max_sentence_size = max(train.max_words_len, dev.max_words_len, test.max_words_len)
    max_word_size = max(train.max_chars_len, test.max_chars_len, dev.max_chars_len)

    processing_word = get_processing_word(vocab_words, lowercase=config.lowercase)
    dev_deps = DepsDataset(config.dev_deps_filename, processing_word, processing_relation)
    test_deps = DepsDataset(config.test_deps_filename, processing_word, processing_relation)
    train_deps = DepsDataset(config.train_deps_filename, processing_word, processing_relation)    

    deps_data = DepsDataset(config.deps_file, processing_word, processing_relation)

    data = [dev_deps, test_deps, train_deps]
    _ = map(len, chain.from_iterable(w for w in (s for s in data)))
    max_btup_deps_len = max(dev_deps.max_btup_deps_len, test_deps.max_btup_deps_len, train_deps.max_btup_deps_len)
    max_upbt_deps_len = max(dev_deps.max_upbt_deps_len, test_deps.max_upbt_deps_len, train_deps.max_upbt_deps_len)

    # build model
    config.ntags = len(vocab_tags)
    config.nwords = len(vocab_words)
    config.nchars = len(vocab_chars)
    config.nrels = len(vocab_relations)
    config.max_sentence_size = max_sentence_size
    config.max_word_size = max_word_size
    config.max_btup_deps_len = max_btup_deps_len
    config.max_upbt_deps_len = max_upbt_deps_len
    model = DepsModel(config, embeddings, logger=logger)
    model.build()

    print len(vocab_tags)
    print len(vocab_words)
    print len(vocab_chars)
    print len(vocab_relations)
    print max_sentence_size
    print max_word_size
    print max_btup_deps_len
    print max_upbt_deps_len


    logger.info("Max sentence length of train data is {}, current max sent length is {}".format(max_sentence_size, data_in_CoNLL.max_words_len))

    model.predict(data_in_CoNLL, deps_data, vocab_words, vocab_tags)
