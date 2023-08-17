import warnings
import jieba_fast
from transformers import BertTokenizer
from transformers import BigBirdModel


class JiebaTokenizer(BertTokenizer):
    def __init__(
            self, pre_tokenizer=lambda x: jieba_fast.cut(x, HMM=False), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_tokenizer = pre_tokenizer

    def _tokenize(self, text, *arg, **kwargs):
        split_tokens = []
        for word in self.pre_tokenizer(text):
            if word in self.vocab:
                split_tokens.append(word)
            else:
                split_tokens.extend(super()._tokenize(word))
        return split_tokens


class DefaultConfig(object):
    BERT_PATH = 'bert_models/chinese-bigbird-base-4096'
    dataset_name = 'CMT'
    class_num = 12
    root_dir_train = 'dataset/' + dataset_name + '/data_train'
    root_dir_test = 'dataset/' + dataset_name + '/data_test'
    feature_train = './dataset/' + dataset_name + '/' + dataset_name + '_data_train.csv'
    feature_test = './dataset/' + dataset_name + '/' + dataset_name + '_data_test.csv'
    # just default
    batch_size = 1
    train_batch_size = batch_size
    test_batch_size = batch_size
    seed = 1
    annealing_factor = 0.35
    lr = 0.00002
    weight_decay = 0.01
    max_epoch = 20
    schedule = True
    warmup_steps = 0.1 * max_epoch
    max_text_length = 2048  # the maximum length of text
    max_sentence_length = 128  # the maximum length of sentences
    max_sentence_num = 160  # Maximum number of sentences
    feature_num = 21  # Linguistic feature number
    feature_index = [0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 30, 31, 40, 41, 46, 50, 51, 52, 53,
                     66]  # List index of linguistic features

    def parse(self, kwargs):

        for key, value in kwargs.items():
            if not hasattr(self, key):
                warnings.warn("Warnning: has not attribut %s" % key)
            setattr(self, key, value)

        print()
        print("config:")
        print("{")
        for key, value in self.__class__.__dict__.items():
            if not key.startswith('__'):
                if key != 'bert_hidden_size':
                    print('     ', key, ' = ', getattr(self, key))
        print("}")
        print()
