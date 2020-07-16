#coding=utf-8
import sys
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import codecs
import random
import argparse
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--pr_input_file', type=str, default='./data/convai_sample.pr')
parser.add_argument('--test_file', type=str, default='./data/valid_none_original.txt')
parser.add_argument('--pr_embedding_file', type=str, default='./output/pr_embedding.txt')
parser.add_argument('--vocab_file', type=str, default='./data/vocab.align')
parser.add_argument('--train_file', type=str, default='./data/train.txt')
parser.add_argument("--do_train", action='store_true')
parser.add_argument("--do_eval", action='store_true')
parser.add_argument('--valid_file', type=str, default='./data/dev.txt')
parser.add_argument('--stopword_file', type=str, default='./data/stopwords.txt')
parser.add_argument('--model_weights', type=str, default='./model/model_epoch18.hdf5')
parser.add_argument('--word_level_embedding', type=str, default='./output/word_level_learning.txt')
parser.add_argument('--output_model_path', type=str, default='./model')
parser.add_argument('--post_maxlen', type=int, default=15)
parser.add_argument('--reply_maxlen', type=int, default=15)
parser.add_argument('--embedding_size', type=int, default=500)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--filters_num', type=int, default=50)
parser.add_argument('--kernel_shape', type=int, default=3)
config = parser.parse_args()

class Processor:
    def __init__(self):
        self.stopword_file = config.stopword_file
        self.emb_size = config.embedding_size
        self.post_maxlen = config.post_maxlen
        self.reply_maxlen = config.reply_maxlen
        self.word2id_p, self.id2word_p = self.load_vocab(fname=config.vocab_file, pre_str='P_')
        self.word2id_r, self.id2word_r = self.load_vocab(fname=config.vocab_file, pre_str='R_')
        self.pr_input_file = config.pr_input_file
        self.train_file = config.train_file
        self.dev_file = config.valid_file

    def load_stopwords(self):
        vocab_file = codecs.open(self.stopword_file, 'r', encoding='utf-8')
        stopword_list = []
        for word in vocab_file:
            stopword_list.append(word.strip())
        return stopword_list

    def load_word_embeddings(self, fname):
        logger.info('Loading pre-trained Glove embeddings......')
        embeddings_idx = {}
        count = 0 
        embeddings_file = codecs.open(fname, 'r', encoding='utf-8')
        for line in embeddings_file:
            count += 1
            line_split = line.replace('\n', '').strip().split(' ')
            assert len(line_split) == self.emb_size + 1 
            word = line_split[0]
            embedding = np.asarray(line_split[1:], dtype='float32')
            embeddings_idx[word] = embedding
        embeddings_file.close()
        logger.info(f'Loading {count} words in embeddings file')
        return embeddings_idx

    def pretrain_embedding(self, fname, word2id_d):
        logger.info('Loading pre-trained embedding from file %s...' % fname)
        embeddings_idx = {}
        f = codecs.open(fname, encoding='utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            start_index = len(values) - config.embedding_size
            coefs = np.asarray(values[start_index:], dtype='float32')
            embeddings_idx[word] = coefs
        f.close()

        #
        cnt = 0
        vocab_size = len(word2id_d.values()) + 1
        embedding_matrix = np.zeros((vocab_size, config.embedding_size))
        logger.info('vocab size: %d' % (vocab_size))
        logger.info('word2id_d size: %d' % (len(word2id_d)))
        logger.info('embedding_matrix shape: {}'.format(embedding_matrix.shape))
        for word, i in word2id_d.items():
            #print('word: %s\tidx: %d' % (word, i))
            embedding_vector = embeddings_idx.get(word)
            if embedding_vector is not None:
                cnt += 1
                embedding_matrix[i] = embedding_vector
        logger.info('Load {} words from the file {}'.format(str(cnt), fname))
        return embedding_matrix

    def load_vocab(self, fname, pre_str):
        vb_fd = codecs.open(fname, 'r', encoding='utf-8')
        word2id_dict = {}
        id2word_dict = {}
        word_list = vb_fd.readlines()
        logger.info('Building the vocab from the file: %s' % fname)
        word_idx = 0
        for line in word_list:
            segs = line.strip().split(" ")
            if len(segs) < 2:
                continue
            word = segs[0]
            if not word.startswith(pre_str):
                continue
            word_idx += 1 
            word2id_dict[word] = word_idx
            id2word_dict[word_idx] = word
        word = pre_str + '<unk>'
        word_idx += 1 
        word2id_dict[word] = word_idx
        id2word_dict[word_idx] = word
        #assert len(word_list) == len(word2id_dict)
        logger.info('line num: %d\tvocab_size: %d' % (len(word_list), len(word2id_dict)))
        return word2id_dict, id2word_dict

    def create_dataset_for_training(self):
        res_fd = codecs.open(self.pr_input_file, 'r', encoding='utf-8')
        train_fd = codecs.open(self.train_file, 'w', encoding='utf-8')
        dev_fd   = codecs.open(self.dev_file, 'w', encoding='utf-8')
        qa_pair = []
        for line in res_fd:
            segs = line.strip().split('\t')
            query = segs[0]
            reply = segs[1]
            qa_pair.append((query, reply))
        random.shuffle(qa_pair)
        for idx, pair in enumerate(qa_pair):
            query, reply = pair
            rand_r = None
            while(1):
                random_idx = random.randint(0, len(qa_pair)-1)
                rand_q, rand_r = qa_pair[random_idx]
                if rand_r != reply:
                    break
            out_p = '%s\t%s\t1\n' % (query, reply)
            out_n = '%s\t%s\t0\n' % (query, rand_r)
            if idx < len(qa_pair) * 0.9:
                train_fd.write(out_p)
                train_fd.write(out_n)
            else:
                dev_fd.write(out_p)
                dev_fd.write(out_n)
        res_fd.close()
        train_fd.close()
        dev_fd.close()

    def word2id_sequence(self, qa_seq, word2id_dict, maxlen, pre_str):
        segs = qa_seq.strip().split()
        id_list = []
        for w in segs:
            if w in word2id_dict:
                id = word2id_dict[w]
                id_list.append(id)
            else:
                id = word2id_dict[pre_str+'<unk>']
                id_list.append(id)
        if len(id_list) > maxlen:
            id_list = id_list[:maxlen]
        else:
            while len(id_list) < maxlen:
                id_list.append(0)
        return id_list

    def init_for_train(self, fname):
        res_fd = codecs.open(fname, 'r', encoding='utf-8')
        query_mtx = []
        reply_mtx = []
        label_l = []
        for line in res_fd:
            segs = line.strip().split('\t')
            query = segs[0].strip()
            reply = segs[1].strip()
            label = int(segs[2])
            query_wids = self.word2id_sequence(query, self.word2id_p, maxlen=config.post_maxlen, pre_str='P_')
            reply_wids = self.word2id_sequence(reply, self.word2id_r, maxlen=config.reply_maxlen, pre_str='R_')
            query_mtx.append(query_wids)
            reply_mtx.append(reply_wids)
            label_l.append(label)
        ret_q = np.asarray(query_mtx)
        ret_r = np.asarray(reply_mtx)
        ret_y = np.asarray(label_l)
        return ret_q, ret_r, ret_y

    def dump_embedding(self, model):
        p_embeddings = model.get_layer(name='post_embedding').get_weights()[0]
        r_embeddings = model.get_layer(name='reply_embedding').get_weights()[0]

        emb_file = codecs.open(config.pr_embedding_file, 'w', encoding='utf-8')
        for w, idx in self.word2id_p.items():
            if w.find(' ') != -1:
                w = w.replace(' ', '')
            p_emb = p_embeddings[idx]
            out_s = '%s %s\n' % (w, ' '.join(map(str, p_emb)))
            emb_file.write(out_s)

        for w, idx in self.word2id_r.items():
            if w.find(' ') != -1:
                w = w.replace(' ', '')
            r_emb = r_embeddings[idx]
            out_s = '%s %s\n' % (w, ' '.join(map(str, r_emb)))
            emb_file.write(out_s)

if __name__ == '__main__':
    processor = Processor()
    processor.create_dataset_for_training()
