from utils import Processor, logger
import codecs
import json
import numpy as np
import random
import pickle
import os

processor = Processor()

def cosine_sim(vec_a, vec_b):
    vec_a = np.mat(vec_a)
    vec_b = np.mat(vec_b)
    num = float(vec_a * vec_b.T)
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    cos = num / denom
    return 0.5 + 0.5 * cos

def sequence_encoder(seq_segs, word2emb_d, stopwords):
    for w in seq_segs:
        if w[2:] in stopwords and len(seq_segs) > 1:
            seq_segs.remove(w)
    seq_repr = []
    for w in seq_segs:
        if w not in word2emb_d:
            continue
        emb = word2emb_d[w]
        seq_repr.append(emb)
    if not seq_repr:
        #logger.warning('Error sequnce for encoding: %s' % ' '.join(seq_segs))
        return None
    seq_avg = np.mean(seq_repr, axis=0)
    seq_avg /= np.linalg.norm(seq_avg)
    return seq_avg

def evaluate_emb(conf):
    test_file = conf.test_file
    emb_file = conf.pr_embedding_file
    word2emb_d = processor.load_word_embeddings(emb_file)
    stopwords = processor.load_stopwords()
    recall_20_at_1 = recall_20_at_5 = recall_20_at_10 = sample_all_num = 0
    test_fd = codecs.open(test_file, 'r', encoding='utf-8')
    for line in test_fd:
        prediction_list = []
        parts = line.strip().split('\t\t')
        query = parts[0].split('\t')[0]
        query_words = query.split()[1:]
        reply_words = parts[0].split('\t')[1].split()
        #print('query: %s\treply: %s' % (' '.join(query_words), ' '.join(reply_words)))
        cands = parts[1].split('|')[:-1]
        #print(f'len of cands: {len(cands)}')
        assert len(cands) == 19
        query_words = ['P_' + w for w in query_words]
        reply_words = ['R_' + w for w in reply_words]
        query_repr = sequence_encoder(query_words, word2emb_d, stopwords)
        reply_repr = sequence_encoder(reply_words, word2emb_d, stopwords)
        if query_repr is None or reply_repr is None:
            #logger.warning(f'Error line: {line.strip()}')
            exit(-1)
        qr_sim = cosine_sim(query_repr, reply_repr)
        #print(f'qr_sim: %f' % qr_sim)
        prediction_list.append((1, qr_sim))
        for cand in cands:
            cand_words = cand.split()
            cand_words = ['R_' + w for w in cand_words]
            cand_repr = sequence_encoder(cand_words, word2emb_d, stopwords)
            if cand_repr is None:
                continue
            qc_sim = cosine_sim(query_repr, cand_repr)
            prediction_list.append((0, qc_sim))
        
        sorted_list = sorted(prediction_list, key=lambda x: x[1], reverse=True)
        sorted_label_list = [x for x, y in sorted_list]

        if sorted_label_list[0] == 1:
            recall_20_at_1 += 1
            recall_20_at_5 += 1
            recall_20_at_10 += 1
        elif 1 in sorted_label_list[:5]:
            recall_20_at_5 += 1
            recall_20_at_10 += 1
        elif 1 in sorted_label_list[:10]:
            recall_20_at_10 += 1
        else:
            pass
        sample_all_num += 1

    recall_20_at_1 = recall_20_at_1 * 1.0 / sample_all_num
    recall_20_at_5 = recall_20_at_5 * 1.0 / sample_all_num
    recall_20_at_10 = recall_20_at_10 * 1.0 / sample_all_num

    logger.info(f"rank evaluate sample_all_num: {sample_all_num}\trecall_20_at_1: {recall_20_at_1}\t" \
          f"recall_20_at_5: {recall_20_at_5}\trecall_20_at_10: {recall_20_at_10}")
    return recall_20_at_1
