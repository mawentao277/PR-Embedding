#coding=utf-8
import sys
import random
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--original_input_file', type=str, default='./data/convai_sample')
parser.add_argument('--pr_input_file', type=str, default='./data/convai_sample.pr')
parser.add_argument('--pr_align_file', type=str, default='./data/convai_sample.align')
parser.add_argument('--align_p2r_dict', type=str, default='./data/align_p2r.dict')
parser.add_argument('--align_r2p_dict', type=str, default='./data/align_r2p.dict')
parser.add_argument("--add_pr_tag", action='store_true')
parser.add_argument('--add_alignment', action='store_true')
config = parser.parse_args()

class Aligner:
    def __init__(self):
        self.config = config
        self.p2r_dict = self.load_dict(config.align_p2r_dict)
        self.r2p_dict = self.load_dict(config.align_r2p_dict)

    def load_dict(self, res_f, reverse=False):
        res_fd = open(res_f, 'r', encoding='utf-8')
        res_d = {}
        print("Begin to load the dict...")
        num_line = num_tgt = 0
        for line in res_fd:
            num_line += 1
            segs = line.strip().split(' ')
            tgt1_token = segs[0].strip()
            tgt2_token = segs[1].strip()
            if reverse:
                res_d[(tgt2_token, tgt1_token)] = float(segs[2])
            else:
                res_d[(tgt1_token, tgt2_token)] = float(segs[2])
        res_fd.close()
        print('Load the dict[%s] line_num[%d]\ttoken_pair_num[%d]' % (res_f, num_line, len(res_d)))
        return res_d

    def find_align_index(self, post_tokens, reply_tokens, p2r_dict, reverse=False):
        p_tgt = []
        for p_w in post_tokens:
            p_preds = []
            has_score = False
            for r_w in reply_tokens:
                key = None
                if reverse:
                    key = (r_w, p_w)
                else:
                    key = (p_w, r_w)
                if key in p2r_dict:
                    score = p2r_dict[key]
                    p_preds.append(score)
                    has_score = True
                else:
                    p_preds.append(0)
            if has_score:
                p_tgt.append(p_preds.index(max(p_preds)))
            else:
                p_tgt.append(-1)
        return p_tgt

    def insert_align_pr(self, res_f, out_f):
        res_fd = open(res_f, 'r', encoding='utf-8')
        out_fd = open(out_f, 'w', encoding='utf-8')
        for line in res_fd:
            #print(line.strip())
            post, reply = line.strip().split('\t')
            post_tokens = post.strip().split(' ')
            reply_tokens = reply.strip().split(' ')
            p_tgt = self.find_align_index(post_tokens, reply_tokens, self.p2r_dict)
            r_tgt = self.find_align_index(reply_tokens, post_tokens, self.r2p_dict)
            p_str = ' '.join(map(str, p_tgt))
            r_str = ' '.join(map(str, r_tgt))
            out = '%s\t%s\t%s\t%s\n' % (post, reply, p_str, r_str)
            out_fd.write(out)
        res_fd.close()
        out_fd.close()

class PRTagger:
    def __init__(self, post_tag="P_", reply_tag="R_"):
       self.post_tag  = post_tag
       self.reply_tag = reply_tag

    def add_pr_tag(self, infile, outfile):
        in_fd = open(infile, 'r', encoding='utf-8')
        out_fd = open(outfile, 'w', encoding='utf-8')
        for line in in_fd:
            segs = line.strip().split("\t")
            post = segs[0].strip().split(" ")
            reply = segs[1].strip().split(" ")
            post_tag = [self.post_tag + w for w in post]
            reply_tag = [self.reply_tag + w for w in reply]
            out = '%s\t%s\n' % (" ".join(post_tag), " ".join(reply_tag))
            out_fd.write(out)
        in_fd.close()
        out_fd.close()

def main():
    if config.add_pr_tag:
        pr_tagger = PRTagger()
        pr_tagger.add_pr_tag(config.original_input_file, config.pr_input_file)
    if config.add_alignment:
        aligner = Aligner()
        aligner.insert_align_pr(config.pr_input_file, config.pr_align_file)
    if not config.add_pr_tag and not config.add_alignment:
        raise ValueError("You must set preprocess function add_pr_tag or add_alignment!!!")

if __name__ == '__main__':
    main()
