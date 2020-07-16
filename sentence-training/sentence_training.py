from __future__ import print_function
from keras.callbacks import Callback
from keras.models import Model
from keras import layers as ly
from utils import Processor, logger, config
from evaluate import evaluate_emb
import sys
import os

processor = Processor()

class SaveModel(Callback):
    def __init__(self, model):
        self.model = model
    
    def on_epoch_end(self, epoch, log={}):
        if not os.path.isdir(config.output_model_path):
            os.makedirs(config.output_model_path)
        model_path = config.output_model_path + '/model_epoch' + str(epoch) + '.hdf5'
        self.model.save_weights(model_path)
        processor.dump_embedding(self.model)
        evaluate_emb(config)

class PRSentenceTraining:
    def __init__(self):
        self.post_maxlen = config.post_maxlen
        self.reply_maxlen = config.reply_maxlen
        self.embedding_size =config.embedding_size
        self.pretrained_embedding = config.word_level_embedding
        self.model = self.init_model()

    def init_model(self, init_pretrain_emb=True):

        query_input = ly.Input(shape=(self.post_maxlen,), dtype='int32')
        reply_input = ly.Input(shape=(self.reply_maxlen,), dtype='int32')

        emb_mat_p = None
        emb_mat_r = None
        pEmb = None
        rEmb = None
        if init_pretrain_emb:
            emb_mat_p = processor.pretrain_embedding(self.pretrained_embedding, processor.word2id_p)
            emb_mat_r = processor.pretrain_embedding(self.pretrained_embedding, processor.word2id_r)
            pEmb = ly.Embedding(input_dim=len(processor.word2id_p)+1, output_dim=self.embedding_size,\
                weights=[emb_mat_p], trainable=True, mask_zero=True, name="post_embedding")
            rEmb = ly.Embedding(input_dim=len(processor.word2id_r)+1, output_dim=self.embedding_size,\
                weights=[emb_mat_r], trainable=True, mask_zero=True, name="reply_embedding")
        else:
            pEmb = ly.Embedding(input_dim=len(word2id_p)+1, output_dim=self.embedding_size, trainable=True,\
                mask_zero=True, name="post_embedding")
            rEmb = ly.Embedding(input_dim=len(word2id_r)+1, output_dim=self.embedding_size, trainable=True,\
                mask_zero=True, name="reply_embeding")

        p_emb = pEmb(query_input)
        r_emb = rEmb(reply_input)
        pr_sim = ly.dot(inputs=[p_emb, r_emb], axes=[-1,-1], normalize=True)
        pr_conv = ly.Conv1D(filters=config.filters_num, kernel_size=config.kernel_shape, activation='tanh')(pr_sim)
        pr_pooling = ly.GlobalMaxPooling1D()(pr_conv)
        pr_sim = ly.Dense(1)(pr_pooling)
        
        pr_output = ly.Activation('sigmoid', name='output')(pr_sim)

        model = Model(inputs=[query_input, reply_input], output=pr_output)
        model.compile(loss='binary_crossentropy', optimizer='Adagrad', metrics=['accuracy'])

        return model

def main():

    if not config.do_train and not config.do_eval:
        raise ValueError("You must do_train or do_eval!!!")

    processor.create_dataset_for_training()
    pr_sentence_training = PRSentenceTraining()
    if config.do_train:
        train_query, train_reply, train_label = processor.init_for_train(config.train_file)    
        test_query, test_reply, test_label = processor.init_for_train(config.valid_file)
        logger.info('shape information of training data: train_query: {}\ttrain_reply: {}\ttrain_label:{}'.format(train_query.shape, train_reply.shape, train_label.shape))
        logger.info('shape information of test data: test_query: {}\ttest_reply: {}\ttest_label:{}'.format(test_query.shape, test_reply.shape, test_label.shape))

        logger.info('Building sentence-level training model...')
        logger.info('post_maxlen: %d' % config.post_maxlen)
        logger.info('reply_maxlen: %d' % config.reply_maxlen)
        logger.info('batch_size: %d' % config.batch_size)
        logger.info('epochs: %d' % config.epochs)
        logger.info('Sentence-level training start...')
        model = pr_sentence_training.init_model()
        #logger.info(model.summary())

        saver = SaveModel(model)
        model.fit(x=[train_query, train_reply], y=train_label, batch_size=config.batch_size, epochs=config.epochs, shuffle=True,\
            validation_data=([test_query, test_reply], test_label), callbacks=[saver]) 

    if config.do_eval:
        model_obj = pr_sentence_training.init_model(init_pretrain_emb=False)
        model_obj.load_weights(config.model_weights)
        processor.dump_embedding(model_obj)
        evaluate_emb(config)

if __name__ == '__main__':
    main()
