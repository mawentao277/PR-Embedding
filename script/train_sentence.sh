module load cuda/9.0
export CUDA_VISIBLE_DEVICES=2,3
source ~/.bashrc
python sentence-training/sentence_training.py  \
                  --vocab_file ./data/vocab.align \
                  --train_file ./data/train.txt \
                  --valid_file ./data/dev.txt \
                  --test_file ./data/valid_none_original.txt \
                  --do_train \
                  --epochs 10 \
                  --output_model_path ./output/model \
                  --model_weights model/model_best_align.hdf5 \
                  --pr_embedding_file ./output/pr_embedding_convai.txt \
                  --word_level_embedding ./output/word_learn_embedding.txt
