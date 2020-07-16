#!/bin/bash
input=$1  #input corpus, two columns: post"\t"reply
output=$2 #output PR-Embedding

if [ $# != 2 ] ; then
  echo "Please set \$1: input corpos file; \$2: output PR-Embedding path"
  exit 1;
fi

if [ ! -d ./output/tmp ]; then
  mkdir -p ./output/tmp
fi

ORI_CORPUS=${input}
PR_CORPUS=${input}.pr
CORPUS=${input}.align
VOCAB_FILE=./data/vocab.align
COOCCURRENCE_FILE=./output/tmp/cooccurrence.bin
COOCCURRENCE_SHUF_FILE=./output/tmp/cooccurrence.shuf.bin
BUILDDIR=build
SAVE_FILE=./output/word_learn_embedding
VERBOSE=2
MEMORY=50.0
VOCAB_MIN_COUNT=2
VECTOR_SIZE=500
MAX_ITER=15
WINDOW_SIZE=5
BINARY=0
NUM_THREADS=8
X_MAX=10
WORD_ALIGN=0 #0: randomly generate the cross-sentence window; 1: use the word alignmet model

#add pr tag into the corpus
echo "Begin to add pr tag..."
python word-alignment/preprocess.py \
            --original_input_file ${ORI_CORPUS} \
            --pr_input_file ${PR_CORPUS}\
            --add_pr_tag 
echo "Finsh adding pr tag."

#add word alignment results
if test $WORD_ALIGN = "1";then
  echo "Begin to calculate word alignment..."
  sh script/word_alignment.sh ${PR_CORPUS} ${CORPUS}
  echo "Finsh to calculate word alignment..."
else
  CORPUS=${PR_CORPUS}
  echo "Ramdomly generate the cross-sentence window!"
fi

#word-level training
echo "Begin to train word-level embedding..."
make
$BUILDDIR/vocab_premb -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
if [[ $? -eq 0 ]]
  then
  $BUILDDIR/cooccur_premb -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE
  if [[ $? -eq 0 ]]
  then
    $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE
    if [[ $? -eq 0 ]]
    then
       $BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE
    fi
  fi
fi
echo "Finish training word-level embedding..."

if [ ! -d ./output/model ]; then
  mkdir -p ./output/model
fi

#sentence-level training
module load cuda/9.0
source ~/.bashrc
export CUDA_VISIBLE_DEVICES=2,3 
echo "Begin to train sentence-level embedding..."
python sentence-training/sentence_training.py  \
                  --vocab_file ./data/vocab.align \
                  --train_file ./data/train.txt \
                  --valid_file ./data/dev.txt \
                  --test_file ./data/valid_none_original.txt \
                  --do_train \
                  --epochs 10 \
                  --output_model_path ./output/model \
                  --model_weights ./output/model/model_epoch10.hdf5 \
                  --pr_embedding_file ${output} \
                  --word_level_embedding ./output/word_learn_embedding.txt
echo "Finsh training PR-Embedding: ${output}"
