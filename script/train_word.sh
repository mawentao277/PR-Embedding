#!/bin/bash

CORPUS=./data/convai_sample.align
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
