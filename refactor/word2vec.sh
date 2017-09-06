#!/bin/bash

DATA_DIR=word2vec/data
BIN_DIR=word2vec/bin
SRC_DIR=word2vec/src

TEXT_DATA=files/$1
VECTOR_DATA=files/$2
: ${VOCAB_MIN_COUNT:=5}
: ${WINDOW_SIZE:=15}
: ${VECTOR_SIZE:=50}

pushd ${SRC_DIR} && make; popd

$BIN_DIR/word2vec -train $TEXT_DATA -output $VECTOR_DATA -size $VECTOR_SIZE -window $WINDOW_SIZE -threads 12 -min-count $VOCAB_MIN_COUNT
