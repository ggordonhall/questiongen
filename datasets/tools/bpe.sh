#!/bin/bash
# Pipeline for byte pair encoding of data
# Author : Thamme Gowda
# Created : Nov 06, 2017

BASE="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

# update these variables
DATASET=$1
DIR1=$2
DIR2=$3

INP="$DATASET/$2"
OUT="$BASE/$DATASET/$3"
DATA="$BASE/$INP"

echo "Paths..."
echo $DATA
echo $OUT

TRAIN_SRC=$DATA/*train.src
TRAIN_TGT=$DATA/*train.tgt
VALID_SRC=$DATA/*dev.src
VALID_TGT=$DATA/*dev.tgt
TEST_SRC=$DATA/*test.src
TEST_TGT=$DATA/*test.tgt

BPE="src+tgt" # src, tgt, src+tgt

# applicable only when BPE="src" or "src+tgt"
BPE_SRC_OPS=10000

# applicable only when BPE="tgt" or "src+tgt"
BPE_TGT_OPS=10000


#====== EXPERIMENT BEGIN ======

# Check if input exists
for f in $TRAIN_SRC $TRAIN_TGT $VALID_SRC $VALID_TGT $TEST_SRC $TEST_TGT; do
    if [[ ! -f "$f" ]]; then
        echo "Input File $f doesnt exist. Please fix the paths"
        exit 1
    fi
done

function lines_check {
    l1=`wc -l < $1`
    l2=`wc -l < $2`
    if [[ "$l1" != "$l2" ]]; then
        echo "ERROR: Record counts doesnt match between: $1 and $2"
        exit 2
    fi
}
lines_check $TRAIN_SRC $TRAIN_TGT
lines_check $VALID_SRC $VALID_TGT
lines_check $TEST_SRC $TEST_TGT


echo "Output dir = $OUT"
[ -d $OUT ] || mkdir -p $OUT


echo "Step 1a: Preprocess inputs"
if [[ "$BPE" == *"src"* ]]; then
    echo "BPE on source"
    # Here we could use more  monolingual data
    $BASE/tools/learn_bpe.py -s $BPE_SRC_OPS -i $TRAIN_SRC -o $OUT/bpe-codes.src

    $BASE/tools/apply_bpe.py -c $OUT/bpe-codes.src -i $TRAIN_SRC -o $OUT/train.src
    $BASE/tools/apply_bpe.py -c $OUT/bpe-codes.src -i $VALID_SRC -o $OUT/dev.src
    $BASE/tools/apply_bpe.py -c $OUT/bpe-codes.src <  $TEST_SRC > $OUT/test.src
else
    ln -sf $TRAIN_SRC $OUT/train.src
    ln -sf $VALID_SRC $OUT/dev.src
    ln -sf $TEST_SRC $OUT/test.src
fi


if [[ "$BPE" == *"tgt"* ]]; then
    echo "BPE on target"
    # Here we could use more  monolingual data
    # $BASE/tools/learn_bpe.py -s $BPE_SRC_OPS -i $TRAIN_TGT -o $OUT/bpe-codes.tgt

    $BASE/tools/apply_bpe.py -c $OUT/bpe-codes.src -i $TRAIN_TGT -o $OUT/train.tgt
    $BASE/tools/apply_bpe.py -c $OUT/bpe-codes.src -i $VALID_TGT -o $OUT/dev.tgt
    $BASE/tools/apply_bpe.py -c $OUT/bpe-codes.src <  $TEST_TGT > $OUT/test.tgt
    # We dont touch the test References, No BPE on them!
    ln -sf $TEST_TGT $OUT/test.tgt
else
    ln -sf $TRAIN_TGT $OUT/train.tgt
    ln -sf $VALID_TGT $OUT/dev.tgt
    ln -sf $TEST_TGT $OUT/test.tgt
fi

