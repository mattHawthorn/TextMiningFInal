#!/bin/bash
MALLETDIR="/home/matt/Git/Mallet/"
MALLET="bin/mallet"
INPUTDIR="../data/mallet"

cd $INPUTDIR
ls | egrep .+.txt > localtextfiles.list
pwd

cat localtextfiles.list | while read INFILE
do
  echo "Converting $INFILE to MALLET"
  OUTFILE=${INFILE%.*}.mallet
  $MALLETDIR$MALLET import-file --input $INFILE --output $OUTFILE --token-regex '[^\s]+'
done

rm localtextfiles.list
