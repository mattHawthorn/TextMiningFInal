#!/bin/bash

#####################

MALLETDIR="/home/nikhil/mallet/"
MALLET="bin/mallet"
INPUTDIR="/home/nikhil/Documents/TextMiningProject/TextMiningFinal/data/mallet/"
OUTPUTDIR="/home/nikhil/Documents/TextMiningProject/TextMiningFinal/data/topics/"

MALLETFILES=("20_newsgroups_complex.mallet" "20_newsgroups_simple.mallet")
TOPICS=(5 10 20 40 80)

OPTINT=10

####################

for FILE in "${MALLETFILES[@]}"
do
	for TOPIC in "${TOPICS[@]}"
	do
		echo "$MALLETDIR$MALLET train-topics --input $INPUTDIR$FILE --num-topics $TOPIC --output-topic-keys "$OUTPUTDIR$FILE.$TOPIC.topics_KEYS.txt" --output-doc-topics \"$OUTPUTDIR$FILE.$TOPIC.topics_COMPOSITION.txt\" --optimize-interval $OPTINT"		
		$MALLETDIR$MALLET train-topics --input $INPUTDIR$FILE --num-topics $TOPIC --output-topic-keys "$OUTPUTDIR$FILE.$TOPIC.topics_KEYS.txt" --output-doc-topics "$OUTPUTDIR$FILE.$TOPIC.topics_COMPOSITION.txt" --optimize-interval $OPTINT
	done
done



 

