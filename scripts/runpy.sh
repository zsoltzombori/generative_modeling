#!/bin/bash

prefix='vae_'
for i in `seq 4 4`;
do
	epoch=$((i*50))
	echo "$epoch"
	./cparam.sh "$prefix$epoch.ini" nb_epoch $epoch
	./cparam.sh "$prefix$epoch.ini" outdir "pictures/$prefix$epoch"
	# ROOT
	cd ..
	mkdir -p "pictures/$prefix$epoch"
	python generative.py "ini/auto-generated/$prefix$epoch.ini" > "pictures/$prefix$epoch/main.log"
	mkdir -p "pictures/$prefix$epoch/graphs"
	python lookup.py "ini/auto-generated/$prefix$epoch.ini" > "pictures/$prefix$epoch/graphs/main.log"
	# SCRIPTS
	cd scripts
done
