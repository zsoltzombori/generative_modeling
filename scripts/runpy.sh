#!/bin/bash

eps="3 6 9 12 15 18 21 24"
prefix='vae_epoch_'
for i in $eps;
do
	epoch=$i
	echo "$epoch"
	#./cparam.sh "$prefix$epoch.ini" nb_epoch $epoch
	#./cparam.sh "$prefix$epoch.ini" outdir "pictures/$prefix$epoch"
	# ROOT
	cd ..
	#mkdir -p "pictures/$prefix$epoch"
	#python generative.py "ini/auto-generated/$prefix$epoch.ini" > "pictures/$prefix$epoch/main.log"
	mkdir -p "pictures/$prefix$epoch/graphs"
	python lookup.py "ini/auto-generated/$prefix$epoch.ini" > "pictures/$prefix$epoch/graphs/main.log"
	# SCRIPTS
	cd scripts
done
