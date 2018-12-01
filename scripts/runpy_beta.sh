#!/bin/bash

#bts="1.0 1.2 1.4 1.6 1.8 2.0"
#bts="0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0"
bts="3.0 4.0 6.0 8.0 10 20 30 40 50 70 100"
prefix='vae_beta_'
for i in $bts;
do
	epoch="size_loss|$i,variance_loss|$i"
	echo "$epoch"
	rm -f "../ini/auto-encoder/$prefix$epoch.ini"
	#./cparam.sh "$prefix$i.ini" weights $epoch
	#./cparam.sh "$prefix$i.ini" outdir "pictures/$prefix$i"
	# ROOT
	cd ..
	#mkdir -p "pictures/$prefix$i"
	#CUDA_VISIBLE_DEVICES=2 python generative.py "ini/auto-generated/$prefix$i.ini" > "pictures/$prefix$i/main.log"
	mkdir -p "pictures/$prefix$i/graphs"
	CUDA_VISIBLE_DEVICES=0 python lookup.py "ini/auto-generated/$prefix$i.ini" > "pictures/$prefix$i/graphs/main.log"
	# SCRIPTS
	cd scripts
done
