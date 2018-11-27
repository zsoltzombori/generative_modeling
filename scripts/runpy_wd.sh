#!/bin/bash

lrs="0.0005 0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.01"
prefix='vae_wd_'
for i in $lrs;
do
	epoch=$i
	echo "$epoch"
	./cparam.sh "$prefix$epoch.ini" generator_wd $epoch
	./cparam.sh "$prefix$epoch.ini" encoder_wd $epoch
	./cparam.sh "$prefix$epoch.ini" outdir "pictures/$prefix$epoch"
	# ROOT
	cd ..
	mkdir -p "pictures/$prefix$epoch"
	CUDA_VISIBLE_DEVICES=0 python generative.py "ini/auto-generated/$prefix$epoch.ini" > "pictures/$prefix$epoch/main.log"
	#mkdir -p "pictures/$prefix$epoch/graphs"
	#CUDA_VISIBLE_DEVICES=0 python lookup.py "ini/auto-generated/$prefix$epoch.ini" > "pictures/$prefix$epoch/graphs/main.log"
	# SCRIPTS
	cd scripts
done
