#!/bin/bash

lrs="0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.3"
#lrs="0.1 0.3"
prefix='vae_lr_'
for i in $lrs;
do
	epoch=$i
	echo "$epoch"
	#./cparam.sh "$prefix$epoch.ini" lr $epoch
	#./cparam.sh "$prefix$epoch.ini" outdir "pictures/$prefix$epoch"
	# ROOT
	cd ..
	#mkdir -p "pictures/$prefix$epoch"
	#CUDA_VISIBLE_DEVICES=1 python generative.py "ini/auto-generated/$prefix$epoch.ini" > "pictures/$prefix$epoch/main.log"
	mkdir -p "pictures/$prefix$epoch/graphs"
	CUDA_VISIBLE_DEVICES=0 python lookup.py "ini/auto-generated/$prefix$epoch.ini" > "pictures/$prefix$epoch/graphs/main.log"
	# SCRIPTS
	cd scripts
done
