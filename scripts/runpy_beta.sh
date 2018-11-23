#!/bin/bash

prefix='vae_beta_'
for i in `seq 1 5`;
do
	epoch="size_loss|$i|$i|0|1,variance_loss|$i|$i|0|1"
	echo "$epoch"
	#./cparam.sh "$prefix$epoch.ini" weight_schedules $epoch
	#./cparam.sh "$prefix$epoch.ini" outdir "pictures/$prefix$epoch"
	# ROOT
	cd ..
	#mkdir -p "pictures/$prefix$epoch"
	#CUDA_VISIBLE_DEVICES=0 python generative.py "ini/auto-generated/$prefix$epoch.ini" > "pictures/$prefix$epoch/main.log"
	#mkdir -p "pictures/$prefix$epoch/graphs"
	#CUDA_VISIBLE_DEVICES=0 python lookup.py "ini/auto-generated/$prefix$epoch.ini" > "pictures/$prefix$epoch/graphs/main.log"
	# SCRIPTS
	cd scripts
done
