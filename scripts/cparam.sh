#!/bin/sh
# arguments: path_to_file, parameter,_name, new_value
cd ..
if [ ! -f "ini/auto-generated/$1" ]; then
	cp ini/auto-generated/beta_vae_tuned.ini ini/auto-generated/$1
fi
N=`grep -nr $2 ini/auto-generated/$1 | cut -d: -f1`
sed "${N} c\\$2\t$3" ini/auto-generated/$1 > ini/auto-generated/tmp.ini
mv ini/auto-generated/tmp.ini ini/auto-generated/$1
