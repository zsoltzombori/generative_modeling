#!/bin/sh
# arguments: path_to_file, parameter,_name, new_value
cd ..
N=`grep -nr $2 ini/vae_conv.ini | cut -d: -f1`
sed "${N} c\\$2\t$3" $1 > tmp.ini
mv tmp.ini $1
#python lookup.py ini/vae_conv.ini
