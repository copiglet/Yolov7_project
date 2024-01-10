#!/usr/bin/env bash

# ex) ./2.batch_run.sh 2023 03 15 m 01
# params :  2023 03 15 m 01

y=$1
m=$2
d=$3
pm=$4
day=$5

if [ -z "$day" ]; then
	echo "params fail"
	echo "params : yyyy mm dd pm hour"
	echo "example : 2023 03 15 m 01"
	exit -1
fi

d1=/home/huvio/Project/huvio/data/ag_data/$y/$m/$d/$pm

python detect_tool.py -dir ${d1}/${day} -btrtype ${pm} -ymd ${y}-${m}-${d}


