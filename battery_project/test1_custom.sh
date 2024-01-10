# d1=/home/huvio/Project/huvio/data/ag_data/2023/03/15/m
d1=/home/huvio/Project/huvio/data/ag_data/2023/08/22/m

date > test1.log
python detect_custom.py -dir $d1/00 -btrtype m -ymd 2023-08-22 >> test1.log
date >> test1.log


