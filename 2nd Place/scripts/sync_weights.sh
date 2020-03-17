#!/usr/bin/env bash


while true
do
    rsync -zavhP --include="*/" --include="*.pth" --exclude="*" dl2:/mnt/hdd1/learning_dumps/wild/* /mnt/hdd1/learning_dumps/wild/
    rsync -zavhP --include="*/" --include="*.ckpt" --exclude="*" dl2:/mnt/hdd1/learning_dumps/wild/* /mnt/hdd1/learning_dumps/wild/
#    rsync -zavhP ../../thunder-hammer orange:/raid/data_share/code/

    sleep 60
done