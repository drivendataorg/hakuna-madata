#!/usr/bin/env bash


while true
do
    rsync -zavhP --include="*/" --include="*.yml" --exclude="*" ../ orange:/raid/data_share/code/drivendata-identify-wildlife
#    rsync -zavhP --include="*/" --include="*.csv" --exclude="*" ../ orange:/raid/data_share/code/drivendata-identify-wildlife
    rsync -zavhP --include="*/" --include="*.py" --exclude="*" ../ orange:/raid/data_share/code/drivendata-identify-wildlife

    rsync -zavhP ../../thunder-hammer orange:/raid/data_share/code/

    sleep 60
done