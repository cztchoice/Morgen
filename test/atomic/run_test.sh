#!/bin/sh



for j in $(seq 1 12)
do
    ./test  --elements 100000000 --stride $j

done