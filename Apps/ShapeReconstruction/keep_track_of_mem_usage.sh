#!/bin/bash
nPid=$1;
nTimes=10; # customize it
delay=0.1; # customize it
strCalc=`top -d $delay -b -n $nTimes -p $nPid \
  |grep $nPid \
  |sed -r -e "s;\s\s*; ;g" -e "s;^ *;;" \
  |cut -d' ' -f9 \
  |tr '\n' '+' \
  |sed -r -e "s;(.*)[+]$;\1;" -e "s/.*/scale=2;(&)\/$nTimes/"`;
nPercCpu=`echo "$strCalc" | bc -l`
echo $nPercCpu