# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Puts device in a special state giving optimal benchmark results.
# Not very realistic wrt real-world conditions. We are more interested
# in performance on devices in default state. This script is left here
# in case it might be occasionally useful, but hopefully shouldn't
# distract us from benchmarking in real-world conditions.

#!/bin/bash

echo "disabling mpdecision..."
adb root
adb remount
adb shell "mv /system/bin/mpdecision /system/bin/mpdecision-dontfind" > /dev/null

echo "rebooting device..."
adb reboot
adb wait-for-device

echo "restarting adbd as root..."
isroot=0
while [ $isroot -eq 0 ]
do
  adb root > /dev/null
  if [ $? -eq 0 ]
  then
    isroot=1
  fi
  echo "  retrying in 1 s..."
  sleep 1
done

echo "querying ro.hardware..."
hardware="`adb shell getprop ro.hardware`"
while [ "$hardware" == "" ]
do
  echo "retrying in 1 s..."
  sleep 1
  hardware="`adb shell getprop ro.hardware`"
done

echo "got ro.hardware=$hardware"

shouldstopui=0
if [[ "$#" =~ .*--stop-ui.* ]]
then
  shouldstopui=1
else
  if [[ "$hardware" =~ sprout.* ]]
  then
    echo "detected Android One (sprout). Will default to leaving the UI on."
  else
    echo "Will default to stopping the UI."
    shouldstopui=1
  fi
fi

if [ $shouldstopui -ne 0 ]
then
  echo "stopping the UI..."
  isshellstopped=0
  while [ $isshellstopped -eq 0 ]
  do
    adb shell stop > /dev/null
    if [ $? -eq 0 ]
    then
      isshellstopped=1
    fi
    echo "  retrying in 1 s..."
    sleep 1
  done
fi

waitsec=10
echo "sleeping $waitsec s before changing CPU settings, to work around a race with Android startup..."
sleep $waitsec

echo "bringing all cores online..."
for cpu in `seq 0 3`
do
  file="/sys/devices/system/cpu/cpu$cpu/online"
  if [ -e $file ]
  then
    echo "  cpu $cpu"
    echo "echo 1 > $file; exit" | adb shell > /dev/null
    if [ $? -ne 0 ]
    then
      echo "WARNING: failed to bring cpu $cpu online ($file)"
    fi
  fi
done

echo "setting performance governor..."
for cpu in `seq 0 3`
do
  file="/sys/devices/system/cpu/cpu$cpu/cpufreq/scaling_governor"
  if [ -e $file ]
  then
    echo "  cpu $cpu"
    adb shell "echo performance > $file" > /dev/null
    if [ $? -ne 0 ]
    then
      echo "WARNING: failed to set cpufreq governor ($file)"
    fi
  fi
done

cpuloadlowsecs=0
echo "waiting for CPU load to settle down..."
while [ $cpuloadlowsecs -lt 5 ]
do
  cpuload="`adb shell top -n 1 -d 1 -s cpu | awk '{sum += $3} END {print sum}'`"
  if [ "$cpuload" -lt "2" ]
  then
    cpuloadlowsecs=$((cpuloadlowsecs+1))
    echo "CPU load has been low for $cpuloadlowsecs s..."
  else
    cpuloadlowsecs=0
    echo "CPU load isn't low enough ($cpuload %)..."
  fi
  sleep 1
done

if [ $shouldstopui -eq 0 ]
then
  echo "OK, the device might be ready now, but the UI is still running,"
  echo "so take a look at the screen to check if it's not doing something special."
fi
