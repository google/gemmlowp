# Copyright 2014 Google Inc. All Rights Reserved.
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

# Un-does the effect of prepare-device-for-benchmarking.sh

#!/bin/bash

echo "restoring mpdecision..."
adb root
adb remount
adb shell "mv /system/bin/mpdecision-dontfind /system/bin/mpdecision" > /dev/null

echo "rebooting device..."
adb reboot
adb wait-for-device

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
