cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sudo apt install -y linux-tools-common linux-tools-$(uname -r)
sudo cpupower frequency-set -g performance
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
