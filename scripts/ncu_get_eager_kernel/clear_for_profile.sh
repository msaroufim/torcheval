# Stop all GPU monitoring services that block ncu
sudo systemctl stop nvidia-dcgm.service dynologd.service

# Verify they're stopped
sudo systemctl list-units --state=active | grep -E "(nvidia|dynolog)"

# Check GPU is clear
sudo lsof /dev/nvidia7 | grep -v python
