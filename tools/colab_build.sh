#!/usr/bin/env bash
echo "Update environment..."
apt update -q  > /dev/null
echo "Install Java..."
apt-get install -q openjdk-11-jdk-headless > /dev/null
echo "Install Jupyter java kernel..."
curl -L https://github.com/SpencerPark/IJava/releases/download/v1.3.0/ijava-1.3.0.zip -o ijava-kernel.zip > /dev/null
unzip -q ijava-kernel.zip -d ijava-kernel && cd ijava-kernel && python3 install.py --sys-prefix > /dev/null
