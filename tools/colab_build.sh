#!/usr/bin/env bash
echo "Update environment..."
apt update -q  &> /dev/null
echo "Install Java..."
apt-get install -q openjdk-11-jdk-headless &> /dev/null
echo "Install Jupyter java kernel..."
curl -L https://github.com/SpencerPark/IJava/releases/download/v1.3.0/ijava-1.3.0.zip -o ijava-kernel.zip &> /dev/null
unzip -q ijava-kernel.zip -d ijava-kernel && cd ijava-kernel && python3 install.py --sys-prefix &> /dev/null
wget -qO- https://gist.github.com/SpencerPark/e2732061ad19c1afa4a33a58cb8f18a9/archive/b6cff2bf09b6832344e576ea1e4731f0fb3df10c.tar.gz | tar xvz --strip-components=1
python install_ipc_proxy_kernel.py --kernel=java --implementation=ipc_proxy_kernel.py
cd ..
git clone https://github.com/deepjavalibrary/d2l-java &> /dev/null
cp -r d2l-java/utils ../
