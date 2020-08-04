#!/usr/bin/env bash

# build_env.sh
# configure build environments

# build tool
pip install git+https://github.com/roywei/d2l-book.git@d2l-java

# build pdf requirements
sudo apt-get install -y texlive-full
sudo apt-get install -y librsvg2-bin
sudo apt-get install -y pandoc

# install fonts
wget https://raw.githubusercontent.com/d2l-ai/utils/master/install_fonts.sh
sudo bash install_fonts.sh

wget -O source-serif-pro.zip https://www.fontsquirrel.com/fonts/download/source-serif-pro
unzip source-serif-pro -d source-serif-pro
sudo mv source-serif-pro /usr/share/fonts/opentype/

wget -O source-sans-pro.zip https://www.fontsquirrel.com/fonts/download/source-sans-pro
unzip source-sans-pro -d source-sans-pro
sudo mv source-sans-pro /usr/share/fonts/opentype/

wget -O source-code-pro.zip https://www.fontsquirrel.com/fonts/download/source-code-pro
unzip source-code-pro -d source-code-pro
sudo mv source-code-pro /usr/share/fonts/opentype/

wget -O Inconsolata.zip https://www.fontsquirrel.com/fonts/download/Inconsolata
unzip Inconsolata -d Inconsolata
sudo mv Inconsolata /usr/share/fonts/opentype/

sudo fc-cache -f -v
