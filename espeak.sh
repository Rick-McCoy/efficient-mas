# !/bin/bash

set -e

sudo apt update
sudo apt install make autoconf automake libtool pkg-config gcc libsonic-dev ronn kramdown libpcaudio-dev -y

git clone https://github.com/espeak-ng/espeak-ng.git
cp ./util/text/ja_rules_fixed ./espeak-ng/dictsource/ja_rules
cd espeak-ng

./autogen.sh
./configure --prefix=/usr

make
make docs

sudo make LIBDIR=/usr/lib/x86_64-linux-gnu install

cd ..
rm -rf espeak-ng
