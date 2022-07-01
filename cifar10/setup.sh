mkdir data
mkdir checkpoints

pip3 install -r requirements.txt

cd data
gdown --id 1_uPa6fE2wnrFi82vbk4gGaoK3h0fOcUl
wget https://github.com/Xerefic/Deep_Networks/releases/download/dataset/cifar-10.zip
unzip ./cifar-10.zip
rm -rf ./cifar-10.zip
cd ..
