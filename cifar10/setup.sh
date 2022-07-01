mkdir data
mkdir checkpoints

pip3 install -r requirements.txt

cd data
gdown --id 1wdyVIQLAUyzaJhYiJS6oRkc39MApGOtL
wget https://github.com/Xerefic/Deep_Networks/releases/download/dataset/cifar-10.zip
unzip ./cifar-10.zip
rm -rf ./cifar-10.zip
cd ..
