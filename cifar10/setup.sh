mkdir data
mkdir checkpoints

pip3 install -r requirements.txt

cd data
gdown --id 1oYnD7Izl3LVVzjEMyLxLklX30TKWHgGG
unzip ./cifar-10.zip
rm -rf ./cifar-10.zip
mv ./cifar-10/sample_submission.csv ./cifar-10/test_labels.csv
cd ..
