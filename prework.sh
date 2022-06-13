git submodule update --init --recursive

pip install -r requirements.txt

# wget https://pjreddie.com/media/files/yolov3.weights
wget https://pjreddie.com/media/files/yolov3-tiny.weights

cd data/custom
python write_img_paths.py
python voc2yolo.py
cd ../..
