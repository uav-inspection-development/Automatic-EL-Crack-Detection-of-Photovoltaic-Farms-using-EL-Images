# Automatic EL Crack Detection of Photovoltaic Farms using EL Images

## Steps to Run This Project

1. Download the dataset from the link below:
    - [EL Crack Dataset](https://pan.baidu.com/s/11_Qj8LsRqgpXz4PLqeiE0w?pwd=d1dj)
2. Copy it here and follow the third step tutorial "Train the Model" to get the weight file best.pt (it will be automatically generated in the runs folder within the code folder after running train.py).
3. Then follow the third step tutorial to run ui.py in the src folder, load the best.pt obtained in the previous step in the interface that appears, and then you can perform image, video, and real-time camera detection.

## Environment Deployment Steps

```shell
conda create -n pytorch python=3.10
conda activate pytorch
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
conda env list
```

```plaintext
Downloading https://ultralytics.com/assets/Arial.ttf to 'C:\Users\ad\AppData\Roaming\Ultralytics\Arial.ttf'...
```

If this prompt appears, it means the configuration file is being downloaded automatically. Occasionally, a timeout may occur. It is recommended to copy the "Arial.ttf" file from the fonts folder to the path shown on the right and rerun the code (note that the path shown on the right is the path displayed on your computer; the example here is from my computer, and the path may vary on different computers).
