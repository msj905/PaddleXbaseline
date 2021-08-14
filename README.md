# PaddleXbaseline
# [AI训练营]PaddleX实现目标检测baseline

手把手教你基于PaddleX实现目标检测。你需要实现以下任务：

> 1. 配置数据集（数据集选择、数据处理）
> 2. 配置模型并训练
> 3. 项目跑通即可达到结业要求

# 一、数据集说明

本项目使用的数据集是：[[AI训练营]目标检测数据集合集](https://aistudio.baidu.com/aistudio/datasetdetail/103743)，包含口罩识别 、交通标志识别、火焰检测、锥桶识别以及中秋元素识别。

该数据集已加载至本环境中，位于：**data/data103743/objDataset.zip**


```python
# 解压数据集（解压一次即可，请勿重复解压）
!unzip -oq /home/aistudio/data/data103743/objDataset.zip
```

解压完成后，左侧文件夹处会多一个名为**objDataset**的文件夹，该文件夹下有5个子文件夹：
- **barricade**——Gazebo锥桶检测
- **facemask**——口罩检测
- **fire**——火焰检测
- **MidAutumn**——中秋元素检测
- **roadsign_voc**——交通路标检测

每个子文件夹下有2个文件夹，分别存放着图像（**JPEGImages**）和标注文件（**Annotations**），如下所示：


```python
# 查看数据集文件结构
!tree objDataset -L 2
```

    objDataset
    ├── barricade
    │   ├── Annotations
    │   ├── JPEGImages
    │   ├── labels.txt
    │   ├── test_list.txt
    │   ├── train_list.txt
    │   └── val_list.txt
    ├── facemask
    │   ├── Annotations
    │   └── JPEGImages
    ├── fire
    │   ├── Annotations
    │   └── JPEGImages
    ├── MidAutumn
    │   ├── Annotations
    │   └── JPEGImages
    └── roadsign_voc
        ├── Annotations
        └── JPEGImages
    
    15 directories, 4 files


# 二、数据准备

本基线系统使用的数据格式是PascalVOC格式，开发者基于PaddleX开发目标检测模型时，无需对数据格式进行转换，开箱即用。

但为了进行训练，还需要将数据划分为训练集、验证集和测试集。划分之前首先需要**安装PaddleX**。


```python
# 安装PaddleX
!pip install paddlex
```

    Looking in indexes: https://mirror.baidu.com/pypi/simple/
    Collecting paddlex
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/d6/a2/07435f4aa1e51fe22bdf06c95d03bf1b78b7bc6625adbb51e35dc0804cc7/paddlex-1.3.11-py3-none-any.whl (516kB)
    [K     |████████████████████████████████| 522kB 15.2MB/s eta 0:00:01
    [?25hCollecting paddleslim==1.1.1 (from paddlex)
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/d1/77/e257227bed9a70ff0d35a4a3c4e70ac2d2362c803834c4c52018f7c4b762/paddleslim-1.1.1-py2.py3-none-any.whl (145kB)
    [K     |████████████████████████████████| 153kB 24.6MB/s eta 0:00:01
    [?25hRequirement already satisfied: psutil in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (5.7.2)
    Requirement already satisfied: visualdl>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (2.2.0)
    Requirement already satisfied: opencv-python in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (4.1.1.26)
    Requirement already satisfied: sklearn in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (0.0)
    Collecting pycocotools; platform_system != "Windows" (from paddlex)
      Downloading https://mirror.baidu.com/pypi/packages/de/df/056875d697c45182ed6d2ae21f62015896fdb841906fe48e7268e791c467/pycocotools-2.0.2.tar.gz
    Collecting shapely>=1.7.0 (from paddlex)
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/98/f8/db4d3426a1aba9d5dfcc83ed5a3e2935d2b1deb73d350642931791a61c37/Shapely-1.7.1-cp37-cp37m-manylinux1_x86_64.whl (1.0MB)
    [K     |████████████████████████████████| 1.0MB 16.0MB/s eta 0:00:01     |████████████████▉               | 542kB 16.0MB/s eta 0:00:01
    [?25hRequirement already satisfied: tqdm in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (4.36.1)
    Requirement already satisfied: pyyaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (5.1.2)
    Collecting xlwt (from paddlex)
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/44/48/def306413b25c3d01753603b1a222a011b8621aed27cd7f89cbc27e6b0f4/xlwt-1.3.0-py2.py3-none-any.whl (99kB)
    [K     |████████████████████████████████| 102kB 21.8MB/s ta 0:00:01
    [?25hRequirement already satisfied: colorama in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (0.4.4)
    Requirement already satisfied: flask-cors in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlex) (3.0.8)
    Collecting paddlehub==2.1.0 (from paddlex)
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/7a/29/3bd0ca43c787181e9c22fe44b944b64d7fcb14ce66d3bf4602d9ad2ac76c/paddlehub-2.1.0-py3-none-any.whl (211kB)
    [K     |████████████████████████████████| 215kB 25.5MB/s eta 0:00:01
    [?25hRequirement already satisfied: pyzmq in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddleslim==1.1.1->paddlex) (18.1.1)
    Requirement already satisfied: protobuf>=3.11.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (3.14.0)
    Requirement already satisfied: flask>=1.1.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (1.1.1)
    Requirement already satisfied: flake8>=3.7.9 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (3.8.2)
    Requirement already satisfied: requests in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (2.22.0)
    Requirement already satisfied: Pillow>=7.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (7.1.2)
    Requirement already satisfied: shellcheck-py in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (0.7.1.1)
    Requirement already satisfied: six>=1.14.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (1.15.0)
    Requirement already satisfied: pre-commit in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (1.21.0)
    Requirement already satisfied: bce-python-sdk in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (0.8.53)
    Requirement already satisfied: pandas in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (1.1.5)
    Requirement already satisfied: matplotlib in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (2.2.3)
    Requirement already satisfied: Flask-Babel>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (1.0.0)
    Requirement already satisfied: numpy in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->paddlex) (1.20.3)
    Requirement already satisfied: scikit-learn in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from sklearn->paddlex) (0.24.2)
    Requirement already satisfied: setuptools>=18.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pycocotools; platform_system != "Windows"->paddlex) (56.2.0)
    Requirement already satisfied: cython>=0.27.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pycocotools; platform_system != "Windows"->paddlex) (0.29)
    Requirement already satisfied: gitpython in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (3.1.14)
    Requirement already satisfied: packaging in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (20.9)
    Requirement already satisfied: colorlog in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (4.1.0)
    Requirement already satisfied: easydict in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (1.9)
    Requirement already satisfied: paddlenlp>=2.0.0rc5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (2.0.7)
    Requirement already satisfied: gunicorn>=19.10.0; sys_platform != "win32" in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (20.0.4)
    Collecting paddle2onnx>=0.5.1 (from paddlehub==2.1.0->paddlex)
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/37/80/aa6134b5f36aea45dc1b363e7af941dccabe4d7e167ac391ff046f34baf1/paddle2onnx-0.7-py3-none-any.whl (94kB)
    [K     |████████████████████████████████| 102kB 31.7MB/s ta 0:00:01
    [?25hRequirement already satisfied: filelock in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (3.0.12)
    Requirement already satisfied: rarfile in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlehub==2.1.0->paddlex) (3.1)
    Requirement already satisfied: Jinja2>=2.10.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl>=2.0.0->paddlex) (2.10.1)
    Requirement already satisfied: itsdangerous>=0.24 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl>=2.0.0->paddlex) (1.1.0)
    Requirement already satisfied: click>=5.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl>=2.0.0->paddlex) (7.0)
    Requirement already satisfied: Werkzeug>=0.15 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl>=2.0.0->paddlex) (0.16.0)
    Requirement already satisfied: importlib-metadata; python_version < "3.8" in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0->paddlex) (0.23)
    Requirement already satisfied: pyflakes<2.3.0,>=2.2.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0->paddlex) (2.2.0)
    Requirement already satisfied: mccabe<0.7.0,>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0->paddlex) (0.6.1)
    Requirement already satisfied: pycodestyle<2.7.0,>=2.6.0a1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0->paddlex) (2.6.0)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->paddlex) (1.25.6)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->paddlex) (3.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->paddlex) (2019.9.11)
    Requirement already satisfied: idna<2.9,>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->paddlex) (2.8)
    Requirement already satisfied: virtualenv>=15.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (16.7.9)
    Requirement already satisfied: toml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (0.10.0)
    Requirement already satisfied: aspy.yaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (1.3.0)
    Requirement already satisfied: cfgv>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (2.0.1)
    Requirement already satisfied: identify>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (1.4.10)
    Requirement already satisfied: nodeenv>=0.11.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0->paddlex) (1.3.4)
    Requirement already satisfied: future>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl>=2.0.0->paddlex) (0.18.0)
    Requirement already satisfied: pycryptodome>=3.8.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl>=2.0.0->paddlex) (3.9.9)
    Requirement already satisfied: pytz>=2017.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pandas->visualdl>=2.0.0->paddlex) (2019.3)
    Requirement already satisfied: python-dateutil>=2.7.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pandas->visualdl>=2.0.0->paddlex) (2.8.0)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl>=2.0.0->paddlex) (2.4.2)
    Requirement already satisfied: kiwisolver>=1.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl>=2.0.0->paddlex) (1.1.0)
    Requirement already satisfied: cycler>=0.10 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl>=2.0.0->paddlex) (0.10.0)
    Requirement already satisfied: Babel>=2.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl>=2.0.0->paddlex) (2.8.0)
    Requirement already satisfied: scipy>=0.19.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn->sklearn->paddlex) (1.6.3)
    Requirement already satisfied: joblib>=0.11 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn->sklearn->paddlex) (0.14.1)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn->sklearn->paddlex) (2.1.0)
    Requirement already satisfied: gitdb<5,>=4.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from gitpython->paddlehub==2.1.0->paddlex) (4.0.5)
    Requirement already satisfied: seqeval in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp>=2.0.0rc5->paddlehub==2.1.0->paddlex) (1.2.2)
    Requirement already satisfied: multiprocess in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp>=2.0.0rc5->paddlehub==2.1.0->paddlex) (0.70.11.1)
    Requirement already satisfied: jieba in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp>=2.0.0rc5->paddlehub==2.1.0->paddlex) (0.42.1)
    Requirement already satisfied: h5py in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp>=2.0.0rc5->paddlehub==2.1.0->paddlex) (2.9.0)
    Requirement already satisfied: MarkupSafe>=0.23 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Jinja2>=2.10.1->flask>=1.1.1->visualdl>=2.0.0->paddlex) (1.1.1)
    Requirement already satisfied: zipp>=0.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from importlib-metadata; python_version < "3.8"->flake8>=3.7.9->visualdl>=2.0.0->paddlex) (0.6.0)
    Requirement already satisfied: smmap<4,>=3.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from gitdb<5,>=4.0.1->gitpython->paddlehub==2.1.0->paddlex) (3.0.5)
    Requirement already satisfied: dill>=0.3.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from multiprocess->paddlenlp>=2.0.0rc5->paddlehub==2.1.0->paddlex) (0.3.3)
    Requirement already satisfied: more-itertools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from zipp>=0.5->importlib-metadata; python_version < "3.8"->flake8>=3.7.9->visualdl>=2.0.0->paddlex) (7.2.0)
    Building wheels for collected packages: pycocotools
      Building wheel for pycocotools (setup.py) ... [?25ldone
    [?25h  Created wheel for pycocotools: filename=pycocotools-2.0.2-cp37-cp37m-linux_x86_64.whl size=278363 sha256=e2216374909af8f2e40507a66ad3b2ab78953a6033c0c2405d12a8e0b09480de
      Stored in directory: /home/aistudio/.cache/pip/wheels/fb/44/67/8baa69040569b1edbd7776ec6f82c387663e724908aaa60963
    Successfully built pycocotools
    Installing collected packages: paddleslim, pycocotools, shapely, xlwt, paddle2onnx, paddlehub, paddlex
      Found existing installation: paddlehub 2.0.4
        Uninstalling paddlehub-2.0.4:
          Successfully uninstalled paddlehub-2.0.4
    Successfully installed paddle2onnx-0.7 paddlehub-2.1.0 paddleslim-1.1.1 paddlex-1.3.11 pycocotools-2.0.2 shapely-1.7.1 xlwt-1.3.0


使用如下命令即可将数据划分为70%训练集，20%验证集和10%的测试集。


```python
# 划分数据集
!paddlex --split_dataset --format VOC --dataset_dir objDataset/fire --val_value 0.2 --test_value 0.1
```

    Dataset Split Done.[0m
    [0mTrain samples: 345[0m
    [0mEval samples: 98[0m
    [0mTest samples: 49[0m
    [0mSplit files saved in objDataset/fire[0m
    [0m[0m

划分完成后，该数据集下会生成**labels.txt**, **train_list.txt**, **val_list.txt**和**test_list.txt**，分别存储类别信息，训练样本列表，验证样本列表，测试样本列表。如下图所示：

![](https://ai-studio-static-online.cdn.bcebos.com/d29a92b4cfc34b0097ef46dbbc8562af824387889f224948ae49283e0adee19d)

在这里，**你需要将path to dataset部分替换成你选择的数据集路径**。在左侧文件夹处，将鼠标放到你想选择的数据集文件夹上，会出现三个图标，第一个图标表示复制该文件夹路径，点击即可获得该文件夹路径，用这个路径替换path to dataset即可。

![](https://ai-studio-static-online.cdn.bcebos.com/c28ed88586644f64b34709a592fea0b97ec80470c0e041fd9aa6b8da21c8e283)


# 三、数据预处理

在训练模型之前，对目标检测任务的数据进行操作，从而提升模型效果。可用于数据处理的API有：
- **Normalize**：对图像进行归一化
- **ResizeByShort**：根据图像的短边调整图像大小
- **RandomHorizontalFlip**：以一定的概率对图像进行随机水平翻转
- **RandomDistort**：以一定的概率对图像进行随机像素内容变换

更多关于数据处理的API及使用说明可查看文档：
[https://paddlex.readthedocs.io/zh_CN/release-1.3/apis/transforms/det_transforms.html](https://paddlex.readthedocs.io/zh_CN/release-1.3/apis/transforms/det_transforms.html)


```python
from paddlex.det import transforms

# 定义训练和验证时的transforms
# API说明 https://paddlex.readthedocs.io/zh_CN/develop/apis/transforms/det_transforms.html
train_transforms = transforms.Compose([
    # 此处需要补充图像预处理代码
    transforms.Normalize(),
])

eval_transforms = transforms.Compose([
    # 此处需要补充图像预处理代码
    transforms.Normalize(),
])
```

读取PascalVOC格式的检测数据集，并对样本进行相应的处理。


```python
import paddlex as pdx

# 定义训练和验证所用的数据集
# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/datasets.html#paddlex-datasets-vocdetection
train_dataset = pdx.datasets.VOCDetection(
    data_dir='objDataset/fire',
    file_list='objDataset/fire/train_list.txt',
    label_list='objDataset/fire/labels.txt',
    transforms=train_transforms,
    shuffle=True)

eval_dataset = pdx.datasets.VOCDetection(
    data_dir='objDataset/fire',
    file_list='objDataset/fire/val_list.txt',
    label_list='objDataset/fire/labels.txt',
    transforms=eval_transforms)
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/__init__.py:107: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import MutableMapping
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/rcsetup.py:20: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import Iterable, Mapping
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/colors.py:53: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import Sized


    2021-08-14 10:51:41 [INFO]	Starting to read file list from dataset...
    2021-08-14 10:51:41 [INFO]	345 samples in file objDataset/fire/train_list.txt
    creating index...
    index created!
    2021-08-14 10:51:41 [INFO]	Starting to read file list from dataset...
    2021-08-14 10:51:41 [INFO]	98 samples in file objDataset/fire/val_list.txt
    creating index...
    index created!


需要注意的是：
- **data_dir** (str): 数据集所在的目录路径。
- **file_list** (str): 描述数据集图片文件和对应标注文件的文件路径（文本内每行路径为相对data_dir的相对路径）。
- **label_list** (str): 描述数据集包含的类别信息文件路径。

需要将第二步数据准备时生成的labels.txt, train_list.txt, val_list.txt和test_list.txt配置到以上变量中，如下图所示：

![](https://ai-studio-static-online.cdn.bcebos.com/6462f7811da3436290948e5dde0c497d6ae51bcfe1904e0a863ff032363a4448)


# 四、模型训练

PaddleX目前提供了FasterRCNN和YOLOv3两种检测结构，多种backbone模型。本基线系统以骨干网络为MobileNetV1的YOLOv3算法为例。


```python
# 初始化模型
# API说明: https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#paddlex-det-yolov3

# 此处需要补充目标检测模型代码
model = pdx.det.YOLOv3(num_classes=len(train_dataset.labels), backbone='MobileNetV1')
```


```python
# 模型训练
# API说明: https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#id1
# 各参数介绍与调整说明：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html

# 此处需要补充模型训练参数
model.train(
    num_epochs=270,
    train_dataset=train_dataset,
    train_batch_size=8,
    eval_dataset=eval_dataset,
    learning_rate=0.000125,
    lr_decay_epochs=[210, 240],
    save_dir='output/yolov3_mobilenetv1')
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/framework.py:706: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      elif dtype == np.bool:


    2021-08-14 10:51:55 [INFO]	Decompressing output/yolov3_mobilenetv1/pretrain/MobileNetV1_pretrained.tar...
    2021-08-14 10:51:58 [INFO]	Load pretrain weights from output/yolov3_mobilenetv1/pretrain/MobileNetV1_pretrained.
    2021-08-14 10:51:58 [INFO]	There are 135 varaibles in output/yolov3_mobilenetv1/pretrain/MobileNetV1_pretrained are loaded.
    2021-08-14 10:52:02 [INFO]	[TRAIN] Epoch=1/270, Step=2/43, loss=19382.429688, lr=0.0, time_each_step=2.12s, eta=6:55:56
    2021-08-14 10:52:02 [INFO]	[TRAIN] Epoch=1/270, Step=4/43, loss=14993.798828, lr=0.0, time_each_step=1.12s, eta=3:39:8
    2021-08-14 10:52:03 [INFO]	[TRAIN] Epoch=1/270, Step=6/43, loss=11516.267578, lr=1e-06, time_each_step=0.81s, eta=2:38:49
    2021-08-14 10:52:03 [INFO]	[TRAIN] Epoch=1/270, Step=8/43, loss=8492.828125, lr=1e-06, time_each_step=0.65s, eta=2:8:18
    2021-08-14 10:52:04 [INFO]	[TRAIN] Epoch=1/270, Step=10/43, loss=6741.122559, lr=1e-06, time_each_step=0.55s, eta=1:48:56
    2021-08-14 10:52:04 [INFO]	[TRAIN] Epoch=1/270, Step=12/43, loss=3837.194824, lr=1e-06, time_each_step=0.49s, eta=1:35:35
    2021-08-14 10:52:04 [INFO]	[TRAIN] Epoch=1/270, Step=14/43, loss=1872.744873, lr=2e-06, time_each_step=0.44s, eta=1:25:38
    2021-08-14 10:52:04 [INFO]	[TRAIN] Epoch=1/270, Step=16/43, loss=366.815643, lr=2e-06, time_each_step=0.39s, eta=1:16:34
    2021-08-14 10:52:04 [INFO]	[TRAIN] Epoch=1/270, Step=18/43, loss=529.415161, lr=2e-06, time_each_step=0.36s, eta=1:10:34
    2021-08-14 10:52:05 [INFO]	[TRAIN] Epoch=1/270, Step=20/43, loss=414.085297, lr=2e-06, time_each_step=0.33s, eta=1:5:26
    2021-08-14 10:52:05 [INFO]	[TRAIN] Epoch=1/270, Step=22/43, loss=161.896088, lr=3e-06, time_each_step=0.13s, eta=0:25:27
    2021-08-14 10:52:05 [INFO]	[TRAIN] Epoch=1/270, Step=24/43, loss=103.95491, lr=3e-06, time_each_step=0.13s, eta=0:24:50
    2021-08-14 10:52:05 [INFO]	[TRAIN] Epoch=1/270, Step=26/43, loss=119.845589, lr=3e-06, time_each_step=0.12s, eta=0:22:45
    2021-08-14 10:52:05 [INFO]	[TRAIN] Epoch=1/270, Step=28/43, loss=84.962738, lr=3e-06, time_each_step=0.11s, eta=0:20:59
    2021-08-14 10:52:06 [INFO]	[TRAIN] Epoch=1/270, Step=30/43, loss=84.150681, lr=4e-06, time_each_step=0.1s, eta=0:19:21
    2021-08-14 10:52:06 [INFO]	[TRAIN] Epoch=1/270, Step=32/43, loss=62.310337, lr=4e-06, time_each_step=0.1s, eta=0:18:56
    2021-08-14 10:52:06 [INFO]	[TRAIN] Epoch=1/270, Step=34/43, loss=35.541687, lr=4e-06, time_each_step=0.09s, eta=0:18:2
    2021-08-14 10:52:06 [INFO]	[TRAIN] Epoch=1/270, Step=36/43, loss=29.847147, lr=4e-06, time_each_step=0.09s, eta=0:18:20
    2021-08-14 10:52:06 [INFO]	[TRAIN] Epoch=1/270, Step=38/43, loss=35.76247, lr=5e-06, time_each_step=0.09s, eta=0:18:30
    2021-08-14 10:52:07 [INFO]	[TRAIN] Epoch=1/270, Step=40/43, loss=37.585571, lr=5e-06, time_each_step=0.09s, eta=0:18:21
    2021-08-14 10:52:07 [INFO]	[TRAIN] Epoch=1/270, Step=42/43, loss=30.440241, lr=5e-06, time_each_step=0.1s, eta=0:18:52
    2021-08-14 10:52:07 [INFO]	[TRAIN] Epoch 1 finished, loss=3066.871338, lr=3e-06 .
    2021-08-14 10:52:09 [INFO]	[TRAIN] Epoch=2/270, Step=1/43, loss=31.906452, lr=5e-06, time_each_step=0.21s, eta=0:40:34
    2021-08-14 10:52:10 [INFO]	[TRAIN] Epoch=2/270, Step=3/43, loss=28.70286, lr=6e-06, time_each_step=0.22s, eta=0:40:36
    2021-08-14 10:52:10 [INFO]	[TRAIN] Epoch=2/270, Step=5/43, loss=30.124428, lr=6e-06, time_each_step=0.22s, eta=0:40:37
    2021-08-14 10:52:10 [INFO]	[TRAIN] Epoch=2/270, Step=7/43, loss=34.847061, lr=6e-06, time_each_step=0.23s, eta=0:40:38
    2021-08-14 10:52:11 [INFO]	[TRAIN] Epoch=2/270, Step=9/43, loss=21.176519, lr=6e-06, time_each_step=0.24s, eta=0:40:38
    2021-08-14 10:52:11 [INFO]	[TRAIN] Epoch=2/270, Step=11/43, loss=29.101007, lr=7e-06, time_each_step=0.24s, eta=0:40:40
    2021-08-14 10:52:11 [INFO]	[TRAIN] Epoch=2/270, Step=13/43, loss=31.005165, lr=7e-06, time_each_step=0.25s, eta=0:40:41
    2021-08-14 10:52:11 [INFO]	[TRAIN] Epoch=2/270, Step=15/43, loss=32.990963, lr=7e-06, time_each_step=0.25s, eta=0:40:41
    2021-08-14 10:52:12 [INFO]	[TRAIN] Epoch=2/270, Step=17/43, loss=26.797756, lr=7e-06, time_each_step=0.26s, eta=0:40:41
    2021-08-14 10:52:12 [INFO]	[TRAIN] Epoch=2/270, Step=19/43, loss=26.333765, lr=8e-06, time_each_step=0.26s, eta=0:40:41
    2021-08-14 10:52:12 [INFO]	[TRAIN] Epoch=2/270, Step=21/43, loss=26.65469, lr=8e-06, time_each_step=0.15s, eta=0:40:18
    2021-08-14 10:52:12 [INFO]	[TRAIN] Epoch=2/270, Step=23/43, loss=30.988726, lr=8e-06, time_each_step=0.14s, eta=0:40:17
    2021-08-14 10:52:13 [INFO]	[TRAIN] Epoch=2/270, Step=25/43, loss=30.25742, lr=8e-06, time_each_step=0.14s, eta=0:40:17
    2021-08-14 10:52:13 [INFO]	[TRAIN] Epoch=2/270, Step=27/43, loss=25.908953, lr=9e-06, time_each_step=0.14s, eta=0:40:15
    2021-08-14 10:52:13 [INFO]	[TRAIN] Epoch=2/270, Step=29/43, loss=24.12392, lr=9e-06, time_each_step=0.13s, eta=0:40:13
    2021-08-14 10:52:13 [INFO]	[TRAIN] Epoch=2/270, Step=31/43, loss=29.184597, lr=9e-06, time_each_step=0.12s, eta=0:40:11
    2021-08-14 10:52:13 [INFO]	[TRAIN] Epoch=2/270, Step=33/43, loss=24.439894, lr=9e-06, time_each_step=0.11s, eta=0:40:9
    2021-08-14 10:52:14 [INFO]	[TRAIN] Epoch=2/270, Step=35/43, loss=22.284456, lr=1e-05, time_each_step=0.1s, eta=0:40:8
    2021-08-14 10:52:14 [INFO]	[TRAIN] Epoch=2/270, Step=37/43, loss=17.89353, lr=1e-05, time_each_step=0.1s, eta=0:40:6
    2021-08-14 10:52:14 [INFO]	[TRAIN] Epoch=2/270, Step=39/43, loss=17.677759, lr=1e-05, time_each_step=0.1s, eta=0:40:6
    2021-08-14 10:52:14 [INFO]	[TRAIN] Epoch=2/270, Step=41/43, loss=20.959705, lr=1e-05, time_each_step=0.1s, eta=0:40:5
    2021-08-14 10:52:14 [INFO]	[TRAIN] Epoch=2/270, Step=43/43, loss=17.791466, lr=1.1e-05, time_each_step=0.09s, eta=0:40:4
    2021-08-14 10:52:14 [INFO]	[TRAIN] Epoch 2 finished, loss=28.028652, lr=8e-06 .


# 五、总结与升华

遇到的问题：
        1.PaddleX目标MobileNetV1的YOLOv3算法第一次使用熟练程度不够，出现的第一个问题是训练集和验证集分布不均匀导致我重新按步骤执行
		2.训练模型时前期未使用gpu时训练时长一轮需要大约3分钟，使用后速度提升了5秒一轮。
收获：让我熟悉目标检测模型为我自定义项目农业病虫害那块模型选择上有些借鉴，并提升了使用PaddleX的个人能力

# 个人简介

本人迈书杰，目前就职于顺鑫集团-福通互联集团科技有限公司，目前参与的项目中有关于车辆和人脸识别方面的项目，农业方面也有病虫害方面的需求，打算学习下为对自己管理后续项目有所帮助，希望和大家共同交流。


