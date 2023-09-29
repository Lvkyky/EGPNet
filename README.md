# Code for JSTARS article 'Edge-Guided Parallel Network for VHR Remote Sensing Image Change Detection'.
---------------------------------------------
Here I provide PyTorch implementations for EGPNet(FuseNet)


## Requirements
>RTX 3060 <br>
>python 3.7 <br>
>PyTorch 1.13.1


## Installation
Clone this repo:
```shell
git clone https://github.com/Lvkyky/EGPNet.git
cd EPGNet
```

## Test
You can download our pretrained models for LEVIR-CD, SYSU-CD, CDD  [Baidu Netdisk, code: jstar](https://pan.baidu.com/s/1DTazE7I3lhELPRZr5oyniQ), [Baidu Netdisk, code: jstar](https://pan.baidu.com/s/1CDkcUUpdd0w9tz4fe7no0A), [Baidu Netdisk, code: jstar](https://pan.baidu.com/s/1DTazE7I3lhELPRZr5oyniQ).

Then put them in `/EGPNET_LEVIR/bestNet`, `/EGPNET_SYSU/bestNet`,  `/EGPNET_CDD/bestNet` separately.


* Test on the LEVIR-CD dataset
```python Test_LEVIR.py```

* Test on the SYSU-CD dataset
```python Test_YSYU.py```

* Test on the SYSU-CD dataset
```python Test_CDD.py```


## Train & Validation
```python Train_LEVIR.py ```
```python Train_SYSU.py ```
```python Train_CDD.py ```

## Attention
The dataset folder must be named LEVIR,SYSU,CDD, or there will be issues with dataset loading errors.


## Contact
Don't hesitate to contact me if you have any question.

Email: 563167677@qq.com.cn

