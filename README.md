# This project is the code of my paper : Edge-Guided Parallel Network for VHR Remote Sensing Image Change Detection.
I am not sure if this paper will be accepted， but happiness must come first.


---------------------------------------------
Here I provide PyTorch implementations for EPGNet.


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

Then put them in `/hisNet0`, `/hisNet1`, separately.


* Test on the LEVIR-CD dataset
```python Test_LEVIR_0.py```

* Test on the SYSU-CD dataset
```python Test_YSYU_1.py```


## Train & Validation
```python Train_LEVIR_0.py ```


## Contact
Don't hesitate to contact me if you have any question.

Email: 563167677@qq.com.cn

