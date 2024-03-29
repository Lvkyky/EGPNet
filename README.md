# Code for JSTARS article 'Edge-Guided Parallel Network for VHR Remote Sensing Image Change Detection'.
---------------------------------------------
Here I provide PyTorch implementations for EGPNet

![EPGNet](https://github.com/Lvkyky/EGPNet/assets/87217038/e5d8aa0f-cc65-4826-bf96-702471343656)

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
You can download our pretrained models for LEVIR-CD, SYSU-CD, CDD  [Baidu Netdisk, code: 6666](https://pan.baidu.com/s/1EXME33U3bSIOEvkz84kCyg), [Baidu Netdisk, code: 6666](https://pan.baidu.com/s/1GUZViG6n6StY9HmuuPx8gg), [Baidu Netdisk, code: 6666](https://pan.baidu.com/s/18lme7QOl66TriADFcTp71g).

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
The dataset folder must be named LEVIR,SYSU,CDD, or there will be issues with dataset loading.

## Citation
>@article{zhu2023edge,<br>
>  title={Edge-Guided Parallel Network for VHR Remote Sensing Image Change Detection},<br>
> author={Zhu, Ye and Lv, Kaikai and Yu, Yang and Xu, Wenjia},<br>
>  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},<br>
>  year={2023},<br>
>  publisher={IEEE}<br>
}


## Contact
Don't hesitate to contact me if you have any question.

Email: 563167677@qq.com.cn

