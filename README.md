# 在岗人员状态分析

## Contents
1. [Requirements](#requirements)
2. [Pretrained](#pretrained)
3. [Usage](#usage)
3. [Example](#example)

## Requirements
* Python3
* [Pytorch 1.8](https://pytorch.org/get-started/locally/) with cuda enabled
* opencv-python numpy 
```
conda create -n work_env python=3.8
conda activate work_env
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
conda install opencv-python numpy fvcore
```
## Pretrained
Download the pretrained model from [BaiduDisk](https://pan.baidu.com/s/1oqEDl-Glx1afbBl0HKEQ4w)[password:witv]


## Usage

* 16-clip inference(lite,faster)
```
python work.py --video-path input_video.mp4 \
--save-path inference/output_video.mp4 \
--test-crop-size 448 \
--clip-length 16 \
--pretrained pretrained/work_16f_s1_best_ap_01790.pth
```
* 32-clip inference(more accurate)
```
python work.py --video-path input_video.mp4 \
--save-path inference/output_video.mp4 \
--test-crop-size 448 \
--clip-length 32 \
--pretrained pretrained/work_32f_s1_best_ap_01905.pth
```

## Example

Human action anaylse examples visualizations!

<br/>
<br/>
<div align="center" style="width:image width px;">
  <img  src="examples/ava3.gif" width=240 alt="ava_example_1">
  <img  src="examples/ava1.gif" width=240 alt="ava_example_2">
  <img  src="examples/ava4.gif" width=240 alt="ava_example_3">
</div>
<br/>
<br/>