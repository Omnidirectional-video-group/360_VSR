# ICIP 2024: 360° Video Super-Resolution and Quality Enhancement Challenge

<!-- <img title="" src="/imgs/mark.png" alt="" data-align="center"> -->

<div align="center">
  <img src="imgs/logo.png" height="128">
</div>

360° Video Super-Resolution and Quality Enhancement Challenge is held as a part of the **[ICIP2024 conference](https://2024.ieeeicip.org/)** sponsored by **[TII](https://www.tii.ae/)**.

<div align="center">

📕[__Datasets__](https://tiiuae-my.sharepoint.com/:f:/g/personal/ahmed_telili_tii_ae/EogDz0BrzYNLqyj5LpniiOQB6yq-jtpxJFLbTjudB4rGkQ)  __|__ 📝[Evaluation Script](script/quality_assessment.py) __|__ 🧑‍🤝‍🧑[WhatsApp group](https://chat.whatsapp.com/GPy6gBmVbNcC7epkp0t2lW) __|__ [Official website](https://www.icip24-video360sr.ae/)

⏬[Submission example](https://tiiuae-my.sharepoint.com/personal/ahmed_telili_tii_ae/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fahmed%5Ftelili%5Ftii%5Fae%2FDocuments%2F360VistaSR%2Fsubmission%5Ftest%2Ezip&parent=%2Fpersonal%2Fahmed%5Ftelili%5Ftii%5Fae%2FDocuments%2F360VistaSR)

</div>

---

👉 [Track 1：ICIP 2024 Challenge (2x super resolution and quality enhancement)](#track-icip-2024-challenge) **|** [Codalab server](https://codalab.lisn.upsaclay.fr/competitions/17459)

👉 [Track 2：Innovation showcase (4x super resolution and quality enhancement)](#track-2-innovation-showcase) **|** [Codalab server](https://codalab.lisn.upsaclay.fr/competitions/17458)

## 🚩 Track #1: ICIP 2024 challenge timeline 

- ✅ 2024.02.05 Release of train data (input and output videos) and validation data (inputs only)
- ✅ 2024.02.05 Validation server online
- ✅ 2024.02.21 Test phase beginning
- ✅ 2024.02.28 Docker file/code submission deadline
- ✅ 2024.03.06 Final test results release and winner announcement
- ✅ 2024.03.27 Challenge paper submission deadline
- ✅ 2024.10.27 Workshop days, results and award ceremony ([ICIP 2024](https://2024.ieeeicip.org/), Abu DHabi, UAE)

## 🚩 Track #2: Innovation showcase timeline  

- ✅ 2024.02.05 Release of train data (input and output videos) and validation data (inputs only)
- ✅ 2024.02.05 Validation server online
- ✅ 2024.04.15 Validation test release (output videos)
- ✅ 2024.04.15 Test phase beginning (output videos)
- ✅ 2024.04.28 Docker file/code submission deadline
- ✅ 2024.05.05 Final test results release and winner announcement
- ✅ 2024.05.30 Award distribution 

## Introduction

Omnidirectional visual content, commonly referred to as 360-degree images and videos, has garnered significant interest in both academia and industry, establishing itself as the primary media modality for VR/XR applications. 360-degree videos offer numerous features and advantages, allowing users to view scenes in all directions, providing an immersive quality of experience with up to 3 degrees of freedom (3DoF). When integrated on embedded devices with remote control, 360-degree videos offer additional degrees of freedom, enabling movement within the space (6DoF). However, 360-degree videos come with specific requirements, such as high-resolution content with up to 16K video resolution to ensure a high-quality representation of the scene. Moreover, limited bandwidth in wireless communication, especially under mobility conditions, imposes strict constraints on the available throughput to prevent packet loss and maintain low end-to-end latency. Adaptive resolution and efficient compression of 360-degree video content can address these challenges by adapting to the available throughput while maintaining high video quality at the decoder. Nevertheless, the downscaling and coding of the original content before transmission introduces visible distortions and loss of information details that cannot be recovered at the decoder side. In this context, machine learning techniques have demonstrated outstanding performance in alleviating coding artifacts and recovering lost details, particularly for 2D video. Compared to 2D video, 360-degree video presents a lower angular resolution issue, requiring augmentation of both the resolution and the quality of the video. This challenge presents an opportunity for the scientific research and industrial community to propose solutions for quality enhancement and super-resolution for 360-degree videos.

In this challenge, we aim to establish high-quality benchmarks for 360° video SR, and expect to further highlight the challenges and research problems. This challenge presents an opportunity for the scientific research and industrial community to propose solutions for quality enhancement and super-resolution for 360-degree videos.

## Challenge Description

__During the first phase (Development period)__ Participants will have access to public training and a light version of the public testing (validation set) of the 360 VISTA-SR dataset. This streamlined validation set will include a single folder containing LR (Low Resolution) 360° videos encoded at various target bitrates, rather than four distinct bitrate categories. This approach is aimed at reducing data transfer sizes for submissions. Participants can train their models and observe their scores for the validation set. Live score values will be uploaded on the CodaLab platform, with the team's score on the leaderboard regularly updated. For details, refer to the  [Submission](#submission) section.

__During the second phase (Testing period)__, the full validation dataset will be released, allowing participants to further refine their models with an extensive range of data. At the end of this phase,  participants should adhere to the "Docker File Submission Guideline (TBD)" and submit their docker files by  April 15, 2024 23:59 (AOE🌎), to the grand challenge email address: ahmed.telili@tii.ae and brahim.farhat@tii.ae .

### Dataset - 360 VISTA-SR ([Download](https://tiiuae-my.sharepoint.com/:f:/g/personal/ahmed_telili_tii_ae/EogDz0BrzYNLqyj5LpniiOQB6yq-jtpxJFLbTjudB4rGkQ))

We provide a dataset containing 200 360-degree videos, predominantly sourced from YouTube and ODV360 (Link) characterized by high quality and resolution (4K and 2K) in ERP format. All videos are licensed under Creative Commons Attribution (reuse allowed), and our dataset is exclusively designed for academic and research purposes. The video dataset encompasses various content characteristics, including outdoor and indoor scenes, as well as high motion sport contents. Each video consists of 100 frames. The dataset is partitioned into 160 videos for training, 20 for validation, and 20 for testing. Note that additional external content can be incorporated for training.

|         | Training                      | Validation               | Testing                    |
| ------- | ---------------------         | ------------------------ | -------------------------- |
| Source  | Youtube+ODV360                | Youtube+ODV360           | Youtube+ODV360             |
| Number  | 100                           | 20                       | 20                         |
| Storage | 76.7G (HR) + 103.68G (LR)     | 10.6G (HR) + 14.3G (LR)  | 11.5G (HR) + 14.7G (LR)    |

**Note that we do not provide degradation scripts during this challenge to avoid potential testing data leakage.**

### Metrics

We evaluate the super-resolved 360° videos by comparing them to the ground truth HR ERP videos. To measure the fidelity, we adopt the widely used Weighted-to-Spherically-uniform Peak Signal to Noise Ratio (WS-PSNR) as the quantitative evaluation metric. Moreover, we incorporate runtime complexity into our final scoring formula. Therefore, models that optimally balance between quality and processing efficiency are highlight more. For more details, please refer to the detailed descriptions available on the official website [here](https://www.icip24-video360sr.ae/).

## Tracks

#### Track 1：ICIP 2024 challenge 
##### 360° Omnidirectional Video Super-Resolution and Qaulity Enhancement (X2)

This truck for contributors aiming to submit a challenge paper to the ICIP 24. It targets x2 upscaling of the downgraded source videos. Achieving good quality in x2 super resolution is relatively more feasible.  This truck lies in finding the optimal trade-off between complexity and quality.  The complexity score is given a higher weight in the final score.

#### Track 2：Innovation showcase
##### 360° Omnidirectional Video Super-Resolution and Qaulity Enhancement (X4)

This track is a longer-duration compared to Track 1, giving participants more time to work on their ideas. It focuses on achieving x4 upscaling for the input video, which poses a significant challenge in maintaining high quality. Models that achieve superior quality, even at a slower pace, are emphasized in this track. The quality score holds a higher weight in the final score calculation. 

### Baseline example results

| Model         | SwinIR / WS-PSNR (dB) | SwinIR / Runtime (s/2k) | SwinIR / Score | FSRCNN / WS-PSNR (dB) | FSRCNN / Runtime (s/2k) | FSRCNN / Score |
|---------------|-----------------------|-------------------------|----------------|-----------------------|-------------------------|----------------|
| Track #1 (x4) | 29.141                | 0.4458                  | 29.79          | 28.346                | 0.0013                  | 61.10          |
| Track #2 (x2) | 30.014                | 1.5232                  | 13.87          | 29.546                | 0.0041                  | 76.21          |

The table illustrates the WS-PSNR performance and run time of three baseline models on the 360VISTA validation set. Across both x2 and x4 tracks, FSRCNN emerges as the top-performing model based on our scoring criteria. Despite SwinIR exhibiting superior quality, FSRCNN's faster run time provides it with a competitive advantage. Therefore, the optimal model is one that effectively balances quality and complexity.

**Note on computational specifications: The results presented herein were obtained using a desktop computer equipped with an Intel® Xeon 8280 CPU @ 2.70GHz × 56, 128GB RAM, and a NVIDIA RTX 6000 Ada graphics card with 48GB of VRAM.**

## Submission

We use CodaLab for online submission in the development phase. Here, we provide an example [link](https://tiiuae-my.sharepoint.com/personal/ahmed_telili_tii_ae/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fahmed%5Ftelili%5Ftii%5Fae%2FDocuments%2F360VistaSR%2Fsubmission%5Ftest%2Ezip&parent=%2Fpersonal%2Fahmed%5Ftelili%5Ftii%5Fae%2FDocuments%2F360VistaSR) to help participants to format their submissions. In the test phase, participants should adhere to the "Docker File Submission Guideline (TBD)" and submit their docker files by  April 15, 2024 23:59 (AOE🌎), to the grand challenge email address: ahmed.telili@tii.ae and brahim.farhat@tii.ae

## Training and Validation

**We provide a comprehensive framework designed to facilitate both training and testing processes for participants. However, participants are completely free to use their own code in place of our provided resources.**

### Requirement

Use pip to install all requirements:

```bash {"id":"01HNYG6Q4N4K1ZXJSGB07AT17X"}
pip install -r requirements.txt
```

### Configuration

Before training and testing, please make sure the fields in [config.yaml](src/config.yaml) is properly set.

```yaml {"id":"01HNYG6Q4N4K1ZXJSGB0JJP5D5"}
log_dir: "output/FSRCNN"  # Directory for logs and outputs

dataset:
  train:
    hr_root: "data/train/HR"
    lr_root: "data/train/LR_X4"
    lr_compression_levels: ["1", "2", "3", "4"]  # list for Compression levels directories
    crop_size: 64 # The height and width of cropped patch for training.
    transform: True # if True data augmentation is used
    batch_size: 4 
    shuffle: True
    num_workers: 8 # number of cores used for data loader
  val:
    hr_root: "data/val/HR" 
    lr_root: "data/val/LR_X4"
    lr_compression_levels: ["1", "2", "3", "4"]
    batch_size: 4
    shuffle: False
    num_workers: 1
  test:
    hr_root: ''
    lr_root: "data/test/LR_X4"
    lr_compression_levels: ["1"]
    batch_size: 2
    shuffle: False    
    num_workers: 1                       

model:
  path: "src/model/FSRCNN.py"   # Path to the model definition file
  name: "FSRCNN" # Model class name to be instantiated
  scale_factor: 4 # adjust the scale factor

learner:
  general:
    total_steps: 3000000 # The number of training steps.
    log_train_info_steps: 100 # The frequency of logging training info.
    keep_ckpt_steps: 20000 # The frequency of saving checkpoint.
    valid_steps: 5000 # The frequency of validation.
    
  optimizer: # Define the module name and setting of optimizer
    name: "Adam"              
    lr: 0.0001                 
    beta_1: 0.9
    beta_2: 0.999
    
  lr_scheduler: # Define the module name and setting of learning rate scheduler
    name: "ExponentialDecay"
    initial_learning_rate: 0.0001
    decay_steps: 10000
    decay_rate: 0.1
    staircase: True
    
  saver: # The path to checkpoint where would be restored from.
    restore: #checkpoints/step_308000_checkpoint_x4.pth.tar
  loss:
    name: "CharbonnierLoss"   # Type of loss function to use
    params: {}                # Additional parameters for the loss function, if needed


```

### Train

To train the model, use the following command:

```bash {"id":"01HNYG6Q4N4K1ZXJSGB27PT4Q2"}
python main.py --process train --config_path config.yml


```

### Test

To generate testing outputs, use the following command:

```bash {"id":"01HNYG6Q4PB76K6D6XD2HRZ43P"}
python main.py --process test --config_path config.yml
```

## FAQ

1.  We do not restrict competitors from using additional training data. If it is used, it is necessary to indicate the source and amount.

2.  We do not restrict competitors from using pretrained networks. If it is used, it is necessary to provide details.

### Organizers

Ahmed Telili @TII

Ibrahim Farhat @TII

Wassim Hamidouche @TII

Hadi Amirpour @AAU

#### Acknowledgement

We use the GitHub README.md template from [Stereo SR competition](https://github.com/The-Learning-And-Vision-Atelier-LAVA/Stereo-Image-SR/tree/NTIRE2022)

We inspired the Framework template from [mai22-real-time-video-sr](https://github.com/MediaTek-NeuroPilot/mai22-real-time-video-sr/tree/main)

## 🧑‍🤝‍🧑 WhatsApp group

<div align="center">
  <img src="imgs/WhatsApp.jpeg">
</div>
