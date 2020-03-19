[<img src='https://s3.amazonaws.com/drivendata-public-assets/logo-white-blue.png' width='600'>](https://www.drivendata.org/)
<br><br>

<div align="center">
<img src='http://lila.science/wp-content/uploads/2018/10/ss_web.jpg' alt='Competition Image' width='500'>
</div>

# Hakuna Ma-data: Identify Wildlife on the Serengeti with AI for Earth
## Goal of the Competition

Camera traps are an invaluable tool in conservation research, but the sheer amount of data they generate presents a huge barrier to using them effectively. This is where AI can help!

In the Hakuna Ma-Data Challenge, participants built models to tag species from a new trove of camera trap imagery provided by the Snapshot Serengeti project. But that's not all! This was a new kind of DrivenData challenge, where competitors packaged everything needed to do inference and submitted that for containerized execution on Azure. By leveraging Microsoft Azure's cloud computing platform and Docker containers, the competition infrastructure moved one step closer to translating participants’ innovation into impact.

## What's in this Repository
This repository contains code provided by leading competitors in the [Hakuna Ma-data: Identify Wildlife on the Serengeti with AI for Earth](https://www.drivendata.org/competitions/59/camera-trap-serengeti/) DrivenData challenge.

#### Winning code for other DrivenData competitions is available in the [competition-winners repository](https://github.com/drivendataorg/competition-winners).

## Winning Submissions

Place |Team or User | Public Score | Private Score | Summary of Model
--- | --- | --- | --- | ---
1 | Team ValAn_picekl | 0.002343 | 0.005107 | We used lightweight convolutional neural network architectures with one exception. Key models were [EfficientNet B1 and EfficientNet B3](https://arxiv.org/abs/1905.11946), and [SE-ResNext50](https://arxiv.org/abs/1709.01507). For all mentioned architectures we used ImageNet pretrained checkpoints. While training we used cyclical learning rates / one cycle policy, warm start and cosine annealing, a consistent input size of 512x384, horizontal flip data augmentation, and no validation set. We also used both random sampling and chunk sampling (where the training set was divided in chunks comprised of one or more seasons). For inference, we used a weighted ensemble with six forward passes in total. To get to the sequence level, we took the mean for animals and geometric mean for empty.
2 | n01z3 | 0.003240 | 0.005159 | Models were `swsl_resnext50` and `wsl_resnext101d8`. The first convolution and the first batch normalization are frozen during all stages of training. Used WarmUp, CosineDecay, initLR 0.005, SGD, WD 0.0001, 8 GPUs, Batch 256 per GPU. Loss / metric was `torch.nn.MultiLabelSoftMarginLoss`. Progressive increase in size during training. Width resize: 256 -> 320 -> 480 for resnext50; 296 -> 360 for resnext101. During training, resize to ResizeCrop size for the width -> RandomCrop with ResizeCrop / 1.14 size. The crop is not square, but rectangular with the proportion of the original image. During inference, resize to ResizeCrop and that’s it. From augmentations: flip, contrast, brightness, using default parameters from albumentations. Test-time augmentation (TTA) was flip. Used geometric mean to average within one series. Also used geometric mean for TTA prediciton and model averaging.
3 | bragin | 0.002190 | 0.005357 | Found the model that gave me the best score, InceptionResNetV2, on <https://keras.io/applications/>. After training several models and creating an ensemble, looked at the losses of each class and found that empty was the biggest. Extracted the background from sequences of images and trained a binary classifier (empty/ non-empty). Trained lgbm classifier as a second level model using predictions of each image + prediction of background classifier. Built merged DNN based on InceptionResNetV2 which takes background and mean of images simultaneously. Updated lgbm classifier using all model that I trained before. DNNs trained on seasons 1-8 (9,10 for validation), lgbm trained jn season 9 (10 for validation).

Additional solution details can be found in the `reports` folder inside the directory for each submission.

#### [Interview with winners](http://drivendata.co/blog/wildlife-serengeti-winners.html)

#### Benchmark Blog Post: ["How to Use Deep Learning to Identify Wildlife"](http://drivendata.co/blog/ai-for-earth-wildlife-detection-benchmark/)
