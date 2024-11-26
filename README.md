## Street Image Segmentation - Unet

Semantic segmentation is one of the most important fields of computer vision, whose task is to segment images into meaningful parts and assign each part to a pre-defined category. As scenarios understanding is a one of the core issues in computer vision, semantic segmentation helps computers understand and interpret the meaning of scenarios in images, which is also beneficial to more applications in segmentation fields [[1]](https://arxiv.org/abs/1704.06857). Therefore, the development of semantic segmentation plays an important role in CV. At present, this technique is widely used in the medical field and autonomous vehicle driving field, and mainly divided into supervised semantic segmentation, unsupervised semantic segmentation, video segmentation, etc.

Street-view segmentation is a branch of application scenarios in the field of segmentation, which have been widely used in various intelligent transportation systems, such as autonomous driving, vehicle safety, and other fields [[2]](https://arxiv.org/abs/2003.08736). It can deeply understand the different objects in the street-view image, such as pedestrians, streets, cars, etc. Pohlen et. al [[3]](https://arxiv.org/abs/1611.08323) proposed that due to the need of precise boundary and understanding of street scene semantics in these fields, the accuracy of street scene segmentation is an important metric. Generally speaking, the study of semantic segmentation in street-view images could promote the development in multiple fields.

This project mainly focuses on exploring the street-view images through semantic segmentation, which includes fine-tuning on a specific task and unsupervised domain adaptation. The objectives of this project are:

1. Evaluate the pre-trained model on bright sunny test images.
2. Fine-tune the pre-trained model to adapt to the darker cloudy dataset and evaluate the fine-tuned model performance.
3. Suppose the darker cloudy dataset has no labels, implement unsupervised domain adaptation through pseudo-ground truth generation.
4. Apply 2 unsupervised domain adaptation methods, which are using confidence to filter pixels with high uncertainty and try adaptive batch normalization (AdaBN).

### Model Introduction

Unet [[4]](https://arxiv.org/abs/1505.04597) was proposed in 2015 and is an important model in semantic segmentation. It is a classic model in semantic segmentation, and we use it as a pre-trained model in this project. Its is similar to a U-shaped structure, which is divided into two parts: an encoder and a decoder. The encoder uses convolution layers and pooling layers to reduce the size and dimension of feature maps, responsible for extracting high- level features from the image. The decoder uses upsampling and convolutional layers to gradually restore the size and dimension of feature maps, and ultimately outputs segmentation results of the same size as the original image. Meanwhile, Unet adopts a skip connection method to link the decoder and encoder, which helps obtain feature information at different levels and improve the accuracy of final segmentation. This project uses Unet as the backbone model and trains semantic segmentation models for 14 categories with sunny street-view image dataset. In the initial code files,

**Unet.py** contains the backbone network of the model.

**Dataset.py** is used to load images and labels in the dataset.

**Data_aug.py** is used for image augmentations, including resizing, cropping, and other operations. Code_example_train_unet_sunny.ipynb is to train and predict the model with sunny dataset. 

**Camvid_sunny_model.pt** saves pre-trained UNet model parameters.

The initial Unet model performed well on sunny dataset of street-view, with global image accuracy reached 85.1%. Moreover, this model has high accuracy in several specific categories, such as Road, Sky, and Building. 

#### **Note**: The implementation is within this branch.

### Fine tuning

However, the pre-trained Unet model did not perform well on cloudy datasets, with a global image accuracy of around 62.6%, which is 20% lower than the accuracy under sunny conditions. When we observe the IOU of each category, we can find that the accuracy of IOU has almost all decreased for each category, with Building, Road, and Lanemaking showing the most significant decrease (all exceeding 30%), while only Tree's IOU remains almost stable.

Fine-tuning is an important technique in machine learning. It uses a large amount of data to generate a pre-trained model, which, therefore, already has the ability to extract certain features. In order to enable the model to acquire new knowledge and adapt to specific tasks, we will use fine-tuning techniques to optimize and improve the pre-trained model. In this task, we will fine-tune the pre-trained model trained by a sunny dataset to adapt to cloudy conditions. In fine-tuning, the following steps are usually taken as follows: Firstly, choose the pre-trained model. In this project, we choose Unet as the pre-trained model on sunny dataset. Since the semantic segmentation categories for both sunny and cloudy conditions are 14, we do not need to modify the FC layer. Furthermore, due to the high similarity between the data on cloudy and sunny dataset, we can adopt a relatively simple and ideal fine-tuning approach, which is to use pre-trained weights to initialize the entire Unet model and then retrain the entire model without frozen layers in Unet. We will then record the loss of both train and validation datasets after each epoch, and select the model with the lowest loss under validation as the best model.

### Unsupervised Domain Adaptation

Unsupervised domain adaptation is also an important direction in machine learning. Traditional machine learning requires a large amount of manually annotated labels, but due to the high cost of manual annotation, it is difficult to obtain a significant amount of annotated data. For samples without labels, the unsupervised domain adaptation method could be adapted, which means transferring the model from the source domain to the target domain (without labels) [[5]](https://arxiv.org/abs/1409.7495). In our project, the challenge is to assume that the street-view images of cloudy weather are unlabeled, and we need to transfer the pre-trained model to adapt to these unlabeled cloudy datasets.

#### Pseudo labels

Pseudo labels for these unlabeled datasets would be generated in this process. The specific approach to obtaining pseudo labels is as follows: using a pre-trained Unet model to predict the labels for cloudy datasets, and then feed the predicted labels and cloudy data into the model for training. However, due to the uncertainty of the pseudo labels trained by the pre-trained model, we need to filter out pixels with high uncertainty. Entropy is a useful tool in this condition, which can represent the uncertainty of a pixel, and higher entropy indicates a higher uncertainty for the specific pixel. After obtaining a highly credible label and its corresponding pixels, we can use pseudo labels to fine tune a new model.

#### AdaBN

The basic assumption of fine-tuning is that the training data and test data follow the same data distribution, but this is almost non-existent in real situations. Therefore, although fine-tuning can help us save training time and resources, we need to pay attention to the different distributions of data that it cannot handle. Adaptive batch normalization is another adaptive model that improves the ability to transfer features by modifying the batch normalization layer. AdaBN dynamically calculates the statistical measures of the BN layer, such as mean, variance, etc. from samples in the target domain. Therefore, in order to ensure that the parameters of other layers remain unchanged in our experiment, we froze the parameters of other layers through require_grad = False. Then we only use the input image data each time to modify the Batch Normalization layer to update its statistics.

**Note**: All experiments was conducted on a personal computer, and my computer is configured with the maxOS system. The code uses vscode (version 1.87.2) ide, and all models all run on 8-core CPU.
