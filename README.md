# Abstract
![](https://github.com/Wanelle/MFP-CNN-for-Scene-Classification/blob/main/asserts/images/challenge.png?raw=true)
Scene classification is a challenging task in computer vision, involving the assignment of semantic labels to images based on their visual content. This paper proposes a Multi-scale Fusion and Pooling Convolutional Neural Network (MFP-CNN) to address the complexities of scene classification, including intraclass diversity, large-scale data processing, and heterogeneous object sizes. To handle the intra-class diversity, we introduce a Lightweight Multi-Stage Feature Fusion (LMSFF) method, effectively capturing multi-scale information and improving feature discriminability. Spatial Pyramid Pooling (SPP) is incorporated to enhance scene representation by considering features at different spatial scales. Moreover, the Squeeze-and-Excitation (SE) attention mechanism is employed to focus on informative regions and enhance feature extraction. The attention mechanism allows the model to effectively capture discriminative features and improve classification accuracy. We present the Scene7 dataset, containing diverse real-world scenes with comprehensive annotations for scene classification and object detection tasks. Extensive experiments on the Scene7 dataset validate the effectiveness of our MFP-CNN model, demonstrating superior accuracy and parameter efficiency compared to state-of-the-art methods. The proposed MFP-CNN offers a robust and efficient solution for scene classification, contributing to the advancement of computer vision applications. For more details, please refer to [our paper](https://github.com/).

# Scene7
![](https://github.com/Wanelle/MFP-CNN-for-Scene-Classification/blob/main/asserts/images/scene7.png?raw=true)
We present Scene7, a medium-sized dataset specifically curated for scene classification and object detection tasks. This dataset encompasses seven distinct indoor and outdoor life scenes: train interior, car interior, bedroom interior, living room interior, kitchen interior, botanical garden interior, and street interior. Scene7 is designed to cater to diverse research requirements, including the analysis of fast-moving scene analysis.

To compile the Scene7 dataset, we gather images from  various sources, including the internet, real-world scenes, and  the publicly available SUN397 scene dataset. Our data collection process involves extracting relevant scene images from  the SUN397 dataset and supplementing them with additional images acquired through web crawlers. We then meticulously filter out inappropriate and duplicate images, resulting in a final dataset of 10,747 images. The distribution of images across different scenes is as follows: 1,435 train interior images, 1,132 car interior images, 1,507 bedroom images, 1,521 living room images, 2,003 kitchen images, 1,669 botanical garden images, and 1,480 street images.

 Below is the statistics of our Scene7 dataset:
![](https://github.com/Wanelle/MFP-CNN-for-Scene-Classification/blob/main/asserts/images/image%20number.png?raw=true)
![](https://github.com/Wanelle/MFP-CNN-for-Scene-Classification/blob/main/asserts/images/instances%20of%20image.png?raw=true)


Since GitHub does not support uploading very large files, we uploaded the Scene7 dataset and part of the Scene7 dataset that can be used for object detection on Kagglea. The links are:。。。
# MFP-CNN
The MFP-CNN leverages advanced techniques such as feature fusion, spatial pyramid pooling, and attention mechanisms to enhance the accuracy and robustness of scene classification. Specifically, we propose a Lightweight Multi-Stage Feature Fusion (LMSFF) method to handle the intra-class diversity by effectively fusing features from different scales and capturing multi-scale information. This enables the model to capture both global context and fine-grained details, improving the discriminative power of the features. To address the challenge of object size variations, we incorporate Spatial Pyramid Pooling (SPP) into the network architecture. SPP allows the model to consider features at multiple spatial scales, enabling better representation of scenes with objects of different sizes. Furthermore, we introduce the Squeeze-and-Excitation (SE) mechanism to enhance feature extraction by focusing on informative regions of the scene. The SE mechanism enables the model to allocate attention to important image regions and capture discriminative features, leading to improved classification accuracy.
![](https://github.com/Wanelle/MFP-CNN-for-Scene-Classification/blob/main/asserts/images/model.png?raw=true)

if you want to run this model. First, you need to replace the scene dataset at the data loading location. Secondly, by modifying the final fully connected layer based on the dataset you added, you can run it directly and obtain the results.
In addition, you can explore the accuracy of the model by modifying the fold count, epoch count, learning rate, and dropout rate of multi fold cross validation.

# Contact
@gmail.com

