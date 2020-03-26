# Neural_Image_Caption_Generator

This project aims to build an Image Caption Generator using Deep Learning and Natural Language Processing. The idea for this project has been roughly taken from - Show and Tell: A Neural Image Caption Generator by Oriol Vinyals, Alexander Toshev, Samy Bengio and Dumitru Erhan. The paper can be found at https://arxiv.org/abs/1411.4555v2.

The dataset used this project was Flickr8K from Kaggle (https://www.kaggle.com/shadabhussain/flickr8k). It consists of 8000 images (divided into train, dev and test sets) with around 5 captions for each image. This dataset can be found in the directory Neural_Image_Caption_Generator/data.

Pre-Trained GLOVE vectors have been taken from Kaggle (https://www.kaggle.com/watts2/glove6b50dtxt). It contains 50 dimensional vectors for 6 Billion words in English language

For converting an image to feature vector, a ResNet50 Neural architecture (https://arxiv.org/abs/1512.03385) has been used, from Keras. It was trained on the ImageNet (http://www.image-net.org/) dataset. 

