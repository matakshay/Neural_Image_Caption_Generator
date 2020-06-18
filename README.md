<h1> Neural Image Caption Generator </h1>

<p align="justify">
This is a Deep Learning Model for generating captions for images. It uses techniques from Computer Vision and Natural Language Processing.
</p>

<h2 id="intro"> Introduction </h2>
<p align="justify">
Deep Learning and Neural Networks have found profound applications in both NLP and Computer Vision. Before the Deep Learning era, statistical and Machine Learning techniques were commonly used for these tasks, especially in NLP. Neural Networks however have now proven to be powerful techniques, especially for more complex tasks. With the increase in size of available datasets and efficient computational tools, Deep Learning is being throughly researched on and applied in an increasing number of areas.
<br>
In 2012 the Google <a href="http://www.image-net.org/">ImageNet</a> Challenge (ILSVRC) results showed that Convolutional Neural Networks (CNNs) can be an excellent choice for tasks involving visual imagery. Being translation invariant, after learning a pattern in one region of an image, CNNs can very easily recognize it in another region - a task which was quite computationally inefficient in vanilla feed-forward networks. When many convolutional layers are stacked together, they can efficiently learn to recognize patterns in a hierarchical manner - the initial layers learn to detect edges, lines etc. while the later layers make  use of these to learn more complex features. In this project, we make use of a popular CNN architecture - the ResNet50 to process the input images and get the feature vectors.
<br>
For generating the captions, we make use of Long Short-Term Memory (LSTM) networks. LSTMs are a variant of Recurrent Neural Networks which are widely used in Natural Language Processing. Unlike a Dense layer, an RNN layer does not process an input in one go. Instead, it processes a sequence element-by-element, at each step incorporating new data with the information processed so far. This property of an RNN makes it a natural yet powerful architecture for processing sequential inputs.
</p>

<h2 id="model"> Model </h2>

<p align="justify">
    This project uses the ResNet50 architecture for obtaining the image features. ResNets (short for Residual Networks) have been classic approach for many Computer Vision tasks, after this network won the 2015 ImageNet Challenge. ResNets showed how even very Deep Neural Networks (the original ResNet was around 152 layers deep!) can be trained without worrying about the vanishing gradient problem.
The strength of a ResNet lies in the use of Skip Connections - these mitigate the vanishing gradient problem by providing a shorter alternate path for the gradient to flow through. <br>
ResNet50 which is used in this project is a smaller version of the original ResNet152. This architecture is so frequently used for Transfer Learning that it comes preloaded in the Keras framework, along with the weights (trained on the ImageNet dataset). Since we only need this network for getting the image feature vectors, so we remove the last layer (which in the original model was used to classify input image into one of the 1000 classes). The encoded features for training and test images are stored at "encoded_train_features.pkl" and "encoded_test_features.pkl" respectively. 
</p>
	
<div align="center">
	<figure>
		<img src="model_plot.png"
			 alt="Model Plot"
			 width=700>
		<figcaption> A plot of the architecture </figcaption>
	</figure>
</div>

<p align="justify">
<a href="https://nlp.stanford.edu/projects/glove/">GloVe</a> vectors were used for creating the word embeddings for the captions. The version used in this project contains 50-dimensional embedding vectors for 6 Billion English words. It can be downloaded from <a href="https://www.kaggle.com/watts2/glove6b50dtxt"> here</a>. These Embeddings are not processed (fine-tuned using the current data) further during training time. <br>
    The neural network for generating the captions has been built using the Keras Functional API. The features vectors (obtained form the ResNet50 network) are processed and combined with the caption data (which after converting into Embeddings, have been passed through an LSTM layer). This combined information is passed through a Dense layer followed by a Softmax layer (over the vocabulary words). The model was trained for 20 epochs, and at the end of each epoch, the model was saved in the "/model_checkpoints" directory. This process took about half an hour.
</p>
