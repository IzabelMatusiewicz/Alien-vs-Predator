# Alien-vs-Predator
In this Kaggle dataset, instead of the typical cats and dogs, we are presented with images of alien and predator creatures. These images have been organized in a Keras folder structure and are suitable for transfer learning tasks.
![alien_predator](https://github.com/IzabelMatusiewicz/Alien-vs-Predator/assets/101067795/c5458a39-a12e-492c-ae1c-75ff96e2a9bd)

## About the project
Based on a small dataset consisting of 247 alien images and 247 predator images for training, as well as 100 alien images and 100 predator images for validation, I have conducted experiments using three different image augmentation techniques to improve the training base. The effectiveness of these techniques was compared by applying the prepared images, augmented in three different ways, to the same convolutional neural networks.
As a second approach to address the limitations of the small dataset, transfer learning was employed using the ResNet50 network, while incorporating the best augmentation technique identified in the previous step.
To evaluate the model's performance, the results can be tested using the Gradio interface.

## Methodology
### Reading images
During the loading process, the images were resized to 64x64 pixels because they had irregular sizes.
![download (27)](https://github.com/IzabelMatusiewicz/Alien-vs-Predator/assets/101067795/95d14408-9e94-4a0a-922e-5cb60b69463d)

### Augumentation techniques
Image augmentation refers to the technique of applying random transformations to the existing images in a dataset, creating new variations of the original images. It helps to increase the diversity of the dataset and improve the model's ability to generalize. Common augmentations include rotation, shifting, flipping, zooming, and changing brightness/contrast. By introducing these variations, the model becomes more robust to different perspectives, orientations, and lighting conditions, leading to better performance and reduced overfitting. Augmentation is particularly beneficial when the dataset is small, as it effectively increases the effective size of the dataset and helps prevent overfitting.

#### Base augumentation techniques
Our base set of augmentation techniques includes rescaling, rotation, width shift, height shift, and horizontal flip. 
Rescaling ensures the pixel values are normalized between 0 and 1. Rotation randomly rotates the images within a range of -20 to +20 degrees. Width shift introduces horizontal translations by randomly shifting the image width by a fraction of 0.2. Height shift introduces vertical translations by randomly shifting the image height by a fraction of 0.2. Horizontal flip randomly flips the images horizontally, adding variation by reversing them along the horizontal axis. 

![download (29)](https://github.com/IzabelMatusiewicz/Alien-vs-Predator/assets/101067795/6cc04fd9-9e95-4159-9ecb-85c99adaccd6)

In addition to these techniques, we also applied grayscale conversion (re-scaling to a single channel) and introduced noise to further enhance the diversity and robustness of our augmented dataset (next steps).

#### Grey-scale augumentation
Converting images to grayscale eliminates color information and keeps only brightness values. This simplifies data representation and reduces computational complexity by processing a single channel instead of three. In the provided code snippet, the color_mode="grayscale" argument ensures that both the training and testing datasets are loaded as grayscale images.

![download (28)](https://github.com/IzabelMatusiewicz/Alien-vs-Predator/assets/101067795/dac937b0-09b8-430b-b216-444db8e57c96)

#### Noise augumentation
Noise augmentation is a technique where random noise is added to the images in the dataset. This introduces variations in pixel values and helps the model become more robust to noise present in real-world scenarios. By applying noise augmentation, the model can learn to better distinguish signal from noise and improve its performance in noisy environments.
![download (30)](https://github.com/IzabelMatusiewicz/Alien-vs-Predator/assets/101067795/36209ea3-c409-4f9e-8676-cf1e4d09db28)

#### Augumentation summary
The best results were achieved with the baseline data augmentation, and it was utilized for further modeling.

### Transfer learning
Transfer learning is a technique in deep learning where a pre-trained model, such as ResNet50, is used as a starting point for a new task. In this code snippet, the ResNet50 model with pre-trained weights, trained on the ImageNet dataset, is loaded. The pre-trained layers are frozen to prevent their weights from being updated during training. New classification layers are added on top of the pre-trained model to adapt it to the specific task. The model is then compiled with a categorical cross-entropy loss function, the Adam optimizer, and accuracy as the evaluation metric. This allows the model to leverage the knowledge learned from the pre-trained model and fine-tune it for the new task at hand.

![download (31)](https://github.com/IzabelMatusiewicz/Alien-vs-Predator/assets/101067795/77f4b3b3-c999-44e9-bcaa-84601cce98bd)

### Gradio implementation
Gradio is a Python library used to create user-friendly interfaces for machine learning models. In the provided code, the Gradio interface allows users to interact with the "predict_image" function, which takes an input image, preprocesses it, and passes it through a trained model for prediction. The interface displays the input image and the blended heatmap, providing a visual representation of the predicted class and the areas of the image contributing to the prediction. By launching the interface, users can easily upload their own images, view the predicted class, and gain insights into the model's decision-making process.

[gradio.pdf](https://github.com/IzabelMatusiewicz/Alien-vs-Predator/files/11591149/gradio.pdf)

## Summary
The project focused on the Alien-vs-Predator dataset and aimed to explore various techniques to improve model performance. Initially, a small dataset was augmented using different techniques, including rescaling, rotation, width and height shifting, horizontal flipping, grayscale conversion, and noise augmentation. The baseline augmentation approach yielded the best results and was selected for further modeling. Transfer learning was then employed using the ResNet50 model with pre-trained weights. The pre-trained layers were frozen, and new classification layers were added to adapt the model to the specific task. The Gradio library was utilized to create an interactive interface that allows users to upload images and visualize the model's predictions, providing insights into the model's decision-making process. The project highlighted the effectiveness of augmentation techniques and the benefits of transfer learning, while demonstrating the capabilities of the Gradio library in creating user-friendly model interfaces.

## Contact
https://www.linkedin.com/in/izabela-matusiewicz/
