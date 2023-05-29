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
