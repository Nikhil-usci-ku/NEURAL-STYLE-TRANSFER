# NEURAL-STYLE-TRANSFER

*COMPANY*: CODETECH IT SOLUTIONS

*NAME*: NIKHIL KUMAR

*INTERN ID*: CODF69

*DOMAIN*: AIML

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTHOSH KUMAR

> This repository contains a Python script that implements Neural Style Transfer, a technique to recompose the content of one image in the artistic style of another.

> Unlike traditional image filters, this solution leverages deep learning to intelligently combine visual elements, creating unique artistic outputs.

##   Description

This Python code precisely implements Neural Style Transfer, a technique that allows the artistic style of one image to be imposed onto the content of another. It cleverly utilizes a pre-trained VGG-19 Convolutional Neural Network (CNN) to extract abstract representations of images, then optimizes a new image to blend these representations.

Here’s a concise breakdown of its key components:

### 1. Setup and Image Preprocessing:
* The script first configures the device to leverage GPU acceleration (CUDA) if available, falling back to CPU otherwise.
* Images are standardized to a uniform imsize (256x256 pixels) using torchvision.transforms.Resize and converted to PyTorch tensors (ToTensor()) for network input.
* A load_image utility handles opening, converting to RGB, applying transformations, and adding a batch dimension (unsqueeze(0)).
* Crucially, input_img is initialized as a clone of the content_img with requires_grad_(True). This flags it as the parameter to be continuously adjusted by the optimizer during the style transfer process.
* Standard normalization_mean and normalization_std for ImageNet are applied, ensuring consistency with the pre-trained VGG model's training data.

### 2. Custom Loss Modules:
* Normalization: A simple nn.Module applies the aforementioned mean and standard deviation normalization as an integral part of the model's forward pass.
* ContentLoss: This module quantifies content similarity. It calculates the Mean Squared Error (MSE) between feature maps extracted from the input_img and pre-computed target feature maps from the content_img at specific VGG layers. The target features are detached() to prevent gradients from flowing back into the content image itself.
* gram_matrix: This fundamental function computes the Gram matrix of a feature map. The Gram matrix represents the style by capturing the correlations between different feature channels, effectively encoding textural information and artistic patterns.
* StyleLoss: This module measures style similarity. It calculates the MSE between the Gram matrix of the input_img's features and the pre-computed, detached Gram matrix of the style_img's features at designated VGG layers.

### 3. Model Construction and Loss Integration:
* A pre-trained models.vgg19().features is loaded and set to .eval() mode, ensuring it acts solely as a feature extractor without affecting its internal state (e.g., BatchNorm layers).
* content_layers and style_layers lists specify which VGG layers will contribute to content and style loss, respectively. These are chosen empirically to balance abstract content and fine-grained style details.
* The get_model_and_losses function dynamically constructs a new nn.Sequential model. It iterates through the VGG layers, adding them sequentially. When a layer matches a designated content or style layer, a corresponding ContentLoss or StyleLoss module is inserted. This ingenious design automatically calculates loss values during the single forward pass of the input_img through the entire modified network.

### 4. Optimization Loop:
* style_weight and content_weight hyperparameters control the relative influence of style and content objectives; a higher content weight typically preserves the content structure more.
* torch.optim.LBFGS is chosen as the optimizer, known for its efficiency in high-dimensional image optimization tasks like this. It optimizes the input_img tensor.
* The optimization proceeds in a while loop for a fixed number of iterations (e.g., 300).
* Within each iteration, a closure function is executed (required by LBFGS). This closure performs a full forward pass of input_img through the model, sums the weighted style_score and content_score to get the total loss, and then calls loss.backward() to compute gradients.
* input_img.clamp_(0, 1) is regularly applied to keep pixel values within a valid image range, preventing artifacts.

### 5. Output Generation:
* After the optimization loop completes, the final input_img is clamped again.
* It's then converted from a PyTorch tensor back into a standard PIL image using transforms.ToPILImage() and saved as output.png, representing the artistic fusion.


### Features
* Neural Style Transfer: Combines content from one image with style from another.
* VGG-19 Integration: Leverages a pre-trained VGG-19 CNN for robust feature extraction.
* Content and Style Loss: Utilizes Mean Squared Error for content matching and Gram matrices for style matching.
* LBFGS Optimization: Employs the LBFGS optimizer for efficient convergence.
* Configurable Weights: Allows adjustment of content and style loss weights to control the blend.

## Prerequisites
> Python 3.12

## Installation
Clone the repository (or download the zip file):

`git clone https://github.com/Nikhil-usci-ku/NEURAL-STYLE-TRANSFER.git`

`cd NEURAL-STYLE-TRANSFER`

Install the necessary Python libraries provided in requirements.txt file:

`pip install -r requirements.txt`

> The script will attempt to download required models automatically upon first run.

## Running

Run the Python script Task3.py directly.

Enter the image location of Base content.

Enter the image location of styling image.

The new image is stored in `edited.png`

## Output

* User Input

  + Console
    
    ![image](https://github.com/user-attachments/assets/70eaa862-b0e0-4a45-88f4-7788d1131f7b)

  + content.jpg
    
    ![content](https://github.com/user-attachments/assets/40c45ed2-856f-4c7f-8e98-3668d8aaec58)

  + style.jpg
    
    ![style](https://github.com/user-attachments/assets/cc31b295-d507-4417-bced-bf13b04c32dd)



* Output Image
  
  + Console
 
    ![image](https://github.com/user-attachments/assets/d3bda7cb-ec59-4156-a24b-2fca43ce52f6)

  + Edited Image
      
    ![Edited](https://github.com/user-attachments/assets/6ba783fc-eeb8-4b35-9c24-b52c27502b97)


