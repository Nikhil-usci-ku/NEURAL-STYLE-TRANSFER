import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import copy

# Device configuration (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Image size and loader for preprocessing
imsize = 512
loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor()
])

# Function to load and preprocess images
def load_image(img_path):
    image = Image.open(img_path).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

cimg = str(input("Enter Content image location : "))
content_img = load_image(cimg)

simg = str(input("Enter Style image location : "))
style_img = load_image("style.jpg")

# Ensure content and style images are the same size
assert content_img.shape == style_img.shape, "Images must be the same size"

# The input image to be optimized, initialized with content image and tracking gradients
input_img = content_img.clone().requires_grad_(True)

# Normalization values for VGG network, standard for ImageNet
normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# Normalization module to apply mean and std
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

# Content loss module, computes MSE between feature maps
class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach() # Detach target to prevent gradient computation

    def forward(self, x):
        self.loss = nn.functional.mse_loss(x, self.target)
        return x

# Function to compute the Gram matrix for style representation
def gram_matrix(x):
    b, c, h, w = x.size()
    features = x.view(b * c, h * w)
    G = torch.mm(features, features.t())
    return G.div(b * c * h * w)

# Style loss module, computes MSE between Gram matrices
class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach() # Compute and detach target Gram matrix

    def forward(self, x):
        G = gram_matrix(x)
        self.loss = nn.functional.mse_loss(G, self.target)
        return x

# Load pre-trained VGG-19 features and set to evaluation mode
cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()

# Define layers for content and style extraction
content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

# Function to build the model with embedded loss modules
def get_model_and_losses(cnn, norm_mean, norm_std, style_img, content_img):
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(norm_mean, norm_std).to(device)

    content_losses, style_losses = [], []
    model = nn.Sequential(normalization) # Start sequential model with normalization

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False) # Replace in-place ReLU
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            continue

        model.add_module(name, layer) # Add VGG layer to the new model

        # If it's a content layer, compute target features and add ContentLoss
        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        # If it's a style layer, compute target features and add StyleLoss
        if name in style_layers:
            target = model(style_img).detach()
            style_loss = StyleLoss(target)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    return model, style_losses, content_losses

# Build the complete style transfer model
model, style_losses, content_losses = get_model_and_losses(
    cnn, normalization_mean, normalization_std, style_img, content_img
)

# Weights for balancing style and content loss
style_weight = 5e6
content_weight = 1e4

# LBFGS optimizer for image optimization
optimizer = optim.LBFGS([input_img])

print("Optimizing...")

run = [0]
while run[0] <= 300: # Optimization loop for a fixed number of steps

    # Closure function required by LBFGS, performs forward/backward pass
    def closure():
        # Clamp pixel values to [0, 1] range
        with torch.no_grad():
            input_img.clamp_(0, 1)

        optimizer.zero_grad() # Clear gradients
        model(input_img) # Forward pass, computes losses

        style_score = sum(sl.loss for sl in style_losses)
        content_score = sum(cl.loss for cl in content_losses)

        loss = style_weight * style_score + content_weight * content_score
        if torch.isnan(loss):
            raise ValueError("Loss became NaN")

        loss.backward() # Backpropagate the total loss
        run[0] += 1
        if run[0] % 50 == 0:
            print(f"Step {run[0]}: Style: {style_score.item():.2f}, Content: {content_score.item():.2f}")
        return loss

    optimizer.step(closure) # Perform one optimization step

# Final clamping and saving of the generated image
with torch.no_grad():
    input_img.clamp_(0, 1)

unloader = transforms.ToPILImage()
image = input_img.cpu().clone().squeeze(0)
image = unloader(image)
image.save("Edited.png")
print("New image saved as Edited.png")
