
'''
VAE-variational autoencoders
a generative model that consists of an encoder which compress the input and a decodera(which reconstructs or decompress the input from the compressed data)
'''


#importing pytorch library
import torch
import torch.nn as nn #for buildimg layers in the neural network
import torch.optim as optim #for optimization algorithm
import torchvision#pytorch computer vision library for img data
import torchvision.transforms as transforms#img transfoormation
from torch.utils.data import DataLoader #load data in batches
import matplotlib.pyplot as plt
import numpy as np

#define a class for our VAE extending the neural network module class of pytorch(we are teying to wind it ina class)
class VAE(nn.Module):
  def __init__(self,input_dim=784,hidden_dim=400,latent_dim=20):
    #intialize the VAE parameters
    #input_dim:no of input features(28x28=784 for MNIST)
    #hidden_dim:no of nodes in hidden layer (~half of input)
    #latent_dim:no of dimension in latent space(compressed)

    #call the constructor of the parent class
    super(VAE,self).__init__() #nerthe kodtha vars elam self varil ond ath pass aki kodukum parnt classine vilakumbo


    #ENCODER NETWORK:
    ###################
    #create an encoder that converts input into latent dimension
    #create an encoder layer object of sequential class
    self.encoder=nn.Sequential(   #inside this we will define each layers
        nn.Linear(input_dim,hidden_dim),#first fully connected layer with linear activation funct
        nn.ReLU(),#ReLu activation
    )
    #create a seperate layer for learning mean
    self.mu_layer=nn.Linear(hidden_dim,latent_dim)
    #create another seperate layer for learning variance
    self.log_var_layer=nn.Linear(hidden_dim,latent_dim)

    #DECODER NETWORK:
    ##################
    #create a decoder that converts latent dim into output
    #create an Decoder layer object of sequential class
    self.decoder=nn.Sequential(
        nn.Linear(latent_dim,hidden_dim),#first fully connected layer with linear activation funct
        nn.ReLU(),#ReLu activation function
        nn.Linear(hidden_dim,input_dim),
        nn.Sigmoid())#to make sure output values are btwn zero and one,use a sigmoid activation function

  #within the class,define the fn for reparametrization
  def reparametrize (self,mu,log_var):
    #randomness z = mean + std * random noise epsilon
    std= torch.exp(0.5*log_var)
    epsilon= torch.randn_like(std)
    return mu+std*epsilon

  #fn for forward pass of the neural network(both enc and dec)
  def forward(self,x):
    x=self.encoder(x)#encode the input data
    mu=self.mu_layer(x)#finding mean
    log_var=self.log_var_layer(x)#finding log variance
    z=self.reparametrize(mu,log_var)#using reparametrization technq
    reconstructed_x=self.decoder(z)#reconstruct the input data(decoder)
    return reconstructed_x,mu,log_var

#define the loss function and do the loss calculation
#VAE loss fn consists of two types of los calculation
#1.Reconstruction loss-checking if output is similar to input
#2.KL divergence loss-ensure the learned latent space is following a normal distribution
def loss_function(reconstructed_x,x,mu,log_var):
  #calculate reconstruction loss
  recon_loss=nn.functional.binary_cross_entropy(reconstructed_x,x,reduction='sum')
  #calculate KL divergence loss
  kl_loss=-0.5*torch.sum(1+log_var-mu.pow(2)-log_var.exp())
  return recon_loss+kl_loss

#Load the mnist dataset
#step 1 -transform the 28*28 images inro 1D vector
#create a transform object for that
transform= transforms.Compose([
    transforms.ToTensor(),#convert the image to tensor
    transforms.Lambda(lambda x: x.view(-1))#flatten the image to an 1D image
])

#step2- Load the dataset
train_dataset= torchvision.datasets.MNIST(root="./data",train=True,transform=transform,download=True)

#to load the data in batches,use the DataLoader class obj
train_loader=DataLoader(train_dataset,batch_size=64,shuffle=True)

#Train the VAE
#step1: initialize the model,optimizer and device
device=torch.device("cuda"if torch.cuda.is_available() else "cpu")
model=VAE().to(device)#load it to the device
optimizer=optim.Adam(model.parameters(),lr=1e-3)

# Re-initialize the model, optimizer, and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)  # Re-instantiate the model to pick up the corrected VAE class
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("Model re-initialized successfully.")

# Restart the Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    for batch_idx, (x, _) in enumerate(train_loader):
        # FORWARD PASS
        # move data to the device
        x = x.to(device)
        # using the model, reconstruct/generate the image
        reconstructed_x, mu, log_var = model(x)
        # calculate the loss using the available details
        loss = loss_function(reconstructed_x, x, mu, log_var)

        # BACKPROPAGATION
        # reset the gradients of the optimizer
        optimizer.zero_grad()
        # compute the gradients
        loss.backward()
        # update model params using optimizer
        optimizer.step()

        # calculate total loss
        total_loss += loss.item()
    # for each epoch print the training details
    print(f"Epoch[{epoch+1}/{num_epochs}],Loss:{total_loss/len(train_loader.dataset):.4f}")

#generate a set of images using our trained model
import matplotlib.pyplot as plt

with torch.no_grad():
  #create a blank random points in latent space
  z=torch.randn(16,20).to(device)
  #using this blank latent space as input,our model should generate image
  generated_images=model.decoder(z).cpu()

#plot the generated 16 images in a 4x4 grid
fig,axes = plt.subplots(4,4,figsize=(5,5))
#show the image as loop
for i,ax in enumerate(axes.flatten()):
  ax.imshow(generated_images[i].view(28,28).numpy(), cmap="gray")
  ax.axis('off')
#show the drawn plot
plt.show()