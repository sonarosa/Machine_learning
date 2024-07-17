 import torch
 from torch.utils.data import DataLoader, random_split
 from torchvision import datasets, transforms
 # Define a transform to normalize the data
 transform = transforms.Compose([
 transforms.ToTensor(), # Convert the image to a tensor
 transforms.Normalize((0.5,), (0.5,))
 ])
 # Load the MNIST dataset
 train_dataset = datasets.MNIST(root=’./data’,
 train=True, download=True, transform=transform)
 test_dataset = datasets.MNIST(root=’./data’,
 train=False, download=True, transform=transform)
 # Split the train_dataset into training and validation sets
 train_size = int(0.8 * len(train_dataset))
 val_size = len(train_dataset)- train_size
 train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
 # Create DataLoaders for each set
 batch_size = 64
 train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
 val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
 test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

 # Verify the sizes of the datasets
 print(f’Training set size: {len(train_dataset)}’)
 print(f’Validation set size: {len(val_dataset)}’)
 print(f’Test set size: {len(test_dataset)}’)
