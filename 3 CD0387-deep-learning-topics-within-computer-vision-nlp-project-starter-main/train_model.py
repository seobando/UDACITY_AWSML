# Import your dependencies.
# For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse

# Import dependencies for Debugging and Profiling
import torch.autograd.profiler as profiler

def test(model, test_loader, criterion):
    '''
    Function that takes a model and a testing data loader and returns the test accuracy/loss of the model
    Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    test_loss /= len(test_loader.dataset)
    accuracy = correct / total

    print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')
    return test_loss, accuracy

def train(model, train_loader, criterion, optimizer, epochs=5):
    '''
    Function that takes a model and data loaders for training and trains the model
    Remember to include any debugging/profiling hooks that you might need
    '''
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            with profiler.profile(record_shapes=True) as prof:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # Print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # Print every 2000 mini-batches
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')
    return model

def create_data_loaders(data_dir, batch_size):
    '''
    Function to create data loaders for training and testing datasets
    '''
    transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    train_dataset = torchvision.datasets.ImageFolder(root=f'{data_dir}/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = torchvision.datasets.ImageFolder(root=f'{data_dir}/test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def main(args):
    '''
    Main function to initialize model, loss, optimizer, train and test the model
    '''
    # Initialize a model by calling the net function
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, args.num_classes)

    # Create your loss and optimizer
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Create data loaders
    train_loader, test_loader = create_data_loaders(args.data_dir, args.batch_size)

    # Call the train function to start training your model
    model = train(model, train_loader, loss_criterion, optimizer, args.epochs)

    # Test the model to see its accuracy
    test(model, test_loader, loss_criterion)

    # Save the trained model
    torch.save(model.state_dict(), args.model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Specify any training args that you might need
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing the dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_classes', type=int, default=120, help='Number of dog breeds/classes')
    parser.add_argument('--model_path', type=str, default='model.pth', help='Path to save the trained model')
    
    args = parser.parse_args()
    
    main(args)
 that you might need
    '''
    
    args=parser.parse_args()
    
    main(args)
