import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


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
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 2000 == 1999:  # Print every 2000 mini-batches
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')
    return model

def net(num_classes):
    '''
    Function that initializes your model
    Remember to use a pretrained model
    '''
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def create_data_loaders(train_dir, val_dir, batch_size):
    '''
    Function to create data loaders for training and testing datasets
    '''
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = torchvision.datasets.ImageFolder(root=val_dir, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def main(args):
    '''
    Main function to initialize model, loss, optimizer, train and test the model
    '''
    model = net(args.num_classes)
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loader, val_loader = create_data_loaders(args.train_dir, args.val_dir, args.batch_size)

    model = train(model, train_loader, loss_criterion, optimizer, args.epochs)

    _, accuracy = test(model, val_loader, loss_criterion)

    # Save the trained model
    torch.save(model.state_dict(), args.model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='/opt/ml/input/data/training', help='Path to training data')
    parser.add_argument('--val_dir', type=str, default='/opt/ml/input/data/validation', help='Path to validation data')
    parser.add_argument('--model_path', type=str, default='/opt/ml/model/model.pth', help='Path to save the trained model')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_classes', type=int, default=133, help='Number of classes')

    args = parser.parse_args()
    main(args)
