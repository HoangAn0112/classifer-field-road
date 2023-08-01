import numpy
import torch
from tqdm.auto import tqdm
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('default')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_model(epochs, model, optimizer, criterion):
    """
    Function to save the trained model to disk.
    """
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, 'output/model.pth')

def save_plots(train_acc, valid_acc, train_loss, valid_loss):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('output/accuracy.png')
    
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('output/loss.png')

# training
def train(model, trainloader, optimizer, criterion):
    print('Training')
    model.eval()
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # forward pass
        outputs = model(image)
        # calculate the loss
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # calculate the accuracy
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # backpropagation
        loss.backward()
        # update the optimizer parameters
        optimizer.step()
    
    # loss and accuracy for the complete epoch
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc

# validation
def validate(model, testloader, criterion):
    print('Validation')
    model.eval()
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # forward pass
            outputs = model(image)
            # calculate the loss
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
        
    # loss and accuracy for the complete epoch
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc

def visualize_test_predictions(model, test_loader, class_labels, num_images=10, figsize=(8, 8)):
    images, predicted_labels , true_labels = [], [], []
    model.eval()
    with torch.no_grad():
        for inputs, target in test_loader:
            inputs = inputs.to(device)
            target = target.to(device)
                    
            outputs = model(inputs)
            _, predicted_classes = torch.max(outputs, 1)

            images.extend(inputs)
            predicted_labels.extend(predicted_classes.numpy())
            true_labels.extend(target.numpy())

    #display test images
    num_images = min(num_images, len(images))
    rows = int(numpy.sqrt(num_images))
    cols = int(numpy.ceil(num_images / rows))

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    correct = 0

    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)
        image = images[i].numpy().transpose(1, 2, 0)
        image = image * [0.24362235, 0.2223072, 0.2965594] + [0.5374407, 0.5403282, 0.438259]
        plt.imshow(image)
        plt.title(f"Predict {class_labels[predicted_labels[i].item()]}", fontsize=10)
        plt.axis('off')

        if predicted_labels[i].item() == true_labels[i]:
            correct += 1

     # Hide any remaining empty subplots
    for i in range(num_images, rows * cols):
        axes[i // cols, i % cols].axis('off')

    accuracy = 100.0 * correct / num_images
    plt.subplots_adjust(wspace=0.5, hspace=0.5)  # Adjust the horizontal and vertical gap between rows and columns
    fig.suptitle(f"Accuracy on test data: {accuracy:.2f}%")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()