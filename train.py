from torch import nn
import torch # Added import for torch.flatten

# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # Define layers of the neural network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Added pooling layer

        # Changed out_channels to 256 to match the input of fc1 after adaptive pooling
        self.conv2 = nn.Conv2d(64, 256, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Added pooling layer

        # Added adaptive pooling to get a 1x1 spatial output
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(256, 200) # 200 is the number of classes in TinyImageNet

    def forward(self, x):
        # Define forward pass
        # B x 3 x 224 x 224
        x = self.conv1(x).relu() # B x 64 x 224 x 224
        x = self.pool1(x)        # B x 64 x 112 x 112

        x = self.conv2(x).relu() # B x 256 x 112 x 112 (after conv2 change)
        x = self.pool2(x)        # B x 256 x 56 x 56

        x = self.adaptive_pool(x) # B x 256 x 1 x 1
        x = torch.flatten(x, 1)   # B x 256 (flatten from dimension 1)

        x = self.fc1(x)
        return x
    
def train(epoch, model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')

def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0

    correct, total = 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100. * correct / total

    print(f'Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')
    return val_accuracy

model = CustomNet().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

best_acc = 0

# Run the training process for {num_epochs} epochs
num_epochs = 10
for epoch in range(1, num_epochs + 1):
    train(epoch, model, train_loader, criterion, optimizer)

    # At the end of each training iteration, perform a validation step
    val_accuracy = validate(model, val_loader, criterion)

    # Best validation accuracy
    best_acc = max(best_acc, val_accuracy)


print(f'Best validation accuracy: {best_acc:.2f}%')
