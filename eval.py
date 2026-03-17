import dataset.dataset
import utils.utils
import train

best_acc = 0

# Run the training process for {num_epochs} epochs
num_epochs = 10
for epoch in range(1, num_epochs + 1):
    utils.utils.train(epoch, train.model, dataset.dataset.train_loader, train.criterion, train.optimizer)

    # At the end of each training iteration, perform a validation step
    val_accuracy = utils.utils.validate(train.model, dataset.dataset.val_loader, train.criterion)

    # Best validation accuracy
    best_acc = max(best_acc, val_accuracy)


print(f'Best validation accuracy: {best_acc:.2f}%')