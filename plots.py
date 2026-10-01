
from imports import *


#plot training curves

def plot_training_curves(train_loss, val_loss):
    plt.figure(figsize=(10, 6))

    plt.plot(
        train_loss,
        label="Training Loss"
    )

    plt.plot(
        val_loss,
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


#plot validation dice and iou scores

def plot_validation_dice(dice_values):
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(dice_values) + 1),
        dice_values,
        marker="o",
        label="Validation Dice"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title("Validation Dice Score")
    plt.xticks(range(1, len(dice_values) + 1))
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_validation_iou(iou_values):
    plt.figure(figsize=(10, 6))

    plt.plot(
        range(1, len(iou_values) + 1),
        iou_values,
        marker="o",
        label="Validation IoU"
    )
    
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.title("Validation Mean IoU")
    plt.legend()
    plt.grid(True)
    plt.show()