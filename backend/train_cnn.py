import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from emotion_cnn import EmotionCNN
from dataset_builder import prepare_dataset

import matplotlib.pyplot as plt

# === Hyperparameters ===
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load dataset ===
data_dir = "/Users/hanyildirim/Downloads/Audio_Speech_Actors_01-24"
train_dataset, val_dataset, label_encoder = prepare_dataset(data_dir)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# === Model, optimizer, loss ===
model = EmotionCNN(num_classes=3).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

train_losses, val_accuracies = [], []
best_val_acc = 0

# === Training loop ===
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

    # === Validation ===
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = correct / total
    val_accuracies.append(val_acc)

    print(f"📚 Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Accuracy: {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "emotion_cnn.pt")
        print("💾 Saved best model so far!")

# === Plotting results ===
plt.plot(train_losses, label="Loss")
plt.plot(val_accuracies, label="Val Accuracy")
plt.title("Training Loss & Validation Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("training_plot.png")
plt.show()