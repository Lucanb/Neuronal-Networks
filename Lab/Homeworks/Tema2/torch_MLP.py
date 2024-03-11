import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import accuracy_score, f1_score

class MyNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        init.xavier_uniform_(self.fc1.weight)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        init.xavier_uniform_(self.fc2.weight)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

class DigitClassifier:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001):
        self.model = MyNN(input_size, hidden_size, output_size)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self, train_loader, epochs=5):
        for epoch in range(epochs):
            for inputs, labels in train_loader:
                self.optimizer.zero_grad()
                inputs = inputs.view(inputs.size(0), -1)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

    def evaluate(self, data_loader):
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.view(inputs.size(0), -1)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average=None)
        return accuracy, f1

input_size = 784
hidden_size = 128
output_size = 10

classifier = DigitClassifier(input_size, hidden_size, output_size)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
batch_size = 100
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
classifier.train(train_loader, epochs=5)
test_accuracy, test_f1 = classifier.evaluate(test_loader)
print(f'Accuracy: {test_accuracy} \n f1 : {test_f1}')
