import torch
import torch.optim
import torch.nn

# import matplotlib.plt as plt


class ModelTrainer:
    def __init__(self, model):
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.1,
            momentum=0.9,
        )
        self.train_epoch_loss = []
        self.train_epoch_accuracy = []
        self.test_epoch_loss = []
        self.test_epoch_accuracy = []

    def _train(self, train_loader):
        
        # for epoch in range(1, num_epoch+1):
        running_loss = 0.0
        accurate = 0
        total = 0
        self.model.train()

        for images, labels in train_loader:
            # images, labels = images.cuda(), labels.cuda()
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            accurate += (predicted == labels).sum().item()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * accurate / total
        self.train_epoch_loss.append(epoch_loss)
        self.train_epoch_accuracy.append(epoch_accuracy)

    def _test(self, test_loader):
        self.model.eval()
        running_loss = 0.0
        accurate = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                self.optimizer.step()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                accurate += (predicted == labels).sum().item()

                running_loss += loss.item()

        epoch_loss = running_loss / len(test_loader)
        epoch_accuracy = 100 * accurate / total
        self.test_epoch_loss.append(epoch_loss)
        self.test_epoch_accuracy.append(epoch_accuracy)

    def train_pipe(self, train_loader, test_loader, num_epoch):
        for i in range(num_epoch):
            self._train(train_loader)
            self._test(test_loader)
        return self.model

