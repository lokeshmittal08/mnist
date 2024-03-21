import os
import unittest
import torch
import argparse  # Add this line
from torchvision import datasets, transforms
import mlflow
from train import Net, train, test, main


class TestModelCheckpointManagement(unittest.TestCase):
    def setUp(self):
        # Define default parameters
        self.batch_size = 64
        self.epochs = 1  # train for 2 epochs for testing purpose
        self.lr = 0.001
        self.log_interval = 10
        self.dry_run = False
        self.seed = 1
        self.no_cuda = False
        self.no_mps = False
        self.gamma = 0.7
        self.test_batch_size = 1000

        # Initialize model, optimizer, and data loaders
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Net().to(self.device)
        self.optimizer = torch.optim.Adadelta(self.model.parameters(), lr=self.lr)
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])), batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])), batch_size=self.test_batch_size, shuffle=True)

    def test_train_and_test(self):
        args = argparse.Namespace(
            batch_size=self.batch_size,
            test_batch_size=self.test_batch_size,
            epochs=self.epochs,
            lr=self.lr,
            gamma=self.gamma,
            no_cuda=self.no_cuda,
            no_mps=self.no_mps,
            dry_run=self.dry_run,
            seed=self.seed,
            log_interval=self.log_interval
        )

        # Train the model
        for epoch in range(1, self.epochs + 1):
            train_loss = train(args, self.model, self.device, self.train_loader, self.optimizer, epoch)
            test_loss, accuracy = test(self.model, self.device, self.test_loader)

        # Check that the model checkpoint is saved
        self.assertTrue(os.path.exists("best_model_checkpoint.pt"))

        # Check that the MLFlow run has been started and ended
        self.assertTrue(mlflow.active_run())
        mlflow.end_run()

    def tearDown(self):
        # Remove the saved model checkpoint file
        if os.path.exists("best_model_checkpoint.pt"):
            os.remove("best_model_checkpoint.pt")


if __name__ == '__main__':
    unittest.main()
