import argparse
import sys

import torch
import click
from torch import nn, optim
import torch.nn.functional as F

from data import mnist
from model import MyAwesomeModel


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train, _ = mnist()
    a=zip(train['images'], train['labels'].T)
    trainloader=torch.utils.data.DataLoader(list(a), batch_size=64, shuffle=True)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epochs = 30
    steps = 0

    train_losses, test_losses = [], []
    for e in range(epochs):
        running_loss = 0
        for batch in trainloader:
            #batch = next(iter(trainloader))
            images = batch[0]
            labels = batch[1]
            images = images.resize_(images.size()[0], 784)

            optimizer.zero_grad()

            log_ps = model(images.float())
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
    torch.save(model.state_dict(), 'trained_model.pt')



@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    state_dict = torch.load(model_checkpoint)
    model=MyAwesomeModel()
    model.load_state_dict(state_dict)
    _, test_set = mnist()
    a=zip(test_set['images'], test_set['labels'].T)
    testloader=torch.utils.data.DataLoader(list(a), batch_size=64, shuffle=True)
    
    
    with torch.no_grad():
        accuracy = 0
        test_loss = 0
        n=0
        for batch in testloader:
            images=batch[0]
            labels=batch[1]
            criterion = nn.NLLLoss()
            #optimizer = optim.Adam(model.parameters(), lr=lr)
            
            n+=1
            images = images.resize_(images.size()[0], 784)

            output = model.forward(images.float())
            test_loss += criterion(output, labels).item()

            ## Calculating the accuracy 
            # Model's output is log-softmax, take exponential to get the probabilities
            ps = torch.exp(output)
            # Class with highest probability is our predicted class, compare with true label
            equality = (labels.data == ps.max(1)[1])
            # Accuracy is number of correct predictions divided by all predictions, just take the mean
            accuracy += equality.type_as(torch.FloatTensor()).mean()
        print(accuracy/n)


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()


    
    
    
    