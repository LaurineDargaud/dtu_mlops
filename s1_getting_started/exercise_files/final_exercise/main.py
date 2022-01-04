import argparse
import sys

import torch

import matplotlib.pyplot as plt

from data import mnist
from model import MyAwesomeModel

from torch import nn, optim


class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.1, type=float)
        # add any additional argument that you want
        parser.add_argument('--nb_epochs', default=30, type=int)
        parser.add_argument('--save_file', default='best_model')
        parser.add_argument('--criterion', default='NLLLoss')
        parser.add_argument('--optimizer', default='SGD')
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement training loop here
        model = MyAwesomeModel()
        trainloader, _ = mnist()
        
        #criterion = nn.CrossEntropyLoss()
        criterion = eval(f'nn.{args.criterion}()')
        optimizer = eval(f'optim.{args.optimizer}(model.parameters(), lr = {args.lr})')
        
        epochs = args.nb_epochs
        steps = 0

        loss_tracking = {}
        best_loss = None

        for e in range(epochs):
            running_loss = 0
            for images, labels in trainloader:
                
                # set model to train mode
                model = model.train()
                
                optimizer.zero_grad()
                
                log_ps = model(images)
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()

            loss_tracking[steps] = running_loss
            steps += 1
            print(f'Epoch {e+1}/{epochs}...   Loss: {running_loss}')

            # save best model
            if best_loss == None or running_loss < best_loss:
                best_loss = running_loss
                torch.save(model, f'{args.save_file}.pth')
            
            # save figure with training loss VS steps
            plt.plot(list(loss_tracking.keys()), list(loss_tracking.values()))
            plt.xlabel('Steps')
            plt.ylabel('Training Loss')
            plt.title(f'Training Loss evolution using {args.criterion} criterion, {args.optimizer} optimizer and {args.nb_epochs} epochs', size=10)
            plt.savefig('training_loss_plot.png')
    
        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default='best_model.pth')
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        model = torch.load(args.load_model_from)
        _, testloader = mnist()
        
        with torch.no_grad():
            # set model to evaluation mode
            model = model.eval()
            all_equals = []
            for images, labels in testloader:
                probas = torch.exp(model(images))
                _, top_class = probas.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                all_equals.append(equals)
            equals = torch.cat(all_equals, 0)
            accuracy = torch.mean(equals.type(torch.FloatTensor)).item()
            print(f'Accuracy: {round(accuracy*100,2)}%')

if __name__ == '__main__':
    TrainOREvaluate()   