"""
Main module which provides
    - loading model
    - training model

"""

from train import Train


def main():

    trainer = Train()
    trainer.train()


if __name__ == '__main__':
     main()