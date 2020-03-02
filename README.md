# Autonomous systems 1: Object detection

## The goal of the project

We want to take the [MobileNet](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/) implementation with PyTorch and try to extend it.

The goal is to use some data from the [Food 101 dataset](https://www.kaggle.com/kmader/food41) and try to use it to extend MobileNet so that it can classify pictures of food as well.

## How to set up an environment with Python

Make sure you are using Python 3: `python --version`. On Ubuntu, you can probably use the `python3` command to resolve any ambiguities.

Create an environment directory, the packages you install with `pip` (for instance, PyTorch) will be stored there. I put it in the "env" directory.

	virtualenv -p python3 env

Activate the virtual environment:

	# If you are using bash
	source env/bin/activate

Your prompt should have if the activation of the virtual env worked.

## Installing Jupyter

If you want to use a jupyter notebook, you can install Jupyter with

	pip install jupyter

And then:

	jupyter notebook

## Installing PyTorch

You need to go on [the PyTorch page](https://pytorch.org/get-started/locally/#mac-anaconda) and choose from a selector the things you have, it will generate a command to paste in your terminal to install it.
