# CS 155 Mini Project 1: Songs Classifier

This is the repository for our mini project 1. We will build a songs classifier that can classify songs into different genres based on their audio features.

## Workflow
If you are one of the team members, please refer to [workflow.md](workflow.md) for instructions on how to work with this repository.

## Setup

To set up the environment, please run the following command in the root directory of the repository:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate
pip install -r requirements.txt
```

To update the dependencies, please run the following command in the terminal:
```bash
python -m pipreqs.pipreqs . --force
```

## Run

### Terminal
To run the code in terminal, please follow the instructions below:
```bash
python -m scripts.train_cv
python -m scripts.make_submission
```

### Jupyter Notebook
To quickly run the code in Jupyter Notebook, please open the `notebooks/setup.ipynb` notebook and run the cells in order.
To follow our colab demo in Jupyter Notebook, please open the `notebooks/colab_demo.ipynb` notebook. We put the dataset analysis and model tuning code in this notebook.