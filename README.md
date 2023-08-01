# XAI Project 2
## Project 2 - Language Task & Feature Attribution Analysis

# Task Data

## TweetTopic: Twitter Topic Classification
The TweetTopic repository contains a dataset on Twitter topic classification with 6 labels, including timestamps from September 2019 to August 2021. \n
The below link provide direct access to the task the task dataset.
- [Hugging Face Dataset](https://huggingface.co/datasets/cardiffnlp/tweet_topic_single)


# Training and Testing Setup(Linux SetUp)

1. Clone the repository

```git clone [git clone https URL]```

2. Create a Python virtual environment

```
# Update and upgrade
sudo apt update
sudo apt -y upgrade

# check for python version "ideal: 3.8.2"
python3 -V

# install python3-pip
sudo apt install -y python3-pip

# install-venv
sudo apt install -y python3-venv

# Create virtual environment
python3 -m venv my_env


# Activate virtual environment
source my_env/bin/activate
```

3. Install project dependent files

```
pip install requirements.txt
```

4. Run main.py

```
python3 main.py
```

# Project Directory Tree

```
└── Project2/
    ├── classification_dataset.py
    ├── config.yaml
    ├── main.py
    ├── models.py
    ├── trainer.py
    ├── utils.py
    └── requirements.txt
```

# NOTE

```
If there are any dependency issues related to SHAP or LIME not compatibile on local environment try using the .ipynb notebook instead. 
```

## Reference

## Repository Information

- **Repository Name:** TweetTopic
- **Version:** COLING main conference 2022
- **Dataset Labels:** 6
- **Related Repository:** [cardiffnlp/tweet_topic_multi](https://github.com/cardiffnlp/tweet_topic_multi)
