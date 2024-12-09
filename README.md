# InPoDA Project - IN304

### Authors: [@Egeyae](https://github.com/Egeyae) and [@JulienKonstantinov](https://github.com/JulienKonstantinov)

---

## About InPoDA

InPoDA is a project given during 3rd Semester at UVSQ, France.
The final goal is to provide a small application that extract tweets from a given database and then perform some data analysis on the extracted tweets.

Our project uses JSON parsing for data loading, SentenceTransformers for Topics extraction and textblob for sentiment analysis.
A side project was to implement from scratch a Genetic Algorithm to train a sentiment analysis model.
Unfortunately, the performances were not the ones expected so we used textblob for this part.

---

## Installation

---
It is first recommended to use a Python Virtual environnement, using virtualenv:
```shell
python3 -m virtualenv .venv
# For Windows:
.\.venv\Scripts\activate

# For Unix (MacOS, Linux):
source ./.venv/bin/activate.sh
```
---


**WARNING:** When this project was made, PyTorch only supported Python 3.12.x ! If you have any troubles installing this package, please check the [official PyTorch page](https://pytorch.org/get-started/locally/) first.
