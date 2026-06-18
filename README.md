## Repo Structure

Inspired by [Cookie Cutter Data Science](https://github.com/drivendata/cookiecutter-data-science).

```
├── LICENSE
├── README.md          <- The top-level README for users of this project.
│
├── docs               <- A default Sphinx project; see sphinx-doc.org for details
│
├── Experiments        <- Saves of the Experiments run001 and the different checkpoints of the models
│
├── src                <- Source code for use in this project.
│   │
│   ├── data           <- Script to generate data and make data_loaders
│   │   └── make_dataset.py
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │   │                 predictions
│   │   ├── deeper_deepLearning.py <- Script to run the CNN Deeplearning model
│   │   └── pneumonia_baseline.py  <- Script to run the baseline KNN machine learning model
│   │
│   └── visualisation    <-Files that show the different plots made
│       └── visualise.py <- Script to plot the train/val/test distribution
└────── utils.py         <- Utility file with useful utils

