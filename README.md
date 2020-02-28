# wine-quality-predictor
A demo on how to package and deploy machine learning code.
In this sample, we learn to:
- make a proper python package, enabling portability of the machine learning training code
- write a unit test
- write a custom Command Line Tool
- automatically deploy our package to a private repository if the unit tests are successful

Once the whole pipeline has been executed, the package is available for installation via pip.

## QUICKSTART

### Package installation in a virtual env

```bash
virtualenv venv_wqp
source venv_wqp/bin/activate
pip install wqp==1.0.0 --extra-index-url https://pypi.fury.io/ngallot/
```

### Train wine quality predictor

```bash
wqp jobs train --data-path http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
```

Output:
```bash
INFO:wqp.main:Starting wine quality predictor training...
INFO:wqp.main:Fetching data...
INFO:wqp.main:Building train and test datasets...
INFO:wqp.main:Fitting model...
INFO:wqp.main:Evaluating model...
INFO:wqp.main:Finished model evaluation. Metrics: {'rmse': 0.7401465102553071, 'mae': 0.6175043256143696, 'r2': 0.04206888455849722}
```

That's it!

NB: I emphasize the fact that here, we absolutely do not care about the performance of the machine learning model, as it's out of the topic :-)


