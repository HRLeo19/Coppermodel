
# Project Title

Industrial Copper Model Predition

This Project focus on Predicting the Price value and Predicting the status where it is "won" or "lost" using Machine Learning models of Regression and Classification.




## Packages

Python

import numpy as np

import pickle

import streamlit as st,
from streamlit_option_menu import option_menu
## Intro

This Project is basically related to predicting the price value of the copper and the predicting the status won or lost which means capturing the leads how likely they will become a customer.

So to do this , with the old data of the copper modeling will train ML best regression model for price predictions and ML best Classifiction model to predict the status. 
## Demo




## Deployment

Take all the relevant data from the repository, like py file and all pickle files which are trained models file and load it in the appropariate variable in the py file.

mode=LRmodel.pkl
ohe1=oheit.pkl
ohe2=ohes.pkl
scalar=LRscalar.pkl
rfc=RFCmodel.pkl
clscalar=Classifyscalar.pkl

To deploy this project run in command prompt

```bash
streamlit run app.py
```

check demo video...
## What can expect

After running the streamlit app.

In the price predictions option , give all the relative informations needed in the boxes like

1.Thickness

2.Width

3.Quantity in tons

4.Country code

5.Item type

6.Product ref id

7.Customer id

8.Application Number

9.Status

After filling all the details press the "predict the price" option to check the price.

SAme to be filled in the Status check option to check the future status , one change is instead of status column there will price box, so update it and select the "predict the status" option to check the status .


## Note

All the manual enter values should be in numbers to perform the model correctly.

The Price Prediction model is with 70% accuracy.

The Status Prediction model is with 90% accuracy.

Thanks...