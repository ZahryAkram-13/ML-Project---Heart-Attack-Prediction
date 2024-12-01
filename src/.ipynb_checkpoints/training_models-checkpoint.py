

def train():
    
    import pandas as pd 
    import numpy as np
    import matplotlib.pyplot as plt
    import plotly.express as px
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import seaborn as sns
    
    random_state = 42
    test_size = 0.2
    
    import sys
    sys.path.append('../lib')
    from models import apply_model, MODELS
    from scaler import SCALER, MINMAX, STANDARD, ROBUST, get_scaled
    from tools import get_xy_data
    
    
    data_path = '../data/processed/randomForest_selected_features.csv'
    rfe_data = '../data/processed/RFE_selected_features.csv'
    data = get_xy_data(rfe_data)
    
    
    df = data['df']
    columns = data['columns']
    target  = data['target']
    columns
    
    x = data['x']
    y = data['y']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state = 42)
    
    #normalisation
    x_train_scaled, x_test_scaled = get_scaled(x_train = x_train, x_test = x_test, scaler = MINMAX)
    
    #for code see models.py
    
    for model_name in list(MODELS.keys()):
        print("******************", model_name, "****************************")
        apply_model(x_train_scaled = x_train, y_train = y_train,
                    x_test_scaled = x_test, y_test = y_test,
                    model_name = model_name)
        print("**************************************************************")
        print("\n \n \n")