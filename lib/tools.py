
test = 't'

def get_xy_data(path):
    import pandas as pd 
    import numpy as np

    data = {}
    
    df = pd.read_csv(path, sep=',')
    columns = df.columns.to_numpy()
    data['df'] = df
    target  = "Heart Attack Risk"
    x = df.drop(target, axis = 1)
    y = df[target]
    data['x'] = x
    data['y'] = y
    data['columns'] = columns
    data['target'] = target
    
    return data