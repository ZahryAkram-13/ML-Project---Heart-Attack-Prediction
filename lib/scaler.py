MINMAX = 'minmax'
STANDARD = 'standard'
ROBUST = 'robust'

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler 
SCALER = {
    MINMAX : MinMaxScaler(), 
    STANDARD: StandardScaler(),
    ROBUST: RobustScaler
}

def get_scaled(**kwarg):
    x_train = kwarg["x_train"]
    x_test = kwarg["x_test"]
    scaler = SCALER[kwarg["scaler"]]
    
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled


