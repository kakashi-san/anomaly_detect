import sys
import warnings
import os
import numpy as np
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# if not sys.warnoptions:
#     warnings.simplefilter("ignore")
from lstm_vae import LSTM_VAE

params = {'z_dim': 8,
              'time_steps': 16,
              'outlier_fraction': 0.01,
              'n_hidden': 16,
              'batch_size': 128,
              'learning_rate': 0.0005,
              'train_iters': 400,
              'data_path': '../data/data.csv',
              'data_columns': ['delT','wE1','wE2','wE3','wE4','wE5','ISP.0090.0003C','ISS.0007.0001E','ISS.0012.0012W','ISS.0024.0009E','ISS.0048.0013E','ISS.0053.0002C'],
              }
def Driver(params):
    z_dim            = params['z_dim']
    time_steps       = params['time_steps']
    outlier_fraction = params['outlier_fraction']
    n_hidden         = params['n_hidden']
    batch_size       = params['batch_size']
    learning_rate    = params['learning_rate']
    train_iters      = params['train_iters']
    dataset_name     = params['data_path']
    data_columns     = params['data_columns']

    lstm_vae = LSTM_VAE(dataset_name = dataset_name,
        columns= data_columns, 
        z_dim = z_dim, 
        time_steps = time_steps, 
        outlier_fraction = outlier_fraction, 
        n_hidden = n_hidden, 
        batch_size = batch_size, 
        learning_rate = learning_rate, 
        train_iters = train_iters)
    
    lstm_vae.train()
    # lstm_vae.scores
    if not os.path.exists('./processed'):
        os.mkdir('./processed')
        with open('./processed/scores.pkl', 'wb') as fh:
            fh.write(lstm_vae.scores)
    
    np.savetxt("./processed/anomaly_scores.csv", 
           lstm_vae.scores,
           delimiter =", ", 
           fmt ='% s')


    

drive = Driver(params=params)