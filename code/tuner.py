import numpy as np
import itertools
from driver import Driver

z_dim_array        =  (2   ** np.arange(5)).tolist()
time_steps_array   =  (2   ** np.arange(5)).tolist()
outlier_frac_array =  (0.1 ** np.arange(4)).tolist()[1:]
n_hidden           =  (2   ** np.arange(4)).tolist()
batch_size_array   =  (2   ** np.arange(5)).tolist()
l_rate_array       =  (0.1 ** np.arange(3)).tolist()
train_iters        =  (200 * np.arange(15)).tolist()[1:]
# print(train_iters)
# print(outlier_frac_array)
param_list         = [z_dim_array, time_steps_array, outlier_frac_array, n_hidden, batch_size_array, l_rate_array,train_iters]
res = list(itertools.product(*param_list))
# print(res)
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

print("params before:", params)
for parameters in res :
    params['z_dim'] = parameters[0]
    params['time_steps'] = parameters[1]
    params['outlier_fraction'] = parameters[2]
    params['n_hidden'] = parameters[3]
    params['batch_size'] = parameters[4]
    params['learning_rate'] = parameters[5]
    params['train_iters'] = parameters[6]
    

    

