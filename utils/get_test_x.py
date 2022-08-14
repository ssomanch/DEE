
import numpy as np
import pandas as pd

test_x1,test_x2 = np.meshgrid(np.linspace(0,1,100),np.linspace(0,1,100))
test_x = np.concatenate([test_x1.flatten().reshape(1,-1).T,
                         test_x2.flatten().reshape(1,-1).T],
                       axis=1)
test_x_df = pd.DataFrame(test_x)
test_x_df.columns = ['X1','X2']
test_x_df.to_csv('output/test_grid.csv',index=False)
