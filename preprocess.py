import pandas as pd
import numpy as np

def load_and_prepare(filepath):
    nfl_df = pd.read_csv(filepath)

    nfl_df['True_Total'] = nfl_df['Tm_Pts'] + nfl_df['Opp_Pts']
    nfl_df['Over'] = np.where(nfl_df['True_Total'] > nfl_df['Total'], 1, 0)
    nfl_df['Under'] = np.where(nfl_df['True_Total'] < nfl_df['Total'], 1, 0)
    nfl_df['Push'] = np.where(nfl_df['True_Total'] == nfl_df['Total'], 1, 0)

    nfl_df = nfl_df.sort_values(by=['Season', 'Week']).reset_index(drop=True)
    return nfl_df.query('Home == 1').reset_index(drop=True)
