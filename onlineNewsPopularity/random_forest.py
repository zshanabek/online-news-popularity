import numpy as np
import pandas as pd
df = pd.read_csv('OnlineNewsPopularity.data')

df.drop(['url'], 1, inplace=True)

print(df)



# y = np.array(df['n_tokens_content'])
