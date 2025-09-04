import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

data_file = '../data/Data_Input.xlsx'
data = {
    'V_S': pd.read_excel(data_file, sheet_name='V_S&alpha_S', engine='openpyxl'),
    'V_wind': pd.read_excel(data_file, sheet_name='V_Wind', engine='openpyxl'),
    'alpha_wind': pd.read_excel(data_file, sheet_name='alpha_wind', engine='openpyxl')
} 

df = data['V_wind']


fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111, projection='3d')

X = df['S(nm)'].values
Y = [int(col.replace('t=','').replace('h','')) for col in df.columns[1:]]
X, Y = np.meshgrid(X, Y)
Z = df.iloc[:,1:].values.T

surf = ax.plot_surface(X, Y, Z, cmap='viridis', 
                      rstride=1, cstride=1,
                      linewidth=0, antialiased=True)

ax.view_init(30, 45)
ax.set_xlabel('Position (nm)')
ax.set_ylabel('Time (hours)')
ax.set_zlabel('Wind Speed')
fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.show()

g = sns.FacetGrid(df.melt(id_vars='S(nm)', 
                         var_name='Time', 
                         value_name='Wind'), 
                 col='Time', 
                 col_wrap=6,
                 height=2, aspect=1.2)
g.map(sns.lineplot, 'S(nm)', 'Wind', 
      color='steelblue', ci=None)
g.set_titles("Time: {col_name}")
g.set_axis_labels("Position (nm)", "Wind Speed")
plt.tight_layout() 

plt.show()
#        S(nm)      t=2h      t=4h      t=6h      t=8h     t=10h     t=12h  ...     t=68h     t=70h     t=72h     t=74h     t=76h     t=78h     t=80h
# V_S
# plt.figure()
# plt.plot(data['V_S']['V_S'], data['V_S']['alpha_S'])
