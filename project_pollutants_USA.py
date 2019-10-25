import math
import numpy as np
import pandas as pd
#import plotly as plt
import matplotlib.pyplot as plt
import random
import re


#----------------------Reading data and intial clean up---------------------------

df = pd.read_csv('pollution_us_2000_2016.csv')
gdp = pd.read_csv('qgsp_all_C.csv')

df = df.dropna(0)
gdp = gdp.dropna(0)

# ---------------- Removing unnecessary info and summing data according to states -------------------

df_1 = df[['State','Date Local','NO2 Mean','O3 Mean','SO2 Mean', 'CO Mean']].copy()
df_1 = df_1[(df_1['Date Local'].str.contains('2016', regex= True, na=False))]
df_1 = df_1.groupby('State').sum().reset_index()


gdp_1 = gdp[['GeoName', '2016Q1', '2016Q2', '2016Q3', '2016Q4']].copy()
gdp_1 = gdp_1.convert_objects(convert_numeric=True)
gdp_1 = gdp_1.groupby('GeoName').sum()
list_of_columns = ['2016Q1','2016Q2', '2016Q3', '2016Q4']
gdp_1['Sum']=gdp_1.sum(axis=1)
gdp_1= gdp_1.reset_index()


#------------------Connecting two tables together ----------------------------

gdp_2 = gdp_1[['Sum']].copy()

df_comp = pd.concat([df_1, gdp_2], axis=1, join ='inner')#.reindex(df_.index)
df_comp = df_comp.dropna(0)
df_comp.to_csv("Data_summed.csv")
print (df_comp.tail())

print (df_comp.dtypes)

#--------------------Plotting data-------------------

x = df_comp['State'].tolist()
print (x)
y=df_comp['NO2 Mean']
y_pos = np.arange(len(df_comp.index))
values = df_comp['State']

plt.bar(y_pos, y)  
#ax.title('accumalated NO2 emission per State')  
plt.xticks(y_pos, x, rotation='vertical')
plt.ylabel('accumalated NO2 Mean')
#plt.legend('y=x')
plt.savefig('together.png')
plt.show()


# ---------------- Highest state, compare emission hours -------------------



# df.plot(x='availability_365', y='number_of_reviews', style='o')  
# plt.title('availability_365 vs # of reviews')  
# plt.xlabel('availability_365')  
# plt.ylabel('# of reviews')   
# plt.show()



#minimum nights and # of reviews have seemingly a negative correlation
#