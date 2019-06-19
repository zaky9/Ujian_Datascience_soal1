import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

df = pd.read_excel('indo_12_1.xls',
    header=3, 
    index_col =0,
    nrows=34,
    na_values=['-'])
# print(df)

#  populasi paling rendah di 1971
ledmin=  df[df[1971]==df[1971].min()].index.values[0] # Buat legend (ledmin)
min1971 = df[df[1971]==df[1971].min()].values[0]

# Populasi paling banyak di 2010
ledmax=  df[df[2010]==df[2010].iloc[0:33].max()].index.values[0]  # Buat legend (ledmax)
max2010 = df[df[2010]==df[2010].iloc[0:33].max()].values[0]

# Total Populasi Indonesia
ledsum=  df[df[1971]==df[1971].max()].index.values[0]  # Buat legend (ledpop)
sumpop = df[df[1971]==df[1971].max()].values[0]

x = np.array(df.columns)


# ----- Machine Learning ---------- #
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

# defining model 1,2 & 3
modelmax2010 = LinearRegression()         # model buat 1971
modelmin1971 = LinearRegression()         # model buat 2010
modelindo = LinearRegression()         # model buat total populasi

# training data .fit(data independent[2D], data dependent [1D])
xaxis = x.reshape(-1,1) # reshape x from 1D to 2D

# -- 1971 populasi (bmin1971) --
modelmin1971.fit(xaxis,min1971)
# prediksi 2050
ban = modelmin1971.predict([[2050]])
print('Jumlah penduduk di banten tahun 2050 =', ban)
# Accuracy
print('R^2 = ', round(modelmin1971.score(xaxis,min1971)*100,2),'%')
# best fit line buat prediksi
bmin1971 = modelmin1971.predict(xaxis)  


# -- 2010 populasi (bmax2010) --
modelmax2010.fit(xaxis,max2010)
# prediksi 2050
jabar = modelmax2010.predict([[2050]])
print('Jumlah penduduk di jabar tahun 2050 =', jabar)
# Accuracy
print('R^2 = ', round(modelmax2010.score(xaxis,max2010)*100,2),'%')
# best fit line buat prediksi
bmax2010 = modelmax2010.predict(xaxis)


# -- Polulasi Indonesia (bindo) --
modelindo.fit(xaxis,sumpop)
# prediksi 2050
indo = modelindo.predict([[2050]])
print('Jumlah penduduk di indo tahun 2050 =',indo)
# Accuracy
print('R^2 = ', round(modelindo.score(xaxis,sumpop)*100,2),'%')
# best fit line buat prediksi
bindo = modelindo.predict(xaxis)

# Best fit line
plt.plot(xaxis, bmin1971, 'k-', marker='o', label = ledmin)
plt.plot(xaxis, bmax2010, 'k-', marker='o', label = ledmax)
plt.plot(xaxis, bindo, 'k-', marker='o', label = ledsum)

plt.plot(x,min1971, 'r-', marker = 'o', label = ledmin)
plt.plot(x,max2010, 'b-', marker = 'o', label = ledmax)
plt.plot(x,sumpop, 'g-', marker = 'o', label = ledsum)


# print(type(l1))
plt.legend(loc='upper left')
plt.grid(True)
plt.show()
