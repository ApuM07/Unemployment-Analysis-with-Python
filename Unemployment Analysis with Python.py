#Oasis Infobytes_Data Science Internship
#Task 2 : UNEMPLOYMENT ANALYSIS WITH PYTHON
#Name of Intern : APU MANDAL
#Batch  : SEPTEMBER Phase 1 AICTE OIB-SIP 2024

import pandas as pd , numpy as np
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler , MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
lE = LabelEncoder()

import warnings
warnings.filterwarnings("ignore")

sns.set_theme()
un = pd.read_csv("UMP.csv")
print("Rate of Data Present in Percentage:")
print(un.nunique())

un.drop(columns=[" Date"],inplace=True)
print("Information of Data:")
un.info()
print("Sample data Present:")
print(un.head())

un["Region"] = lE.fit_transform(un["Region"])
un[" Frequency"] = lE.fit_transform(un[" Frequency"])
un["Area"] = lE.fit_transform(un["Area"])

print("Sample data Present after Transformation:")
print(un.head())

correlation_mat = un.corr()
print(correlation_mat)

sns.heatmap(correlation_mat,annot=True,linewidths=.5,cmap="twilight")
plt.show()

X = un.drop(columns=[" Estimated Unemployment Rate (%)"])
Y = un[" Estimated Unemployment Rate (%)"]
X_train , X_test , y_train , y_test = train_test_split(X,Y,test_size=0.33,random_state=0)

StSc = StandardScaler()
X_train  = StSc.fit_transform(X_train)
X_test  = StSc.fit_transform(X_test)

from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_train, y_train)
y_pred = reg.predict(X_test)

from sklearn.metrics import mean_absolute_error as mae
print("Mean Absolute Error of Model:",mae(y_test,y_pred))

from sklearn.metrics import mean_squared_error as mse
print("Mean Squared Error in Model:",mse(y_test, y_pred))

from sklearn.metrics import mean_absolute_percentage_error as mape
print("Mean Absolute Percentage Error:",mape(y_test, y_pred))
print("T H A N K   Y O U")
