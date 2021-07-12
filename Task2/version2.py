from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from mpl_toolkits import mplot3d
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
import csv

regr = linear_model.LinearRegression()

datan = ['Hour', 'Solar Radiation', 'Temperature', 'Atmospheric Pressure', 'Humidity', 'Soil Temperature']
data = pd.read_csv("hr_rad_temp_pr_hu_drt.csv", names = datan)

f_testn = ['Hour', 'Solar Radiation', 'Temperature']
f_testdata = pd.read_csv("testing_answer1.csv", names = f_testn)

fx = f_testdata[['Hour', 'Temperature']].values
fy = f_testdata['Solar Radiation'].to_numpy().reshape(-1, 1)

for i in range(len(fy)):
	fy[i] = fy[i]*1000

x = data[['Hour', 'Temperature']].values
y = data['Solar Radiation'].to_numpy().reshape(-1, 1)
z = data['Atmospheric Pressure'].to_numpy().reshape(-1, 1)
a = data['Humidity']
b = data['Soil Temperature']

for j in range(len(y)):
	y[j] = y[j]*1000

x_train = x[:-50]
y_train = y[:-50]


x_test = x[-50:]
y_test = y[-50:]


#print(x_train.shape)
#print(y)


regr.fit(x_train, y_train)

y_pred_test = regr.predict(x_test)
f_pred_test = regr.predict(fx)


'''
plt.scatter(y_test, y_pred_test, color='black')
plt.title('Comparing true and predicted values for test set')
plt.xlabel('True values for y')
plt.ylabel('Predicted values for y')
plt.show()
'''


# Model evaluation
print("Root mean squared error = %.4f" % np.sqrt(mean_squared_error(y_test, y_pred_test)))
print('R-squared = %.4f' % r2_score(y_test, y_pred_test))

print("Results for labor day prediction :\n")
print("Root mean squared error = %.4f" % np.sqrt(mean_squared_error(fy, f_pred_test)))
print('R-squared = %.4f' % r2_score(fy, f_pred_test))

with open('employee_file.csv', mode='w') as employee_file:
    employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    employee_writer.writerow(fy)
    employee_writer.writerow(f_pred_test)
