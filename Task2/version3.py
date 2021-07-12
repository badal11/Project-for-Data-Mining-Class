from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from mpl_toolkits import mplot3d
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
import csv

regr = linear_model.LinearRegression()

datan = ['Hour', 'Temperature', 'Humidity', 'Atmospheric Pressure', 'Soil Temperature', 'Solar Radiation']
data = pd.read_csv("hr_rad_temp_pr_hu_drt-10.csv", names = datan)

x = data[['Hour', 'Temperature', 'Atmospheric Pressure', 'Soil Temperature']].values

print(x[0][0])
for i in range(len(x)):
	x[i][0] = int(x[i][0][11]+x[i][0][12])
print(x)

y = data['Solar Radiation'].to_numpy().reshape(-1, 1)

#for j in range(len(y)):
#	y[j] = y[j]*1000

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

regr.fit(x_train, y_train)

y_pred_test = regr.predict(x_test)

# Model evaluation
print("Root mean squared error = %.4f" % np.sqrt(mean_squared_error(y_test, y_pred_test)))
print('R-squared = %.4f' % r2_score(y_test, y_pred_test))

'''
f_testn = ['Hour', 'Solar Radiation', 'Temperature']
f_testdata = pd.read_csv("testing_answer1.csv", names = f_testn)

fx = f_testdata[['Hour', 'Temperature']].values
fy = f_testdata['Solar Radiation'].to_numpy().reshape(-1, 1)

for i in range(len(fy)):
	fy[i] = fy[i]*1000

x = data[['Hour', 'Temperature', 'Atmospheric Pressure', 'Soil Temperature']].values
y = data['Solar Radiation'].to_numpy().reshape(-1, 1)
z = data['Atmospheric Pressure'].to_numpy().reshape(-1, 1)
a = data['Humidity']
b = data['Soil Temperature']

for j in range(len(y)):
	y[j] = y[j]*1000



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)


#print(x_train.shape)
#print(y)


regr.fit(x_train, y_train)

y_pred_test = regr.predict(x_test)
#f_pred_test = regr.predict(fx)



plt.scatter(y_test, y_pred_test, color='black')
plt.title('Comparing true and predicted values for test set')
plt.xlabel('True values for y')
plt.ylabel('Predicted values for y')
plt.show()



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
'''