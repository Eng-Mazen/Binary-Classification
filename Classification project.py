import numpy as np
import pandas as pd
import sklearn.linear_model as linear
while True:
    print("The file accepts onlt text files")
    path = input("Enter the path using(Ctrl+Shift+C) on the file:\n")
    data = pd.read_csv(rf"{path[1 : len(path) - 1]}")
    columns_num = len(data.columns)
    X = data.iloc[:, 0 : columns_num -1]
    y = data.iloc[:, columns_num -1]
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    Logistic_regression_model = linear.LogisticRegression(solver='liblinear', random_state=0).fit(X, y)
    theta_0 = Logistic_regression_model.intercept_
    theta_1 = Logistic_regression_model.coef_
    print('theta 0 =', theta_0)
    print("theta 1 =", theta_1)
    print('The predicted values are:', Logistic_regression_model.predict(X))
    print(f'The error percentage is equal to: {1 - Logistic_regression_model.score(X, y)}')
    print("Enter the value you want to predict:")
    prediction = np.zeros(columns_num - 1).reshape(1,columns_num - 1)
    for i in range(prediction.shape[1]):
        k = float(input(f"Enter the value number {i + 1}: "))
        prediction[0, i] = k 
    print(f'The result value will be {Logistic_regression_model.predict(prediction)[0]}')
    print('Want to start again?:')
    choice = input()
    if 'y' in choice:
        continue
    else:
        break
print('program has ended')