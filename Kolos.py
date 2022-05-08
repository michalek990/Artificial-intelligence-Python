----------------------------------------LAB1-------------------------------------------------   
Zadanie 2
import numpy as np
import pandas as pd


data = pd.read_csv("./practice_lab_1.csv", sep=";")
col = data.columns
val = data.values

arr_1 = val[::2,:] - val[1:2,:]                         #1
arr_2 = (val - val.mean()) / val.std()                  #2
arr_3 = (val - val.mean(axis=0)) / val.std(axis=0)      #3
arr_4 = (val.std(axis=0) / val.mean(axis=0))            #4
max_variation = arr_4.max()                             #5
arr_5 = (val[:,::1] > val.mean(axis=0)).sum(axis=0)     #6
mask = (val == val.max())[:,::1].sum(axis=0) > 0        #7
arr_6 = np.array(col)[mask]
mask = (val == 0).sum(axis=0) == (val == 0).sum(axis=0).max()   #8
arr_7 = np.array(col)[mask]
mask = val[::2,:].sum(axis=0) > val[1::2,:].sum(axis=0) #9
arr_8 = np.array(col)[mask] 

Zadanie 3   
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def printPlot(x, y, title=""):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title(title)
    ax.set(xlabel="x", ylabel="y")
    
#zad 1
x = np.arange(-5, 5, 0.01)


y = np.tanh(x)
printPlot(x, y, "Podpunkt 1")


y = (math.e ** x - math.e ** (-x)) / (math.e ** x + math.e ** (-x))
printPlot(x, y, "Podpunkt 2")


y = 1/ (1 + math.e ** (-x))
printPlot(x, y, "Podpunkt 3")


y = [0 if i <= 0 else i for i in x]
printPlot(x, y, "Podpunkt 4") 


y = [i if i <= 0 else (math.e ** i - 1) for i in x]
printPlot(x, y, "Podpunk 5")

Zadanie 4   
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def printPlot(x, y, title=""):
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_title(title)
    ax.set(xlabel="x", ylabel="y")

data = pd.read_csv("practice_lab_1.csv", sep=";")
col = data.columns
val = data.values

corr_arr = data.corr()  # macierz korelacji

num_col = np.shape(val)[1]  # liczba kolumn tablicy val
for i in range(0, num_col): # generowanie wykresów punktowych 
    for j in range(0, num_col):
        printPlot(val[:,i], val[:,j], col[i] + ' od ' + col[j])
        
----------------------------------------LAB2-------------------------------------------------   
Zadanie 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def printScatterPlot(x, y, xl="", yl="", title=""):
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_title(title)
    ax.set(xlabel=xl, ylabel=yl)

data = pd.read_csv("practice_lab_2.csv", sep=";")
col = data.columns.to_list()
val = data.values

corr_arr = data.corr()  # wygenerowanie macierzy korelacji
print(corr_arr)

num_col = np.shape(val)[1] # liczba kolumn macierzy z wartościami

# wygenerowanie wykresów zależności mediany ceny mieszkań od poszczególnych wartości niezależnych
for i in range(num_col - 1):
    printScatterPlot(val[:, i], val[:,-1], col[i], col[-1])
 
 Zadanie 2
 import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

def test_regression(X, y, n):
    arr = np.zeros(n)
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
        linReg = LinearRegression()
        linReg.fit(X_train, y_train)
        y_pred = linReg.predict(X_test)
        arr[i] = mean_absolute_percentage_error(y_test, y_pred)
    return arr.mean()

data = pd.read_csv("practice_lab_2.csv", sep=";")
col = data.columns.to_list()
val = data.values

X, y = val[:,:-1], val[:,-1] # Podzielenie zbioru wartości na wejście i wyjście

print(test_regression(X, y, 100))

Zadanie 3

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

# funkcja do usuwania wartości odstających 
def remove_outliers(X_train, y_train):
    outliers = np.abs((y_train - y_train.mean()) / y_train.std()) > 3   # wyznaczenie wartości odstajacych
    y_train_no_outliers = y_train[~outliers]    # usunięcie wartości odstających z zbioru wyjściowego
    X_train_no_outliers = X_train[~outliers,:]  # usunięcie wartości odstających z zbioru wejściowego
    return X_train_no_outliers, y_train_no_outliers

# funkcja do zastępowania warotści odstających
def replace_outliers_by_mean(y_train):
    outliers = np.abs((y_train - y_train.mean()) / y_train.std()) > 3
    y_train_mean = y_train.copy()
    y_train_mean[outliers] = y_train.mean()
    return y_train_mean

def test_regression(X, y, n):
    res = np.zeros(n)
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
        linReg = LinearRegression()
        X_train, y_train = remove_outliers(X_train, y_train)
        linReg.fit(X_train, y_train)
        y_pred = linReg.predict(X_test)
        res[i] = mean_absolute_percentage_error(y_test, y_pred)
    return res.mean()

data = pd.read_csv('practice_lab_2.csv', sep=';') 
col = data.columns.to_list()
val = data.values

X, y = val[:,:-1], val[:,-1]
print(test_regression(X, y, 100))

Zadanie 4
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

# funkcja do zastępowania warotści odstających
def replace_outliers_by_mean(y_train):
    outliers = np.abs((y_train - y_train.mean()) / y_train.std()) > 3
    y_train_mean = y_train.copy()
    y_train_mean[outliers] = y_train.mean()
    return y_train_mean

def test_regression(X, y, n):
    res = np.zeros(n)
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
        linReg = LinearRegression()
        y_train = replace_outliers_by_mean(y_train)
        linReg.fit(X_train, y_train)
        y_pred = linReg.predict(X_test)
        res[i] = mean_absolute_percentage_error(y_test, y_pred)
    return res.mean()

data = pd.read_csv('practice_lab_2.csv', sep=';') 
col = data.columns.to_list()
val = data.values

X, y = val[:,:-1], val[:,-1]
additional_data = np.stack([X[:,4]/X[:,7], 
                            X[:,4]/X[:,5],
                            X[:,4]*X[:,3],
                            X[:,4]/X[:,-1]], axis=-1)
X = np.concatenate([X, additional_data], axis=-1)
print(test_regression(X, y, 100))

Zadanie 5
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.datasets import load_diabetes

def replace_outliers_by_mean(y_train):
    outliers = np.abs((y_train - y_train.mean()) / y_train.std()) > 3
    y_train_mean = y_train.copy()
    y_train_mean[outliers] = y_train.mean()
    return y_train_mean

def test_regression(X, y, n):
    res = np.zeros(n)
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
        linReg = LinearRegression()
        y_train = replace_outliers_by_mean(y_train)
        linReg.fit(X_train, y_train)
        y_pred = linReg.predict(X_test)
        res[i] = mean_absolute_percentage_error(y_test, y_pred)
    return res.mean()

data = load_diabetes()
df = pd.DataFrame(data.data, columns = data.feature_names)
col = df.columns.to_list() # kolumny
X = df.values   # zbiór wejściowy
y = data.target # zbiór wyjściowy

print(test_regression(X, y, 100))

----------------------------------------LAB3------------------------------------------------- 
Zadanie 2
import pandas as pd
pd.options.mode.chained_assignment = None # żeby nie pojawiały się warningi z Pandas

def qualitative_to_0_1(data, column, value_to_be_1):
    mask = data[column].values == value_to_be_1
    data[column][mask] = 1
    data[column][~mask] = 0
    return data

data = pd.read_csv("practice_lab_3.csv", sep=";")

# Przyporządkowanie binarnym wartosciom jakosciowym wartosci 0 lub 1
data = qualitative_to_0_1(data, 'Gender', 'Male')
data = qualitative_to_0_1(data, 'Married', 'Yes')
data = qualitative_to_0_1(data, 'Education', 'Graduate')
data = qualitative_to_0_1(data, 'Self_Employed', 'Yes')
data = qualitative_to_0_1(data, 'Loan_Status', 'Y')

# Przekształcenie nie binarnych wartoci jakosciowych na zbior cech o wartosciach 0 lub 1
# Utworzenie kodu 1-z-n dla wartosci jakosciowej
cat_feature = pd.Categorical(data['Property_Area'])
one_hot = pd.get_dummies(cat_feature)
# Zastapienie wartosci jakosciowej przez wygenerowany kod 1-z-n
data = pd.concat([data, one_hot], axis=1)
data = data.drop(columns=['Property_Area']

Zadanie 3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.svm import SVC as SVM
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
pd.options.mode.chained_assignment = None # żeby nie pojawiały się warningi z Pandas


TP = (0,0)  # True Positive
FN = (0,1)  # False Negative
FP = (1,0)  # False Positive
TN = (1,1)  # True Negative

def qualitative_to_0_1(data, column, value_to_be_1):
    mask = data[column].values == value_to_be_1
    data[column][mask] = 1
    data[column][~mask] = 0
    return data

def calculate_metrics(cm):
    sensivity = cm[TP] / (cm[TP] + cm[FN])
    precision = cm[TP] / (cm[TP] + cm[FP])
    specificity = cm[TN] / (cm[FP] + cm[TN])
    accuracy = (cm[TP] + cm[TN]) / (cm[TP] + cm[FN] + cm[FP] + cm[TN])
    f1 = (2 * sensivity * precision) / (sensivity + precision)
    return sensivity, precision, specificity, accuracy, f1

data = pd.read_csv("practice_lab_3.csv", sep=";")

# Przyporządkowanie binarnym wartosciom jakosciowym wartosci 0 lub 1
data = qualitative_to_0_1(data, 'Gender', 'Male')
data = qualitative_to_0_1(data, 'Married', 'Yes')
data = qualitative_to_0_1(data, 'Education', 'Graduate')
data = qualitative_to_0_1(data, 'Self_Employed', 'Yes')
data = qualitative_to_0_1(data, 'Loan_Status', 'Y')

# Przekształcenie nie binarnych wartoci jakosciowych na zbior cech o wartosciach 0 lub 1
# Utworzenie kodu 1-z-n dla wartosci jakosciowej
cat_feature = pd.Categorical(data['Property_Area'])
one_hot = pd.get_dummies(cat_feature)
# Zastapienie wartosci jakosciowej przez wygenerowany kod 1-z-n
data = pd.concat([data, one_hot], axis=1)
data = data.drop(columns=['Property_Area'])


col = list(data.columns)
X = data.drop(columns=('Loan_Status')).values.astype(float)
y = data['Loan_Status'].values.astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

models = [kNN(), SVM()]
cm_arr = []
for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm_arr.append(confusion_matrix(y_test, y_pred))

se, p, sp, acc, f1 = calculate_metrics(cm_arr[0])

Zadanie 4
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.svm import SVC as SVM
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
pd.options.mode.chained_assignment = None # żeby nie pojawiały się warningi z Pandas


TP = (0,0)  # True Positive
FN = (0,1)  # False Negative
FP = (1,0)  # False Positive
TN = (1,1)  # True Negative

def qualitative_to_0_1(data, column, value_to_be_1):
    mask = data[column].values == value_to_be_1
    data[column][mask] = 1
    data[column][~mask] = 0
    return data

def calculate_metrics(cm):
    sensivity = cm[TP] / (cm[TP] + cm[FN])
    precision = cm[TP] / (cm[TP] + cm[FP])
    specificity = cm[TN] / (cm[FP] + cm[TN])
    accuracy = (cm[TP] + cm[TN]) / (cm[TP] + cm[FN] + cm[FP] + cm[TN])
    f1 = (2 * sensivity * precision) / (sensivity + precision)
    return sensivity, precision, specificity, accuracy, f1

def printMetrics(a_name, cm, se, p, sp, acc, f1):
    print(f"{a_name}\nMacierz:\n{cm}\nSensivity: {se}\nPrecision: {p}\nSpecificity: {sp}\nAccuracy: {acc}\nf1: {f1}\n")

data = pd.read_csv("practice_lab_3.csv", sep=";")

# Przyporządkowanie binarnym wartosciom jakosciowym wartosci 0 lub 1
data = qualitative_to_0_1(data, 'Gender', 'Male')
data = qualitative_to_0_1(data, 'Married', 'Yes')
data = qualitative_to_0_1(data, 'Education', 'Graduate')
data = qualitative_to_0_1(data, 'Self_Employed', 'Yes')
data = qualitative_to_0_1(data, 'Loan_Status', 'Y')

# Przekształcenie nie binarnych wartoci jakosciowych na zbior cech o wartosciach 0 lub 1
# Utworzenie kodu 1-z-n dla wartosci jakosciowej
cat_feature = pd.Categorical(data['Property_Area'])
one_hot = pd.get_dummies(cat_feature)
# Zastapienie wartosci jakosciowej przez wygenerowany kod 1-z-n
data = pd.concat([data, one_hot], axis=1)
data = data.drop(columns=['Property_Area'])


col = list(data.columns)
X = data.drop(columns=('Loan_Status')).values.astype(float)
y = data['Loan_Status'].values.astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# Z domyslnymi parametrami
models = [kNN(), SVM()]
cm_arr = []
for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm_arr.append(confusion_matrix(y_test, y_pred))

print("Domyslne parametry")
se, p, sp, acc, f1 = calculate_metrics(cm_arr[0])
printMetrics("kNN", cm_arr[0], se, p, sp, acc, f1)

se, p, sp, acc, f1 = calculate_metrics(cm_arr[1])
printMetrics("SVM", cm_arr[1], se, p, sp, acc, f1)

models = [kNN(n_neighbors=6, weights="uniform"), SVM(kernel="rbf")]
cm_arr = []
for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm_arr.append(confusion_matrix(y_test, y_pred))

print("Z parametrami innymi niz domyslne")
se, p, sp, acc, f1 = calculate_metrics(cm_arr[0])
printMetrics("kNN", cm_arr[0], se, p, sp, acc, f1)

se, p, sp, acc, f1 = calculate_metrics(cm_arr[1])
printMetrics("SVM", cm_arr[1], se, p, sp, acc, f1)

Zadanie 5
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.svm import SVC as SVM
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
pd.options.mode.chained_assignment = None # żeby nie pojawiały się warningi z Pandas


TP = (0,0)  # True Positive
FN = (0,1)  # False Negative
FP = (1,0)  # False Positive
TN = (1,1)  # True Negative

def qualitative_to_0_1(data, column, value_to_be_1):
    mask = data[column].values == value_to_be_1
    data[column][mask] = 1
    data[column][~mask] = 0
    return data

def calculate_metrics(cm):
    sensivity = cm[TP] / (cm[TP] + cm[FN])
    precision = cm[TP] / (cm[TP] + cm[FP])
    specificity = cm[TN] / (cm[FP] + cm[TN])
    accuracy = (cm[TP] + cm[TN]) / (cm[TP] + cm[FN] + cm[FP] + cm[TN])
    f1 = (2 * sensivity * precision) / (sensivity + precision)
    return sensivity, precision, specificity, accuracy, f1

def printMetrics(a_name, cm, se, p, sp, acc, f1):
    print(f"{a_name}\nMacierz:\n{cm}\nSensivity: {se}\nPrecision: {p}\nSpecificity: {sp}\nAccuracy: {acc}\nf1: {f1}\n")

data = pd.read_csv("practice_lab_3.csv", sep=";")

# Przyporządkowanie binarnym wartosciom jakosciowym wartosci 0 lub 1
data = qualitative_to_0_1(data, 'Gender', 'Male')
data = qualitative_to_0_1(data, 'Married', 'Yes')
data = qualitative_to_0_1(data, 'Education', 'Graduate')
data = qualitative_to_0_1(data, 'Self_Employed', 'Yes')
data = qualitative_to_0_1(data, 'Loan_Status', 'Y')

# Przekształcenie nie binarnych wartoci jakosciowych na zbior cech o wartosciach 0 lub 1
# Utworzenie kodu 1-z-n dla wartosci jakosciowej
cat_feature = pd.Categorical(data['Property_Area'])
one_hot = pd.get_dummies(cat_feature)
# Zastapienie wartosci jakosciowej przez wygenerowany kod 1-z-n
data = pd.concat([data, one_hot], axis=1)
data = data.drop(columns=['Property_Area'])


col = list(data.columns)
X = data.drop(columns=('Loan_Status')).values.astype(float)
y = data['Loan_Status'].values.astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

models = [kNN(), SVM()]
cm_arr = []
for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm_arr.append(confusion_matrix(y_test, y_pred))

print("Bez skalera")
se, p, sp, acc, f1 = calculate_metrics(cm_arr[0])
printMetrics("kNN", cm_arr[0], se, p, sp, acc, f1)

se, p, sp, acc, f1 = calculate_metrics(cm_arr[1])
printMetrics("SVM", cm_arr[1], se, p, sp, acc, f1)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

models = [kNN(), SVM()]
cm_arr = []
for model in models:
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    cm_arr.append(confusion_matrix(y_test, y_pred))

print("StandardScaler")
se, p, sp, acc, f1 = calculate_metrics(cm_arr[0])
printMetrics("kNN", cm_arr[0], se, p, sp, acc, f1)

se, p, sp, acc, f1 = calculate_metrics(cm_arr[1])
printMetrics("SVM", cm_arr[1], se, p, sp, acc, f1)


scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

models = [kNN(), SVM()]
cm_arr = []
for model in models:
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    cm_arr.append(confusion_matrix(y_test, y_pred))

print("MinMaxScaler")
se, p, sp, acc, f1 = calculate_metrics(cm_arr[0])
printMetrics("kNN", cm_arr[0], se, p, sp, acc, f1)

se, p, sp, acc, f1 = calculate_metrics(cm_arr[1])
printMetrics("SVM", cm_arr[1], se, p, sp, acc, f1)


scaler = RobustScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

models = [kNN(), SVM()]
cm_arr = []
for model in models:
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    cm_arr.append(confusion_matrix(y_test, y_pred))

print("RobustScaler")
se, p, sp, acc, f1 = calculate_metrics(cm_arr[0])
printMetrics("kNN", cm_arr[0], se, p, sp, acc, f1)

se, p, sp, acc, f1 = calculate_metrics(cm_arr[1])
printMetrics("SVM", cm_arr[1], se, p, sp, acc, f1)

Zadanie 6
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.svm import SVC as SVM
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.tree import plot_tree
pd.options.mode.chained_assignment = None 

TP = (0,0)  # True Positive
FN = (0,1)  # False Negative
FP = (1,0)  # False Positive
TN = (1,1)  # True Negative

def calculate_metrics(cm):
    sensivity = cm[TP] / (cm[TP] + cm[FN])
    precision = cm[TP] / (cm[TP] + cm[FP])
    specificity = cm[TN] / (cm[FP] + cm[TN])
    accuracy = (cm[TP] + cm[TN]) / (cm[TP] + cm[FN] + cm[FP] + cm[TN])
    f1 = (2 * sensivity * precision) / (sensivity + precision)
    return sensivity, precision, specificity, accuracy, f1

def printMetrics(a_name, cm, se, p, sp, acc, f1):
    print(f"{a_name}\nMacierz:\n{cm}\nSensivity: {se}\nPrecision: {p}\nSpecificity: {sp}\nAccuracy: {acc}\nf1: {f1}\n")

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
col = list(df.columns)

X = df.values
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

dt = DT(max_depth=5)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(40,20))
tree_vis = plot_tree(dt, feature_names=col, class_names=['Y','N'], fontsize=20)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

models = [kNN(), SVM()]
cm_arr = []
for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm_arr.append(confusion_matrix(y_test, y_pred))
    
se, p, sp, acc, f1 = calculate_metrics(cm_arr[0])
printMetrics("kNN", cm_arr[0], se, p, sp, acc, f1)

se, p, sp, acc, f1 = calculate_metrics(cm_arr[1])
printMetrics("SVM", cm_arr[1], se, p, sp, acc, f1)


----------------------------------------LAB4-------------------------------------------------

Zadanie 1
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.pipeline import Pipeline

def qualitative_to_0_1(data, column, value_to_be_1):
    mask = data[column].values == value_to_be_1
    data[column][mask] = 1
    data[column][~mask] = 0
    return data

# Zbuduj wykres procentu wyjaśnionej wariancji, dobierz optymalną liczbę cech dla progu 95%
def wyjasnionaWariancja(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    pca_transform = PCA()
    pca_transform.fit(X_train)
    variances = pca_transform.explained_variance_ratio_
    cumulated_variances = variances.cumsum()
    plt.scatter(np.arange(variances.shape[0]), cumulated_variances)
    plt.yticks(np.arange(0, 1.1, 0.1))
    return (cumulated_variances < 0.95).sum()

data = pd.read_csv("voice_extracted_features.csv", sep=',')
data = qualitative_to_0_1(data, 'label', 'female')
col = list(data.columns)
val = data.values.astype(float)

X = val[:,:-1]  # Podział zbioru na podzbiory
y = val[:,-1]

print(wyjasnionaWariancja(X, y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_paced = PCA(2).fit_transform(X_train)
female = y_train == 1
fig, ax = plt.subplots(1, 1, figsize=(20, 10))
ax.scatter(X_paced[female, 0], X_paced[female, 1], label='female')
ax.scatter(X_paced[~female, 0], X_paced[~female, 1], label='male')
ax.legend()

pipe = Pipeline([['transformer', PCA(9)],
                ['scaler', StandardScaler()],
                ['classifier', kNN(weights='distance')]])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

Zadanie 2
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.svm import SVC as SVM
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

def qualitative_to_0_1(data, column, value_to_be_1):
    mask = data[column].values == value_to_be_1
    data[column][mask] = 1
    data[column][~mask] = 0
    return data

def scaling(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def meanConfusionMatrix(X, y, model, n):
    res = np.zeros((2,2))
    args = [['transformer', PCA(9)],
            ['scaler', StandardScaler()],
            ['classifier', model]]
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train, X_test = scaling(X_train, X_test)
        pipe = Pipeline(args)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        res += confusion_matrix(y_test, y_pred)
    return res / n

data = pd.read_csv("voice_extracted_features.csv", sep=',')
data = qualitative_to_0_1(data, 'label', 'female')
col = list(data.columns)
val = data.values.astype(float)

X = val[:,:-1]
y = val[:,-1]

models = [kNN(n_neighbors=6, weights='distance'), SVM(), DT()]

for model in models:
    print(meanConfusionMatrix(X, y, model, 30))
    

Zadanie 3
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.svm import SVC as SVM
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

class Components:
    def fit(self, x, y=None):
        self.pca_transform = PCA()
        self.pca_transform.fit(x)
        variances = self.pca_transform.explained_variance_ratio_
        cumulated_variances = variances.cumsum()
        PCA_num = (cumulated_variances < 0.95).sum()
        self.pca_transform = PCA(PCA_num)
        self.pca_transform.fit(x)
        return self
    
    def transform(self, x, y=None):
        return self.pca_transform.transform(x)
    
    def fit_transform(self, x, y):
        self.fit(x)
        return self.transform(x)
    
def qualitative_to_0_1(data, column, value_to_be_1):
    mask = data[column].values == value_to_be_1
    data[column][mask] = 1
    data[column][~mask] = 0
    return data    

def scaling(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def meanConfusionMatrix(X, y, model, n):
    res = np.zeros((2,2))
    args = [['transformer', Components()],
            ['scaler', StandardScaler()],
            ['classifier', model]]
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train, X_test = scaling(X_train, X_test) # Jak się nie przeskaluje to PCA_num w Components będzie się równać 0
        pipe = Pipeline(args)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        res += confusion_matrix(y_test, y_pred)
    return res / n

data = pd.read_csv("voice_extracted_features.csv", sep=',')
data = qualitative_to_0_1(data, 'label', 'female')
col = list(data.columns)
val = data.values.astype(float)

X = val[:,:-1]
y = val[:,-1]

models = [kNN(n_neighbors=6, weights='distance'), SVM(), DT()]
for model in models:
    print(meanConfusionMatrix(X, y, model, 30))
    


Zadanie 4
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.svm import SVC as SVM
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

class Outliers:
    def fit(self, x, y=None):
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        return self
    
    def transform(self, x, y=None):
        outliers = np.abs((x - self.mean) / self.std) > 3
        x_out = x.copy()
        for i in range(outliers.shape[1]):
            x_out[outliers[:,i], i] = self.mean[i]
        return x_out
    
    def fit_transform(self, x, y):
        self.fit(x)
        return self.transform(x)
    
class Components:
    def fit(self, x, y=None):
        self.pca_transform = PCA()
        self.pca_transform.fit(x)
        variances = self.pca_transform.explained_variance_ratio_
        cumulated_variances = variances.cumsum()
        PCA_num = (cumulated_variances < 0.95).sum()
        self.pca_transform = PCA(PCA_num)
        self.pca_transform.fit(x)
        return self
    
    def transform(self, x, y=None):
        return self.pca_transform.transform(x)
    
    def fit_transform(self, x, y):
        self.fit(x)
        return self.transform(x)
    
def qualitative_to_0_1(data, column, value_to_be_1):
    mask = data[column].values == value_to_be_1
    data[column][mask] = 1
    data[column][~mask] = 0
    return data    

def scaling(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def meanConfusionMatrix(X, y, model, n):
    res = np.zeros((2,2))
    args = [['outliers', Outliers()],
            ['transformer', Components()],
            ['scaler', StandardScaler()],
            ['classifier', model]]
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train, X_test = scaling(X_train, X_test) # Jak się nie przeskaluje to PCA_num w Components będzie się równać 0
        pipe = Pipeline(args)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        res += confusion_matrix(y_test, y_pred)
    return res / n

data = pd.read_csv("voice_extracted_features.csv", sep=',')
data = qualitative_to_0_1(data, 'label', 'female')
col = list(data.columns)
val = data.values.astype(float)

X = val[:,:-1]
y = val[:,-1]

models = [kNN(n_neighbors=6, weights='distance'), SVM(), DT()]
for model in models:
    print(meanConfusionMatrix(X, y, model, 30))
    
 
 
 
 
----------------------------------------LAB5-------------------------------------------------
Zadanie 2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = load_digits()
X = data.data
y = data.target
y = pd.Categorical(y)
y = pd.get_dummies(y).values
class_num = y.shape[1]

model = Sequential()
model.add(Dense(64, input_shape=(X.shape[1],), activation='relu',))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(class_num, activation='softmax'))
learning_rate = 0.0001
model.compile(optimizer=Adam(learning_rate), loss="categorical_crossentropy", metrics='accuracy')
model.summary()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model.fit(X_train, y_train, batch_size=32, epochs=500, validation_data=(X_test, y_test))

historia = model.history.history
floss_train = historia['loss']
floss_test = historia['val_loss']
acc_train = historia['accuracy']
acc_test = historia['val_accuracy']
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
epochs = np.arange(0, 500)
ax[0].plot(epochs, floss_train, label='floss_train')
ax[0].plot(epochs, floss_test, label='floss_test')
ax[0].set_title("Funkcja start")
ax[0].legend()
ax[1].set_title("Dokladnosc")
ax[1].plot(epochs, acc_train, label='acc_train')
ax[1].plot(epochs, acc_test, label='acc_test')
ax[1].legend()


Zadanie 3
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

def createModel(layers_num, neurons_num, activation_func, loss_func, metrics_func, learning_rate, class_num, input_shape):
    model = Sequential()
    if layers_num == 1:
        model.add(Dense(class_num, input_shape=(input_shape,), activation=activation_func))
    else:
        model.add(Dense(neurons_num, input_shape=(input_shape,), activation=activation_func))
        for i in range(layers_num - 2):
            model.add(Dense(neurons_num, activation=activation_func))
        model.add(Dense(class_num, activation=activation_func))
    model.compile(optimizer=Adam(learning_rate), loss=loss_func, metrics=metrics_func)
    return model

def crossValidation(X, y, model, epochs_num):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    accs=[]
    scaler = StandardScaler()
    for train_index, test_index in KFold(5).split(X_train):
        X_train_cv = X_train[train_index,:]
        X_test_cv = X_train[test_index,:]
        y_train_cv = y_train[train_index,:]
        y_test_cv = y_train[test_index,:]
        X_train_cv = scaler.fit_transform(X_train_cv)
        X_test_cv = scaler.transform(X_test_cv)
        model.fit(X_train_cv, y_train_cv, batch_size=32, epochs=epochs_num, validation_data=(X_test_cv, y_test_cv), verbose=2)
        y_pred = model.predict(X_test_cv).argmax(axis=1)
        y_test_cv = y_test_cv.argmax(axis=1)
        accs.append(accuracy_score(y_test_cv, y_pred))
    return accs
        

data = load_digits()
y = data.target
X = data.data
y = pd.Categorical(y)
y = pd.get_dummies(y).values

class_num = y.shape[1]
input_shape = X.shape[1]
layers_num = [1, 2, 3, 4, 5]
neurons_num = [32, 64, 100]
activation_func =['relu', 'softmax', 'tanh']
learningRate = [0.1, 0.001, 0.0001, 0.00001]
epoch_num = [100, 200, 300]

for layer in layers_num:
    for neurons in neurons_num:
        for activation in activation_func:
            for rate in learningRate:
                for epoch in epoch_num:
                    model = createModel(layer, neurons, activation, 'categorical_crossentropy', 'accuracy', rate, class_num, input_shape)
                    accs = crossValidation(X, y, model, epoch)
    

for acc in accs:
    print(acc)

----------------------------------------LAB6-------------------------------------------------

Zadanie 1
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from keras.regularizers import l2
from matplotlib import pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def clasiffy(X, y, epoch_num, model):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, )
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)
  model.fit(X_train, y_train, batch_size=32, epochs=epoch_num, validation_data=(X_test, y_test), verbose=0)
  return model

def createModel(activation_func, regularication_val, class_num, neurons_num, input_shape, n):
  model = Sequential()
  model.add(Dense(neurons_num, input_shape=(input_shape, ), activation=activation_func, kernel_regularizer=l2(regularication_val)))
  for i in range(n):
    model.add(Dense(neurons_num, activation=activation_func, kernel_regularizer=l2(regularication_val)))
  model.add(Dense(class_num, activation='softmax', kernel_regularizer=l2(regularication_val)))
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
  model.summary()
  return model

def drawPlot(accsResult, params, plot_title=''):
    accsResultArray = np.array(accsResult)
    fig,ax = plt.subplots(1, 1, figsize=(10, 10))
    x = np.arange(1, len(params) + 1)
    ax.scatter(x, np.mean(accsResultArray, axis=1))
    ax.set_xticks(x)
    ax.set_xticklabels(params)
    ax.set_title(plot_title)

data = load_iris()
X = data.data
y = data.target
y = pd.Categorical(y)
y = pd.get_dummies(y).values
class_num = y.shape[1]

# Tak były te wartości zdefiniowane u Karczmarka ale z tablic layer_num, neurons_num i epoch_num 
# zawsze jest brana tylko jedna wartość wiec można by to było zamienić na zwykłe zmienne.
# Jedynie regularication_rates musi zostać tablicą.
layers_num = [1,2,3,4,5]
neurons_num = [32,64,100]
activation_func = 'relu'
regularication_rates = np.array([0, 0.0001, 0.001, 0.01, 0.1])
epoch_num = np.array([30, 200, 300])
input_shape = X.shape[1]
repeatCount = 2

accs = []
accsResult = []
for regRate in regularication_rates:
  accs = []
  for i in range(repeatCount):
    model = createModel(activation_func, regRate, class_num, neurons_num[0], input_shape, layers_num[3])
    model = clasiffy(X, y, epoch_num[0], model)
    acc = max(model.history.history['val_accuracy'])
    accs.append(acc)
  accsResult.append(accs)
drawPlot(accsResult, regularication_rates, 'Zaleznosc sredniej dokladnosci od współczynnika regularyzacji')

Zadanie 2
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def clasiffy(X, y, epoch_num, model):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, )
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)
  model.fit(X_train, y_train, batch_size=32, epochs=epoch_num, validation_data=(X_test, y_test), verbose=0)
  return model

def createModel(activation_func, rate, class_num, neuron_num, input_shape, layer_num):
  model = Sequential()
  model.add(Dense(neuron_num, input_shape=(input_shape, ), activation=activation_func))
  for i in range(4):
    model.add(Dense(neuron_num, activation=activation_func))
    model.add(Dropout(rate))
  model.add(Dense(class_num, activation='softmax'))
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
  model.summary()
  return model

def drawPlot(accsResult, params, plot_title=''):
    accsResultArray = np.array(accsResult)
    fig,ax = plt.subplots(1, 1, figsize=(10, 10))
    x = np.arange(1, len(params) + 1)
    ax.scatter(x, np.mean(accsResultArray, axis=1))
    ax.set_xticks(x)
    ax.set_xticklabels(params)
    ax.set_title(plot_title)

data = load_iris()
X = data.data
y = data.target
y = pd.Categorical(y)
y = pd.get_dummies(y).values
class_num = y.shape[1]

# Tak były te wartości zdefiniowane u Karczmarka ale z tablic layer_num, neurons_num i epoch_num 
# zawsze jest brana tylko jedna wartość wiec można by to było zamienić na zwykłe zmienne.
# Jedynie do_rate musi zostać tablicą.
layers_num = [1,2,3,4,5]
neurons_num = [32,64,100]
activation_func = 'relu'
do_rate = [0,0.2,0.3,0.5]
epoch_num = np.array([30, 200, 300])
input_shape = X.shape[1]
repeatCount = 2

accs = []
accsResult = []
for rate in do_rate:
  accs = []
  for i in range(repeatCount):
    model = createModel(activation_func, rate, class_num, neurons_num[0], input_shape, layers_num[0])
    model = clasiffy(X, y, epoch_num[0], model)
    acc = max(model.history.history['val_accuracy'])
    accs.append(acc)
  accsResult.append(accs)
drawPlot(accsResult, do_rate, 'Zaleznosc sredniej dokladnosci od parametru do_rate')

Zadanie 3
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from keras.layers import Dense
from keras.layers import GaussianNoise
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def clasiffy(X, y, epoch_num, model):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, )
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)
  model.fit(X_train, y_train, batch_size=32, epochs=epoch_num, validation_data=(X_test, y_test), verbose=0)
  return model

def createModel(activation_func, noise, class_num, neuron_num, input_shape, layer_num):
  model = Sequential()
  model.add(Dense(neuron_num, input_shape=(input_shape, ), activation=activation_func))
  for i in range(layer_num):
    model.add(Dense(neuron_num, activation=activation_func))
    model.add(GaussianNoise(noise))
  model.add(Dense(class_num, activation="softmax"))
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
  model.summary()
  return model

def drawPlot(accsResult, params, plot_title=''):
    accsResultArray = np.array(accsResult)
    fig,ax = plt.subplots(1, 1, figsize=(10, 10))
    x = np.arange(1, len(params) + 1)
    ax.scatter(x, np.mean(accsResultArray, axis=1))
    ax.set_xticks(x)
    ax.set_xticklabels(params)
    ax.set_title(plot_title)
    
data = load_iris()
X = data.data
y = data.target
y = pd.Categorical(y)
y = pd.get_dummies(y).values
class_num = y.shape[1]

# Tak były te wartości zdefiniowane u Karczmarka ale z tablic layer_num, neurons_num i epoch_num 
# zawsze jest brana tylko jedna wartość wiec można by to było zamienić na zwykłe zmienne.
# Jedynie noises musi zostać tablicą.
layers_num = [1,2,3,4,5]
neurons_num = [32,64,100]
activation_func = 'relu'
noises = [0,0.1,0.2,0.3]
epoch_num = np.array([30, 200, 300])
input_shape = X.shape[1]
repeatCount = 2

accsResult = []
accs = []
for noise in noises:
  acc = []
  for i in range(repeatCount):
    model = createModel(activation_func, noise, class_num, neurons_num[0], input_shape, layers_num[0])
    model = clasiffy(X, y, epoch_num[0], model)
    acc = max(model.history.history['val_accuracy'])
    accs.append(acc)
  accsResult.append(accs)
drawPlot(accsResult, noises, 'Zaleznosc sredniej dokladnosci od parametru noise')

----------------------------------------LAB7-------------------------------------------------

Zadanie 2

from keras.layers import Conv2D, Flatten, Dense
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.datasets import mnist
import numpy as np

def createModel(X_train, class_cnt, filter_cnt, neuron_cnt, learning_rate, act_func, kernel_size, layer_cnt):
  model = Sequential()
  conv_rule = 'same'
  for i in range(layer_cnt):
    model.add(Conv2D(filters=filter_cnt, kernel_size=kernel_size, padding=conv_rule, activation=act_func, input_shape=X_train.shape[1:]))
  model.add(Flatten())
  model.add(Dense(class_cnt, activation='softmax'))
  return model

train, test = mnist.load_data()

X_train, y_train = train[0], train[1]
X_test, y_test = test[0], test[1]
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

class_cnt = np.unique(y_train).shape[0]
filter_cnt = 32
neuron_cnt = 32
learning_rate = 0.0001
act_func = 'relu'
kernel_size = (3, 3)
layer_cnt = 3
nrEpoch = 10

accs = []

for i in range(1, 5):
  model = createModel(X_train, class_cnt, filter_cnt, neuron_cnt, learning_rate, act_func, kernel_size, i)
  model.compile(optimizer=Adam(learning_rate), loss='SparseCategoricalCrossentropy', metrics='accuracy')
  hist = model.fit(X_train, y_train, epochs=nrEpoch, validation_data=(X_test, y_test), verbose=0)
  accs.append(max(hist.history['val_accuracy']))
  
for i in range(1, 5):
    print(f"Dla {i} warstw ukrytych:")
    print(accs[i - 1])
    
Zadanie 3
from keras.layers import Conv2D, Flatten, Dense, AveragePooling2D, MaxPooling2D
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.datasets import mnist
import numpy as np
from matplotlib import pyplot as plt

def createModel(X_train, class_cnt, filter_cnt, act_func, kernel_size, layer_cnt, pooling_cnt, pooling_method):
  model = Sequential()
  conv_rule = 'same'
  for i in range(layer_cnt):
    model.add(Conv2D(filters=filter_cnt, kernel_size=kernel_size, padding=conv_rule, activation=act_func, input_shape=X_train.shape[1:]))
    model.add(pooling_method(pooling_cnt))
  model.add(Flatten())
  model.add(Dense(class_cnt, activation='softmax'))
  return model

def createChart(model, nrEpoch):
  historia = model.history.history
  floss_train = historia['loss']
  floss_test = historia['val_loss']
  acc_train = historia['accuracy']
  acc_test = historia['val_accuracy']
  fig,ax = plt.subplots(1, 2, figsize=(20, 20))
  epoches = np.arange(0, nrEpoch)
  ax[0].plot(epoches, floss_train, label='floss_train')
  ax[0].plot(epoches, floss_test, label='floss_test')
  ax[0].set_title('Funkcje strat')
  ax[0].legend()
  ax[1].plot(epoches, acc_train, label='acc_train')
  ax[1].plot(epoches, acc_test, label='acc_test')
  ax[1].set_title('Dokladnosc')
  ax[1].legend()


train, test = mnist.load_data()

X_train, y_train = train[0], train[1]
X_test, y_test = test[0], test[1]
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

class_cnt = np.unique(y_train).shape[0]
filter_cnt = 32
neuron_cnt = 32
learning_rate = 0.0001
act_func = 'relu'
kernel_size = (3, 3)
pooling_size = (2, 2)
layer_cnt = 3
nrEpoch = 10

pooling_layers = [MaxPooling2D, AveragePooling2D]
accs = []
for pooling_layer in pooling_layers:
  model = createModel(X_train, class_cnt, filter_cnt, act_func, kernel_size, layer_cnt, pooling_size, pooling_layer)
  model.compile(optimizer=Adam(learning_rate), loss='SparseCategoricalCrossentropy', metrics='accuracy')
  hist = model.fit(X_train, y_train, epochs=nrEpoch, validation_data=(X_test, y_test), verbose=2)
  accs.append(max(hist.history['val_accuracy']))
  createChart(model, nrEpoch)
print(accs)

----------------------------------------LAB8-------------------------------------------------

Zadanie 1
import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.layers import Input, Reshape, BatchNormalization, Dense, AveragePooling2D, Average
from keras.models import Model
from keras.utils.vis_utils import plot_model

def add_inseption_module(input_tensor):
    act_func = 'relu'
    paths = [
        [Dense(512, activation='softmax'), 
         Dense(128, activation='softmax'), 
         Dense(64, activation='softmax'),
         Dense(16, activation='softmax'),
         Dense(10, activation='softmax')],
        [Dense(512, activation='softmax'),
         Dense(64, activation='softmax'),
         Dense(10, activation='softmax')],
        [Dense(512, activation='softmax'),
         Dense(64, activation='softmax'),
         Dense(10, activation='softmax')],
        [Dense(512, activation='softmax'),
         Dense(64, activation='softmax'),
         Dense(10, activation='softmax')],
        [Dense(512, activation='softmax'),
         Dense(64, activation='softmax'),
         Dense(10, activation='softmax')]
        ]
    
    for_avg = []
    for path in paths:
        output_tensor = input_tensor
        for layer in path:
            output_tensor = layer(output_tensor)
        for_avg.append(output_tensor)
    return Average() (for_avg)

data = mnist.load_data()
X_train, y_train = data[0][0], data[0][1]
X_test, y_test = data[1][0], data[1][1]
y_train = pd.get_dummies(pd.Categorical(y_train)).values
y_test = pd.get_dummies(pd.Categorical(y_test)).values

class_cnt = y_train.shape[1]


output_tensor = input_tensor = Input(X_train.shape[1:])
output_tensor = Reshape((784,)) (output_tensor)
output_tensor = BatchNormalization() (output_tensor)
output_tensor = add_inseption_module(output_tensor)
ANN = Model(inputs = input_tensor, outputs=output_tensor)
ANN.compile(loss='categorialc_crossentropy', metrics='accuracy', optimizer='adam')
plot_model(ANN, show_shapes=True)

Zadanie 2
import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.layers import Conv2D, BatchNormalization, Input, Add, Activation, GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.utils.vis_utils import plot_model

def ResNet(input_tensor):
    path = [Conv2D(filters=input_tensor.shape[-1], kernel_size=(3,3), padding='same', activation='relu'),
            BatchNormalization()]
    output_tensor = input_tensor
    for layer in path:
        output_tensor = layer(output_tensor)
    for_out = [output_tensor, input_tensor]
    output_tensor = Add() (for_out)
    return Activation('relu') (output_tensor)
    

data = mnist.load_data()
X_train, y_train = data[0][0], data[0][1]
X_test, y_test = data[1][0], data[1][1]
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)
y_train = pd.get_dummies(pd.Categorical(y_train)).values
y_test = pd.get_dummies(pd.Categorical(y_test)).values

class_cnt = y_train.shape[1]

output_tensor = input_tensor = Input(X_train.shape[1:])
output_tensor = Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu',) (output_tensor)
for _ in range(5):
    output_tensor = ResNet(output_tensor)    
output_tensor = GlobalAveragePooling2D() (output_tensor)
output_tensor = Dense(class_cnt, activation='softmax') (output_tensor)
ANN = Model(inputs=input_tensor, outputs=output_tensor)
ANN.compile(loss='categorical_crossentropy', metrics='accuracy', optimizer='adam')
plot_model(ANN, show_shapes=True)

Zadanie 3
import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.layers import Conv2D, BatchNormalization, concatenate, Input, GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.utils.vis_utils import plot_model

def DenseNet(input_tensor):
    path = [Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'),
            BatchNormalization()]
    
    output_tensor = input_tensor
    for layer in path:
        output_tensor = layer(output_tensor)
    for_conc = [output_tensor, input_tensor]
    output_tensor = concatenate(for_conc)
    output_tensor = Conv2D(filters=64, kernel_size=(1,1), padding='same', activation='relu') (output_tensor)
    return output_tensor


data = mnist.load_data()
X_train, y_train = data[0][0], data[0][1]
X_test, y_test = data[1][0], data[1][1]
X_train = np.expand_dims(X_train, axis = -1)
X_test = np.expand_dims(X_test, axis = -1)
y_train = pd.get_dummies(pd.Categorical(y_train)).values
y_test = pd.get_dummies(pd.Categorical(y_test)).values

class_cnt = y_train.shape[1]

output_tensor = input_tensor = Input(X_train.shape[1:])
for i in range(5):
    output_tensor = DenseNet(output_tensor)
output_tensor = GlobalAveragePooling2D() (output_tensor)
output_tensor = Dense(class_cnt, activation='softmax') (output_tensor)

ANN = Model(inputs=input_tensor, outputs=output_tensor)
ANN.compile(loss='categorical_crossentropy', metrics='accuracy', optimizer='adam')
plot_model(ANN, show_shapes=True)

Zadanie 4
import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.layers import Conv2D, BatchNormalization, GlobalAveragePooling2D, Input, Lambda, Conv2D, Dense, Add
from keras.models import Model
from keras.utils.vis_utils import plot_model
import tensorflow as tf

def Mish(tensor):
    tensor = tensor + tf.keras.activations.tanh(tf.math.log(1 + tf.math.exp(tensor)))
    return tensor


def add_resnet_module_with_mish(input_tensor):
    paths = [[Conv2D(filters=input_tensor.shape[-1], kernel_size=(3,3), padding='same', activation=Mish),
              BatchNormalization()],
             []]
    for_concat = []
    for path in paths:
        output_tensor = input_tensor
        for layer in path:
            output_tensor = layer(output_tensor)
        for_concat.append(output_tensor)
    output_tensor = Add()(for_concat)
    output_tensor = Lambda(Mish)(output_tensor)
    return output_tensor

data = mnist.load_data()
X_train, y_train = data[0][0], data[0][1]
X_test, y_test = data[1][0], data[1][1]
X_train = np.expand_dims(X_train, axis = -1)
X_test = np.expand_dims(X_test, axis = -1)
y_train = pd.get_dummies(pd.Categorical(y_train)).values
y_test = pd.get_dummies(pd.Categorical(y_test)).values

output_tensor = input_tensor = Input(X_train.shape[1:])
output_tensor = Conv2D(filters=32, kernel_size=(3,3), padding='same', activation=Mish)(output_tensor)
for i in range(2):
    output_tensor = add_resnet_module_with_mish(output_tensor)
    
output_tensor = GlobalAveragePooling2D()(output_tensor)
ANN = Model(inputs = input_tensor, outputs = output_tensor)
ANN.compile(loss='categorical_crossentropy', metrics='accuracy', optimizer='adam')

plot_model(ANN, show_shapes=True)
