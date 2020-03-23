import numpy as np
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
print('Số lượng lớp')
print(len(np.unique(iris_y)))
print('Số lượng data ')
print(len(iris_y))

X0 = iris_X[iris_y == 0,:]
print('5 Ví dụ thuộc lớp 0')
print(X0[:5,:])

X1 = iris_X[iris_y == 1,:]
print('5 Ví dụ thuộc lớp 1')
print(X1[:5,:])


X2 = iris_X[iris_y == 2,:]
print('5 Ví dụ thuộc lớp 2')
print(X2[:5,:])

X_train, X_test, y_train, y_test = train_test_split(
     iris_X, iris_y, test_size=50)

print('Số lượng dữ liệu dùng để huấn luyện')
print(len(y_train))
# print "Training size: %d" %len(y_train)
print('Số lượng dữ liệu dùng để test')
# print "Test size    : %d" %len(y_test)
print(len(y_test))

clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('Kết quả dự đoán của 20 dữ liệu đầu tiên trong tập test')
print('Dự đoán ')
print(y_pred[20:40])
print('Thực tế ')
print(y_test[20:40])

print("Độ chính xác")
print(100*accuracy_score(y_test, y_pred))
 

