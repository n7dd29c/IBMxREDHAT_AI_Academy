x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [0,1,1,0]

# from keras.models import Sequential
# from keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

# model = SVC()
model = DecisionTreeClassifier()

model.fit(x_data, y_data)

y_pred = model.predict(x_data, y_data)
results = model.evaluate(x_data, y_data)
print(results)
acc = accuracy_score(y_data, y_pred)
print(acc)