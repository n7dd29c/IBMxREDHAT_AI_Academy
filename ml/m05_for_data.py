from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#1. 데이터
data_list = [
    load_iris(return_X_y=True),
    load_breast_cancer(return_X_y=True),
    load_digits(return_X_y=True),
    load_wine(return_X_y=True)
    ]
data_name = ['iris', 'breast_cancer', 'digits', 'wine']

model_list = [
    LinearSVC(),
    LogisticRegression(),
    DecisionTreeClassifier(),
    RandomForestClassifier()
    ]
model_name = ['LinearSVC', 'LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier']

for i, data_list in enumerate(data_list):
    x, y = data_list
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=333
    )
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    print('')
    print(f'{data_name[i]}==================')
    
    for j, model in enumerate(model_list):
        model.fit(x_train, y_train)
        score = model.score(x_test, y_test)
        print(f'{data_name[j]} : ', score)
        
# iris==================
# iris :  0.9
# breast_cancer :  0.9666666666666667
# digits :  0.9333333333333333
# wine :  0.9666666666666667

# breast_cancer==================
# iris :  0.9736842105263158
# breast_cancer :  0.9824561403508771
# digits :  0.9035087719298246
# wine :  0.9385964912280702

# digits==================
# iris :  0.9527777777777777
# breast_cancer :  0.9638888888888889
# digits :  0.8527777777777777
# wine :  0.9722222222222222

# wine==================
# iris :  1.0
# breast_cancer :  1.0
# digits :  0.8888888888888888
# wine :  0.9722222222222222