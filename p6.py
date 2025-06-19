from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

dataset=load_iris()

x_train,x_test,y_train,y_test=train_test_split(dataset.data,dataset.target,test_size=0.25,random_state=42)

model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

print("accuracy score",accuracy_score(y_test,y_pred))
print("confusion matrix",confusion_matrix(y_test,y_pred))
print("classification rport",classification_report(y_test,y_pred))
