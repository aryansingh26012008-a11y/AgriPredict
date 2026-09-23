import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate
import pickle
## ENCODING

crop = pd.read_csv('Crop_recommendation.csv')
cat_labels = crop['label'].astype('category')

mapping = dict(enumerate(cat_labels.astype('category').cat.categories))
map_flipped = {v: k+1 for k,v in mapping.items()}

print(map_flipped)


X = crop.drop('label', axis=1)
y = crop['label']



# 2. Split into Train/Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

model = Pipeline([
    ('scaler', MinMaxScaler()),
    ('classifier', RandomForestClassifier(n_estimators=300,random_state=42,n_jobs=-1)),
])


# 3. Train Random Forest
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    scores=cross_validate(model,X,y,cv=cv,scoring=['accuracy','f1_macro'],return_train_score=True,n_jobs=-1)
    return {
        'model':model,'X_train':X_train,'X_test':X_test,'y_test':y_test,
        'train_acc':accuracy_score(y_train,model.predict(X_train)),
        'test_acc':accuracy_score(y_test,model.predict(X_test)),
        'test_pred':model.predict(X_test),'cv':scores
    }


def recommendation(N,P,K,temperature,humidity,ph,rainfall,):
    features = pd.DataFrame([[
        N, P, K, temperature, humidity, ph, rainfall
    ]], columns=X.columns)
    prediction = model.predict(features)

    return prediction[0]

    

N = 78
P = 97
K = 80
temperature = 27.87974
humidity = 82.0027
ph = 6.563
rainfall = 280.980
predict = recommendation(N,P,K,temperature,humidity,ph,rainfall)


if predict in map_flipped:
    
    print(f"{predict} is a best crop to be cultivated")
else:
    print("Sorry we are not able to recommend a proper crop for this environment")

pickle.dump(model, open('model.pkl', 'wb'))
