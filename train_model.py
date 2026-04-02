import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle

# ----------------------------
# IMPROVED DATASET
# ----------------------------
data = {
    'followers': [100,50,300,10,500,5,1000,20,400,15,0,10,200,30,600,2],
    'following': [50,200,150,500,100,800,300,600,200,700,0,5,400,500,200,900],
    'posts': [20,5,50,2,80,1,150,3,60,2,0,1,40,2,90,1],
    'bio_length': [50,10,100,5,120,2,200,3,90,4,0,5,70,3,150,2],
    'fake': [0,1,0,1,0,1,0,1,0,1,0,0,0,1,0,1]
}

df = pd.DataFrame(data)

# ----------------------------
# FEATURE ENGINEERING
# ----------------------------
df['ratio'] = df['followers'] / (df['following'] + 1)

df['low_followers_high_following'] = ((df['followers'] < 50) & (df['following'] > 300)).astype(int)
df['low_posts'] = ((df['posts'] < 3) & (df['followers'] < 20)).astype(int)

# ----------------------------
# MODEL PREP
# ----------------------------
X = df[['followers','following','posts','bio_length',
        'ratio','low_followers_high_following','low_posts']]
y = df['fake']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.2)

# ----------------------------
# TRAIN MODEL
# ----------------------------
model = RandomForestClassifier(n_estimators=200)
model.fit(X_train,y_train)

# ----------------------------
# EVALUATE MODEL
# ----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Model Accuracy: {accuracy*100:.2f}%")

# ----------------------------
# SAVE
# ----------------------------
pickle.dump(model, open('model.pkl','wb'))
pickle.dump(scaler, open('scaler.pkl','wb'))