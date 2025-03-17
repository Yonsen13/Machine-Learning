#create your individual project here!
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from datetime import datetime

# 1. Load dataset
train_df  = pd.read_csv('train.csv')
test_df = pd.read_csv("test.csv")

print('Jumlah nilai yang hilang di TRAIN:')
print(train_df.isnull().sum())

print("\nJumlah nilai yang hilang di TEST:")
print(test_df.isnull().sum())

# for df in [train_df, test_df]:
#     df.fillna({
#         'career_start': df['career_start'].median(),
#         'career_end': df['career_end'].median(),
#         "followers_count": df['followers_count'].median(),
#         'graduation': df['graduation'].mediaan(),
#         'relation': df['relation'].mode()[0],
#         'education_status': df['education_status'].mode()[0],
#         'city': df['city'].mode()[0],
#         'occupation_type': df['occupation_type'].mode()[0],

#     },inplace=True)

def calculate_age(bdate):
    try:
        year=int(bdate.split('.')[-1])
        return datetime.now().year - year
    except:
        return np.nan 
for df in [train_df, test_df]:
    df['age']=df['bdate'].apply(calculate_age)
    df['age'].fillna(df['age'].median(), inplace=True)

# 5. Encode data kategorikal
label_cols=['sex', 'has_photo', 'has_mobile', 'education_form', 'education_status', 'langs', 'life_man', 'people_main', 'city', 'occupation_type', 'occupation_name']

label_encoders= {}
for col in label_cols:
    le= LabelEncoder()
    train_df[col]=le.fit_transform(train_df[col].astype(str))
    test_df[col]= le.transform(test_df[col].astype(str))

# pilih fitur dan target
features= ['sex', 'age', 'has_photo', 'has_mobile', 'followers_count', 'graduation', 'education_form', 'relation', 'education_status', 'langs', 'life_main', 'people_main', 'city', 'last_seen', 'occupation_type', 'occupation_name', 'career_start', 'career_end']

X_train = train_df[features]
X_train = train_df['result']
X_test = test_df[features]

# 7. Normalisasi fitur
scaler = StandardScaler()
X_train= scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 8. Melatih model KNN dengan k=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 9. Prediksi pada test.csv
y_pred = knn.predict(X_test)

# 10. simpan hasil prediksi
test_df['result'] = y_pred
test_df[['d', 'result']].to_csv('submission.csv', index= False)

