import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#Create your df here:
df = pd.read_csv("profiles.csv")
#print(df.income.head())
print(df.income.value_counts())
#print(df.body_type.head())
print(df.body_type.value_counts())
#print(df.diet.head())
print(df.diet.value_counts())
#print(df.drinks.head())
print(df.drinks.value_counts())
#print(df.drugs.head())
print(df.drugs.value_counts())
plt.hist(df.height, bins=20)
plt.xlabel("Height")
plt.ylabel("Frequency")
plt.xlim(50, 90)
#plt.show()
drink_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
df["drinks_code"] = df.drinks.map(drink_mapping)
diet_mapping = {"mostly anything": 0, "anything": 1, "strictly anything": 2, "mostly vegetarian": 3, "mostly other": 4, "strictly vegetarian": 5, "vegetarian": 6, "strictly other": 7, "mostly vegan": 8, "other": 9, "strictly vegan": 10, "vegan": 11, "mostly kosher": 12, "mostly halal": 13, "strictly halal": 14, "strictly kosher": 15, "halal": 16, "kosher": 17}
df["diet_code"] = df.diet.map(diet_mapping)
drugs_mapping = {"never": 0, "sometimes": 1, "often": 2}
df["drugs_code"] = df.drugs.map(drugs_mapping)
body_type_mapping = {"average": 0, "fit": 1, "athletic": 2, "thin": 3, "curvy": 4, "a little extra": 5, "skinny": 6, "full figured": 7, "overweight": 8, "jacked": 9, "used up": 10, "rather not say": 11}
df["body_type_code"] = df.body_type.map(body_type_mapping)
plt.hist(df.body_type_code, bins=20)
plt.xlabel("Body Type")
plt.ylabel("Frequency")
plt.xlim(0,12)
plt.show()
essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]
# Removing the NaNs
all_essays = df[essay_cols].replace(np.nan, '', regex=True)
# Combining the essays
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)
df["essay_len"] = all_essays.apply(lambda x: len(x))
feature_data = df[['drinks_code', 'drugs_code', 'height', 'diet_code']]
x = feature_data.values
min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)

## K neighbors
training_data, validation_data, training_labels,validation_labels = train_test_split(feature_data,df['body_type_code'],test_size = 0.2,random_state = 100)
accuracies =[]
for i in range(1,100):
  classifier = KNeighborsClassifier(n_neighbors = i)
  classifier.fit(training_data, training_labels)
  accuracies.append((classifier.score(validation_data, validation_labels)))
  
k_list = range(1,100)
plt.plot(k_list,accuracies)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Breast Cancer Classifier Accuracy")
plt.show()

#Support Vectors
classifier1 = SVC(kernel = 'rbf', gamma = 100, C = 100)

classifier1.fit(training_set[['drinks_code', 'drugs_code', 'height', 'diet_code']], training_set['body_type_code'])

draw_boundary(ax, classifier)
print(classifier.score(validation_set[['drinks_code', 'drugs_code', 'height', 'diet_code']],validation_set['body_type_code']))

#Multiple Linear Regressor

regr = linear_model.LinearRegression()
regr.fit(feature_data,df['body_type'])
print(regr.coef_)
print(regr.intercept_)

y_predict = regr.predict(feature_data)

plt.plot(y_predict, feature_data)
plt.show()
