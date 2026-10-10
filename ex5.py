import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score
data = 'Dataset/Housing.csv'
df = pd.read_csv(data)
df.head()
df = pd.get_dummies(df, drop_first=True)
df.head()
# In[5]:
X = df.drop("price", axis=1)
y = df["price"]
# In[6]:
X_train,X_test,y_train,y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42
)
# In[7]:
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# In[80]:
poly = PolynomialFeatures(
 degree=4,
 include_bias=False
)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)
print("Original Features :", X_train.shape[1])
print("Polynomial Features :", X_train_poly.shape[1])
# In[81]:
linear = LinearRegression()
linear.fit(X_train_poly, y_train)
train_pred = linear.predict(X_train_poly)
test_pred = linear.predict(X_test_poly)
train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)
print("LINEAR REGRESSION")
print("Training R2 :", round(train_r2,4))
print("Testing R2 :", round(test_r2,4))
# In[83]:
ridge = Ridge(alpha=100)
ridge.fit(X_train_poly, y_train)
train_pred = ridge.predict(X_train_poly)
test_pred = ridge.predict(X_test_poly)
ridge_train = r2_score(y_train, train_pred)
ridge_test = r2_score(y_test, test_pred)
print("RIDGE")
print("Training R2 :", round(ridge_train,4))
print("Testing R2 :", round(ridge_test,4))
# In[ ]:
lasso = Lasso(
 alpha=0.1,
 max_iter=50000
)
lasso.fit(X_train_poly, y_train)
train_pred = lasso.predict(X_train_poly)
test_pred = lasso.predict(X_test_poly)
lasso_train = r2_score(y_train, train_pred)
lasso_test = r2_score(y_test, test_pred)
print("LASSO")
print("Training R2 :", round(lasso_train,4))
print("Testing R2 :", round(lasso_test,4))
# In[77]:
elastic = ElasticNet(
 alpha=0.1,
 l1_ratio=0.5,
 max_iter=50000
)
elastic.fit(X_train_poly, y_train)
train_pred = elastic.predict(X_train_poly)
test_pred = elastic.predict(X_test_poly)
elastic_train = r2_score(y_train, train_pred)
elastic_test = r2_score(y_test, test_pred)
print("ELASTIC NET")
print("Training R2 :", round(elastic_train,4))
print("Testing R2 :", round(elastic_test,4))
# In[78]:
comparison = pd.DataFrame({
 "Model":[
 "Linear Regression",
 "Ridge",
 "Lasso",
 "Elastic Net"
 ],
 "Train R2":[
 train_r2,
 ridge_train,
 lasso_train,
 elastic_train
 ],
 "Test R2":[
 test_r2,
 ridge_test,
 lasso_test,
 elastic_test
 ]
})
print(comparison)
# In[79]:
# In[62]:
comparison["Difference"] = comparison["Train R2"] - comparison["Test R2"]
print(comparison)
comparison.plot(
 x="Model",
 y=["Train R2", "Test R2"],
 kind="bar",
 figsize=(10,6)
)
plt.title("Reducing Overfitting using Regularization")
plt.ylabel("R² Score")
plt.xticks(rotation=10)
plt.grid(axis="y")
plt.show()
