import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Simulated sensor data
np.random.seed(42)
data = pd.DataFrame({
    "temperature": np.random.normal(70, 5, 500),
    "vibration": np.random.normal(0.5, 0.1, 500),
    "pressure": np.random.normal(30, 3, 500),
})

# Failure label (simple engineering rule)
data["failure"] = (
    (data["temperature"] > 75) |
    (data["vibration"] > 0.7)
).astype(int)

X = data.drop("failure", axis=1)
y = data["failure"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
