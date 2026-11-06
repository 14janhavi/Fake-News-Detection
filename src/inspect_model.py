
import joblib

# Load the saved model
model = joblib.load("models/fakenews_model.joblib")

# Display the type of object loaded
print("Model type:", type(model))

# Display the saved object's contents
print("\nModel contents:")
print(model)
