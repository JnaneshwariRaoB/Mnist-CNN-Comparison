import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from scipy.interpolate import make_interp_spline

# Load and preprocess dataset (reduced for speed)
(x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()
x_train_full = x_train_full[:10000]
y_train_full = y_train_full[:10000]

x_train_full = x_train_full.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
y_train_full_cat = to_categorical(y_train_full, 10)
y_test_cat = to_categorical(y_test, 10)

# Define models
def build_simple_cnn():
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(16, (3, 3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_deep_cnn():
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(16, (3, 3), activation='relu'),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(),
        Dropout(0.25),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Smoothing function
def smooth_curve(x, y, points=100):
    if len(x) < 4:
        return x, y
    x_new = np.linspace(x.min(), x.max(), points)
    spline = make_interp_spline(x, y, k=3)
    y_smooth = spline(x_new)
    return x_new, y_smooth

# Train and evaluate
def train_and_evaluate(model_fn, model_name):
    print(f"Training {model_name}...")
    model = model_fn()
    history = model.fit(
        x_train_full,
        y_train_full_cat,
        epochs=10,
        batch_size=128,
        validation_split=0.2,
        verbose=0
    )
    test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
    print(f"{model_name} Test Accuracy: {test_acc:.4f}")
    return model, history

# Train models
simple_model, simple_history = train_and_evaluate(build_simple_cnn, "Simple CNN")
deep_model, deep_history = train_and_evaluate(build_deep_cnn, "Deep CNN with Dropout")

# Plot accuracy curves
plt.figure(figsize=(10, 6))
for model_name, history in [("Simple CNN", simple_history), ("Deep CNN with Dropout", deep_history)]:
    x_vals = np.arange(1, len(history.history['accuracy']) + 1)
    x_smooth, y_train_smooth = smooth_curve(x_vals, np.array(history.history['accuracy']))
    _, y_val_smooth = smooth_curve(x_vals, np.array(history.history['val_accuracy']))

    plt.plot(x_smooth, y_train_smooth, label=f'{model_name} Train Acc')
    plt.plot(x_smooth, y_val_smooth, linestyle='--', label=f'{model_name} Val Acc')

plt.title("Training vs Validation Accuracy by Model")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Confusion Matrix
def plot_conf_matrix(model, name):
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
    print(f"{name} Classification Report:\n", classification_report(y_test, y_pred_classes))

plot_conf_matrix(simple_model, "Simple CNN")
plot_conf_matrix(deep_model, "Deep CNN with Dropout")

# Compare models
simple_val_acc = simple_history.history['val_accuracy'][-1]
deep_val_acc = deep_history.history['val_accuracy'][-1]

simple_test_acc = simple_model.evaluate(x_test, y_test_cat, verbose=0)[1]
deep_test_acc = deep_model.evaluate(x_test, y_test_cat, verbose=0)[1]

print("\n🔍 Model Comparison:")
print(f"Simple CNN - Validation Accuracy: {simple_val_acc:.4f}, Test Accuracy: {simple_test_acc:.4f}")
print(f"Deep CNN with Dropout - Validation Accuracy: {deep_val_acc:.4f}, Test Accuracy: {deep_test_acc:.4f}")

# Decision logic
if simple_test_acc > deep_test_acc:
    print("✅ Simple CNN performs better on Test Data.")
elif simple_test_acc < deep_test_acc:
    print("✅ Deep CNN with Dropout performs better on Test Data.")
else:
    print("✅ Both models perform equally on Test Data.")
