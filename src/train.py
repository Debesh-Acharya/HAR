import pickle
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from models import build_lstm_model

# ===============================
# Load preprocessed data
# ===============================
with open("models/preprocessed_data.pkl", "rb") as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

# Debug prints of labels
print("Unique y_train labels:", np.unique(y_train))
print("Max y_train label:", y_train.max())
print("Unique y_test labels:", np.unique(y_test))
print("Max y_test label:", y_test.max())

# Use max label between train and test to determine num_classes
num_classes = max(y_train.max(), y_test.max()) + 1

input_shape = (X_train.shape[1], X_train.shape[2])

# ===============================
# Build and train model
# ===============================
model = build_lstm_model(input_shape, num_classes)
model.summary()

checkpoint = ModelCheckpoint(
    "models/best_model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

earlystop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=64,
    callbacks=[checkpoint, earlystop]
)

# ===============================
# Evaluate
# ===============================
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc*100:.2f}%")
