import os
import cv2
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# import keras
from tensorflow.keras import Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

def train_with_different_regularizations(X_train, y_train, X_val, y_val, X_test, y_test, input_shape, num_classes):
    """
    train המודל במספר קונפיגורציות שונות של רגולריזציה
    """
    from tensorflow.keras.regularizers import l1, l2, l1_l2
    from tensorflow.keras.layers import Dense, Flatten, Dropout
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping

    results = []
    configurations = [
        {
            'name': '1. WO regularization',
            'kernel_regularizer': None,
            'dropout_rate': 0.0
        },
        {
            'name': '2. L1 regularization (λ=0.001)',
            'kernel_regularizer': l1(0.001),
            'dropout_rate': 0.0
        },
        {
            'name': '2. L1 regularization (λ=0.01)',
            'kernel_regularizer': l1(0.01),
            'dropout_rate': 0.0
        },
        {
            'name': '3. L2 regularization (λ=0.001)',
            'kernel_regularizer': l2(0.001),
            'dropout_rate': 0.0
        },
        {
            'name': '3. L2 regularization (λ=0.01)',
            'kernel_regularizer': l2(0.01),
            'dropout_rate': 0.0
        },
        {
            'name': '4. Dropout (p=0.5)',
            'kernel_regularizer': None,
            'dropout_rate': 0.5
        },
        {
            'name': '5. L2 (λ=0.001) + Dropout (p=0.5)',
            'kernel_regularizer': l2(0.001),
            'dropout_rate': 0.5
        },
        {
            'name': '5. L2 (λ=0.01) + Dropout (p=0.5)',
            'kernel_regularizer': l2(0.01),
            'dropout_rate': 0.5
        }
    ]

    for config in configurations:
        print(f"\n\n========== train: {config['name']} ==========")

        # יצירת המודל עם הרגולריזציה המתאימה
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))

        # שכבת קלט עם 1024 יחידות
        model.add(Dense(1024, activation='relu',
                        kernel_regularizer=config['kernel_regularizer']))
        if config['dropout_rate'] > 0:
            model.add(Dropout(config['dropout_rate']))

        # שכבה מסותרת ראשונה עם 512 יחידות
        model.add(Dense(512, activation='relu',
                        kernel_regularizer=config['kernel_regularizer']))
        if config['dropout_rate'] > 0:
            model.add(Dropout(config['dropout_rate']))

        # שכבה מסותרת שנייה עם 512 יחידות
        model.add(Dense(512, activation='relu',
                        kernel_regularizer=config['kernel_regularizer']))
        if config['dropout_rate'] > 0:
            model.add(Dropout(config['dropout_rate']))

        # שכבת פלט עם 27 יחידות - ללא רגולריזציה
        model.add(Dense(num_classes, activation='softmax'))

        # קומפילציה
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # הגדרת Early Stopping למניעת Overfitting
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )

        # train המודל
        history = model.fit(
            X_train, y_train,
            batch_size=64,
            epochs=50,
            validation_data=(X_val, y_val),
            callbacks=[early_stop],
            verbose=1
        )

        # הערכת המודל על סט הטסט
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"persicion on test set: {test_acc:.4f}")

        # שמירת התוצאות
        results.append({
            'config': config['name'],
            'test_accuracy': test_acc,
            'history': history.history
        })

        # הצגת גרפים
        plt.figure(figsize=(12, 5))

        # גרף persicion
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='train')
        plt.plot(history.history['val_accuracy'], label='validation')
        plt.title(f'persicion - {config["name"]}')
        plt.xlabel('epochs')
        plt.ylabel('persicion')
        plt.legend()

        # גרף loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='validation')
        plt.title(f'loss - {config["name"]}')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()

        plt.tight_layout()
        plt.show()

    # הצגת סיכום התוצאות
    print("\n\n===== סיכום תוצאות =====")
    for result in sorted(results, key=lambda x: x['test_accuracy'], reverse=True):
        print(f"קונפיגורציה: {result['config']}, persicion על סט הטסט: {result['test_accuracy']:.4f}")

    # השוואת כל הקונפיגורציות בגרף אחד
    plt.figure(figsize=(15, 10))

    # גרף persicion validation
    plt.subplot(2, 1, 1)
    for result in results:
        plt.plot(result['history']['val_accuracy'], label=result['config'])
    plt.title('השוואת persicion validation בין הקונפיגורציות השונות')
    plt.xlabel('epochs')
    plt.ylabel('persicion validation')
    plt.legend(loc='lower right')

    # גרף שגיאת validation
    plt.subplot(2, 1, 2)
    for result in results:
        plt.plot(result['history']['val_loss'], label=result['config'])
    plt.title('השוואת שגיאת validation בין הקונפיגורציות השונות')
    plt.xlabel('epochs')
    plt.ylabel('שגיאת validation')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

    return results


# דוגמת קריאה לפונקציה (יש להוסיף בסוף הקוד הראשי):
# results = train_with_different_regularizations(X_train, y_train, X_val, y_val, X_test, y_test, input_shape=(32, 32, 1), num_classes=27)


def preprocess_image(img_path, output_size):
    # קריאה והמרה לגווני אפור
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # קבלת גובה ורוחב
    h, w = img.shape

    # חישוב כמה ריפוד נדרש מכל צד כדי להפוך למרובע
    if h > w:
        diff = h - w
        left = diff // 2
        right = diff - left
        top = bottom = 0
    else:
        diff = w - h
        top = diff // 2
        bottom = diff - top
        left = right = 0

    # הוספת ריפוד לבן (255)
    img_padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255)

    # שינוי גודל
    img_resized = cv2.resize(img_padded, (output_size, output_size))

    # הפיכה ל-negative
    img_negative = 255 - img_resized

    return img_negative


def shuffle_stack(images, labels):
    combined = list(zip(images, labels))

    # ערבב את הרשימה
    np.random.seed(42)  # כדי שהתוצאה תהיה קבועה (אופציונלי)
    np.random.shuffle(combined)

    # פיצול חזרה לשני מערכים
    shuffled_images, shuffled_labels = zip(*combined)

    # הפוך ל־NumPy arrays
    shuffled_images = np.array(shuffled_images)
    shuffled_labels = np.array(shuffled_labels)
    return shuffled_images, shuffled_labels


def split_data(images, labels):
    # חצי ראשוני לחלק את הדאטה בין train וסטינג
    X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.2, random_state=42)

    # חלק את הנתונים ב-X_temp לסט של ולידציה וסט טסטינג
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test


# הגדרות
data_dir = '.'
img_size = 32
num_classes = 27

X = []
y = []
t = datetime.now()
# קריאת התמונות והצגתן
for label in range(num_classes):
    folder_path = os.path.join(data_dir, str(label))
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        preprocessed_image = preprocess_image(img_path=img_path, output_size=img_size)
        # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # img = cv2.resize(img, (img_size, img_size))

        # הצגת תמונה (אפשר להוריד את השורה הזו אחרי בדיקה)
        # plt.imshow(preprocessed_image)
        plt.title(f"Label: {label}")
        plt.axis('off')
        # plt.show()

        X.append(preprocessed_image)
        y.append(label)
print((datetime.now()-t).total_seconds())
# המרה למערך ונרמול
X = np.array(X) / 255.0
X = X.reshape(-1, img_size, img_size, 1)


# המרה ל־one-hot באופן ידני
def manual_one_hot(labels, num_classes):
    one_hot = np.zeros((len(labels), num_classes))
    for idx, val in enumerate(labels):
        one_hot[idx, val] = 1
    return one_hot


# פיצול ל-Train/Test
# X_train, X_val, X_test, y_train, y_val, y_test = split_data(images=X, labels=y)
y = manual_one_hot(y, num_classes)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test = X_test[:len(X_test)//2]
X_val = X_test[len(X_test)//2:]
y_test = y_test[:len(y_test)//2]
y_val = y_test[len(y_test)//2:]
# בניית המודל
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train
# model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

train_with_different_regularizations(X_train, y_train, X_val, y_val, X_test, y_test, input_shape=(img_size, img_size, 1), num_classes=num_classes)
