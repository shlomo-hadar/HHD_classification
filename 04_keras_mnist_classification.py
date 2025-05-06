import os
import cv2
import numpy as np
from sklearn.metrics import accuracy_score
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# import keras
from tensorflow.keras import Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from math import inf
# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

max_accuracy = -inf
show: bool = False
show_conf_matrix: bool = False


def output_model_to_json(model):
    model_json = model.to_json()
    with open("final_model_configuration.json", "w") as json_file:
        json_file.write(model_json)


def plot_and_save_confusion_matrix(model, x_test, y_test, class_names=None, normalize=False, csv_path='confusion_matrix.csv'):
    """
    Plots and saves a confusion matrix to CSV.

    Parameters:
    - model: Trained Keras model
    - x_test: Test images
    - y_test: True test labels (one-hot or label encoded)
    - class_names: List of class labels (e.g., Hebrew letters)
    - normalize: If True, normalize the confusion matrix
    - csv_path: File path to save the CSV matrix
    """
    # Predict class probabilities and convert to labels
    global show_conf_matrix
    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Handle one-hot encoded y_test
    if y_test.ndim > 1 and y_test.shape[1] > 1:
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true = y_test

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)

    # Default class names if not provided
    if class_names is None:
        class_names = [f"{i}" for i in range(cm.shape[0])]

    # Save to CSV using pandas
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    df_cm.to_csv(csv_path)
    print(f"Confusion matrix saved to {csv_path}")

    # Plot the matrix
    if show_conf_matrix:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        fig, ax = plt.subplots(figsize=(12, 10))
        disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
        plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
        plt.tight_layout()
        plt.show()


def evaluate_per_letter_accuracy(model, x_test, y_test, class_names=None):
    """
    Evaluates and prints per-class accuracy and average accuracy for a classification model.

    Parameters:
    - model: Trained Keras model.
    - x_test: Test input data (e.g., images).
    - y_test: Test labels (one-hot encoded or label-encoded).
    - class_names: Optional list of class names (e.g., Hebrew letters). If None, class indices will be used.

    Returns:
    - Dictionary with per-class accuracies and average accuracy.
    """
    # Predict class probabilities
    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Handle both one-hot and label-encoded y_test
    if y_test.ndim > 1 and y_test.shape[1] > 1:
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true = y_test

    num_classes = np.max(y_true) + 1
    class_accuracies = {}

    print("Per-letter accuracy:")

    for i in range(num_classes):
        idx = np.where(y_true == i)[0]
        if len(idx) == 0:
            acc = 0
            label = f"Class {i} (no samples)"
        else:
            acc = accuracy_score(y_true[idx], y_pred[idx])
            label = class_names[i] if class_names else f"Letter {i}"

        class_accuracies[label] = acc
        print(f"{label}: {acc:.4f}")

    avg_acc = np.mean(list(class_accuracies.values()))
    print(f"\nAverage accuracy across all letters: {avg_acc:.4f}")

    with open('final_model_train_test_per_letter_accuracy.txt', 'w') as output_file:
        output_file.write(f'{"Letter":<18} : {"Accuracy":<8}\n')
        for name, val in class_accuracies.items():
            output_file.write(f'{name:<18} : {round(val, 2):<8}\n')
        output_file.write(f'---------------------------------\n')
        output_file.write(f'{"AVG":<18} : {round(avg_acc, 2):<8}\n')

    return {
        "per_class_accuracy": class_accuracies,
        "average_accuracy": avg_acc
    }


def train_with_different_regularizations_NN(X_train, y_train, X_val, y_val, X_test, y_test, input_shape, num_classes):
    """
    train המודל במספר קונפיגורציות שונות של רגולריזציה
    """
    from tensorflow.keras.regularizers import l1, l2, l1_l2
    from tensorflow.keras.layers import Dense, Flatten, Dropout
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    global max_accuracy, show

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

        # current_accuracy = float(history.history['accuracy'][-1])
        current_accuracy = float(test_acc)
        if current_accuracy > max_accuracy:
            plt.savefig('final_model_precision_and_loss_plot.png')
            max_accuracy = current_accuracy
            output_model_to_json(model=model)
            evaluate_per_letter_accuracy(model=model, x_test=X_test, y_test=y_test)
            plot_and_save_confusion_matrix(model=model, x_test=X_test, y_test=y_test)

        plt.tight_layout()
        if not show:
            plt.clf()
            continue
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
    plt.title('compare persicion validation per configuration')
    plt.xlabel('epochs')
    plt.ylabel('persicion validation')
    plt.legend(loc='lower right')

    # גרף שגיאת validation
    plt.subplot(2, 1, 2)
    for result in results:
        plt.plot(result['history']['val_loss'], label=result['config'])
    plt.title(' validation error per configuration')
    plt.xlabel('epochs')
    plt.ylabel('validation error')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show() if show else None

    return results


# def train_with_different_regularizations_CNN(X_train, y_train, X_val, y_val, X_test, y_test, input_shape, num_classes):
# # def train_with_different_regularizations(X_train, y_train, X_val, y_val, X_test, y_test, input_shape, num_classes):
#     """
#     Trains a CNN model following the architecture specified in the assignment:
#     INPUT=>[CONV=>RELU=>CONV=>RELU=>POOL=>DO]*3=>FC=>RELU=>DO=>FC
#
#     Tests with and without data augmentation.
#     """
#     from tensorflow.keras.models import Sequential
#     from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
#     from tensorflow.keras.optimizers import Adam
#     from tensorflow.keras.callbacks import EarlyStopping
#     from tensorflow.keras.preprocessing.image import ImageDataGenerator
#     global max_accuracy, show
#
#     results = []
#
#     # Define configurations for testing
#     configurations = [
#         {
#             'name': '1. Without augmentation',
#             'use_augmentation': False
#         },
#         {
#             'name': '2. With augmentation',
#             'use_augmentation': True
#         }
#     ]
#
#     for config in configurations:
#         print(f"\n\n========== Training: {config['name']} ==========")
#
#         # Create the CNN model as per specifications
#         model = Sequential()
#
#         # First iteration: CONV=>RELU=>CONV=>RELU=>POOL=>DO with 32 filters
#         model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
#         model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
#         model.add(MaxPooling2D(pool_size=(2, 2)))
#         model.add(Dropout(0.25))
#
#         # Second iteration: CONV=>RELU=>CONV=>RELU=>POOL=>DO with 64 filters
#         model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
#         model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
#         model.add(MaxPooling2D(pool_size=(2, 2)))
#         model.add(Dropout(0.25))
#
#         # Third iteration: CONV=>RELU=>CONV=>RELU=>POOL=>DO with 128 filters
#         model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
#         model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
#         model.add(MaxPooling2D(pool_size=(2, 2)))
#         model.add(Dropout(0.25))
#
#         # FC=>RELU=>DO
#         model.add(Flatten())
#         model.add(Dense(512, activation='relu'))
#         model.add(Dropout(0.5))
#
#         # Output layer
#         model.add(Dense(num_classes, activation='softmax'))
#
#         # Print model summary
#         model.summary()
#
#         # Compile the model
#         model.compile(
#             optimizer=Adam(learning_rate=0.001),
#             loss='categorical_crossentropy',
#             metrics=['accuracy']
#         )
#
#         # Define Early Stopping
#         early_stop = EarlyStopping(
#             monitor='val_loss',
#             patience=10,
#             restore_best_weights=True,
#             verbose=1
#         )
#
#         # Train the model
#         if config['use_augmentation']:
#             # Setup data augmentation - Note: Assuming X_train is already normalized (divided by 255)
#             # datagen = ImageDataGenerator(
#             #     width_shift_range=0.1,
#             #     height_shift_range=0.1,
#             #     horizontal_flip=False,
#             #     vertical_flip=False,
#             #     rotation_range=10,
#             #     shear_range=0.2,
#             #     brightness_range=(0.8, 1.2),  # Less aggressive brightness change
#             #     # No rescaling since data is already normalized in preprocessing step
#             # )
#
#             # Train using data augmentation
#             datagen = ImageDataGenerator(
#                 width_shift_range=0.1,
#                 height_shift_range=0.1,
#                 rotation_range=8,
#                 shear_range=0.15,
#                 zoom_range=0.1,
#                 brightness_range=(0.9, 1.1),
#                 horizontal_flip=True,
#                 fill_mode='nearest'
#             )
#
#             # Fit on training data if needed (not mandatory for this setup)
#             # datagen.fit(X_train)
#
#             # Train using updated ImageDataGenerator with .fit()
#             history = model.fit(
#                 datagen.flow(X_train, y_train, batch_size=64, shuffle=True),
#                 steps_per_epoch=int(np.ceil(len(X_train) / 64)),
#                 epochs=50,
#                 validation_data=(X_val, y_val),
#                 callbacks=[early_stop],
#                 verbose=1
#             )
#         else:
#             # Train without data augmentation
#             history = model.fit(
#                 X_train, y_train,
#                 batch_size=64,
#                 epochs=50,
#                 validation_data=(X_val, y_val),
#                 callbacks=[early_stop],
#                 verbose=1
#             )
#
#         # Evaluate the model on test set
#         test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
#         print(f"Accuracy on test set: {test_acc:.4f}")
#
#         # Save results
#         results.append({
#             'config': config['name'],
#             'test_accuracy': test_acc,
#             'history': history.history
#         })
#
#         # Plot and save graphs
#         plt.figure(figsize=(12, 5))
#
#         # Accuracy graph
#         plt.subplot(1, 2, 1)
#         plt.plot(history.history['accuracy'], label='train')
#         plt.plot(history.history['val_accuracy'], label='validation')
#         plt.title(f'Accuracy - {config["name"]}')
#         plt.xlabel('epochs')
#         plt.ylabel('accuracy')
#         plt.legend()
#
#         # Loss graph
#         plt.subplot(1, 2, 2)
#         plt.plot(history.history['loss'], label='train')
#         plt.plot(history.history['val_loss'], label='validation')
#         plt.title(f'Loss - {config["name"]}')
#         plt.xlabel('epochs')
#         plt.ylabel('loss')
#         plt.legend()
#
#         # Save the best model
#         current_accuracy = float(test_acc)
#         if current_accuracy > max_accuracy:
#             plt.savefig('final_model_precision_and_loss_plot.png')
#             max_accuracy = current_accuracy
#             output_model_to_json(model=model)
#             evaluate_per_letter_accuracy(model=model, x_test=X_test, y_test=y_test)
#             plot_and_save_confusion_matrix(model=model, x_test=X_test, y_test=y_test)
#
#         plt.tight_layout()
#         if show:
#             plt.show()
#         else:
#             plt.clf()
#
#     # Summary of results
#     print("\n\n===== Results Summary =====")
#     for result in sorted(results, key=lambda x: x['test_accuracy'], reverse=True):
#         print(f"Configuration: {result['config']}, Test Accuracy: {result['test_accuracy']:.4f}")
#
#     # Compare all configurations in one graph
#     plt.figure(figsize=(15, 10))
#
#     # Validation accuracy graph
#     plt.subplot(2, 1, 1)
#     for result in results:
#         plt.plot(result['history']['val_accuracy'], label=result['config'])
#     plt.title('Validation Accuracy Comparison')
#     plt.xlabel('epochs')
#     plt.ylabel('validation accuracy')
#     plt.legend(loc='lower right')
#
#     # Validation loss graph
#     plt.subplot(2, 1, 2)
#     for result in results:
#         plt.plot(result['history']['val_loss'], label=result['config'])
#     plt.title('Validation Loss Comparison')
#     plt.xlabel('epochs')
#     plt.ylabel('validation loss')
#     plt.legend(loc='upper right')
#
#     plt.tight_layout()
#     if show:
#         plt.show()
#
#     # Create results.txt file
#     with open('results.txt', 'w') as f:
#         f.write("Final Model Configuration:\n")
#
#         # Find the best configuration
#         best_config = max(results, key=lambda x: x['test_accuracy'])
#         f.write(f"Best Configuration: {best_config['config']}\n")
#         f.write(f"Test Accuracy: {best_config['test_accuracy']:.4f}\n\n")
#
#         # Refer to the saved plot
#         f.write("Loss curves are saved in final_model_precision_and_loss_plot.png\n\n")
#
#     return results


def train_with_different_regularizations_CNN(X_train, y_train, X_val, y_val, X_test, y_test, input_shape, num_classes):
    """
    Trains a CNN model following the architecture specified in the assignment:
    INPUT=>[CONV=>RELU=>CONV=>RELU=>POOL=>DO]*3=>FC=>RELU=>DO=>FC

    Tests with and without data augmentation.
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    global max_accuracy, show

    results = []

    # Define configurations for testing
    configurations = [
        {
            'name': '1. Without augmentation',
            'use_augmentation': False
        },
        {
            'name': '2. With augmentation',
            'use_augmentation': True
        }
    ]

    for config in configurations:
        print(f"\n\n========== Training: {config['name']} ==========")

        # Create the CNN model as per specifications
        model = Sequential()

        # First iteration: CONV=>RELU=>CONV=>RELU=>POOL=>DO with 32 filters
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Second iteration: CONV=>RELU=>CONV=>RELU=>POOL=>DO with 64 filters
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Third iteration: CONV=>RELU=>CONV=>RELU=>POOL=>DO with 128 filters
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # FC=>RELU=>DO
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))

        # Output layer
        model.add(Dense(num_classes, activation='softmax'))

        # Print model summary
        model.summary()

        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Define Early Stopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )

        # Train the model
        if config['use_augmentation']:
            # Setup data augmentation
            datagen = ImageDataGenerator(
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=False,
                vertical_flip=False,
                rotation_range=10,
                shear_range=0.2,
                brightness_range=(0.2, 1.8),
                rescale=1. / 255 # This should be applied only if data isn't already normalized
            )

            # Fit the datagen on the training data
            datagen.fit(X_train)

            # Train using data augmentation
            history = model.fit(
                datagen.flow(X_train, y_train, batch_size=64),
                steps_per_epoch=len(X_train) // 64,
                epochs=50,
                validation_data=(X_val, y_val),
                callbacks=[early_stop],
                verbose=1
            )
        else:
            # Train without data augmentation
            history = model.fit(
                X_train, y_train,
                batch_size=64,
                epochs=50,
                validation_data=(X_val, y_val),
                callbacks=[early_stop],
                verbose=1
            )

        # Evaluate the model on test set
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"Accuracy on test set: {test_acc:.4f}")

        # Save results
        results.append({
            'config': config['name'],
            'test_accuracy': test_acc,
            'history': history.history
        })

        # Plot and save graphs
        plt.figure(figsize=(12, 5))

        # Accuracy graph
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='train')
        plt.plot(history.history['val_accuracy'], label='validation')
        plt.title(f'Accuracy - {config["name"]}')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.legend()

        # Loss graph
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='validation')
        plt.title(f'Loss - {config["name"]}')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()

        # Save the best model
        current_accuracy = float(test_acc)
        if current_accuracy > max_accuracy:
            plt.savefig('final_model_precision_and_loss_plot.png')
            max_accuracy = current_accuracy
            output_model_to_json(model=model)
            evaluate_per_letter_accuracy(model=model, x_test=X_test, y_test=y_test)
            plot_and_save_confusion_matrix(model=model, x_test=X_test, y_test=y_test)

        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.clf()

    # Summary of results
    print("\n\n===== Results Summary =====")
    for result in sorted(results, key=lambda x: x['test_accuracy'], reverse=True):
        print(f"Configuration: {result['config']}, Test Accuracy: {result['test_accuracy']:.4f}")

    # Compare all configurations in one graph
    plt.figure(figsize=(15, 10))

    # Validation accuracy graph
    plt.subplot(2, 1, 1)
    for result in results:
        plt.plot(result['history']['val_accuracy'], label=result['config'])
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('epochs')
    plt.ylabel('validation accuracy')
    plt.legend(loc='lower right')

    # Validation loss graph
    plt.subplot(2, 1, 2)
    for result in results:
        plt.plot(result['history']['val_loss'], label=result['config'])
    plt.title('Validation Loss Comparison')
    plt.xlabel('epochs')
    plt.ylabel('validation loss')
    plt.legend(loc='upper right')

    plt.tight_layout()
    if show:
        plt.show()

    # Create results.txt file
    with open('results.txt', 'w') as f:
        f.write("Final Model Configuration:\n")

        # Find the best configuration
        best_config = max(results, key=lambda x: x['test_accuracy'])
        f.write(f"Best Configuration: {best_config['config']}\n")
        f.write(f"Test Accuracy: {best_config['test_accuracy']:.4f}\n\n")

        # Refer to the saved plot
        f.write("Loss curves are saved in final_model_precision_and_loss_plot.png\n\n")

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
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)),
#     MaxPooling2D(pool_size=(2, 2)),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dense(num_classes, activation='softmax')
# ])
#
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
# # train
# # model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
# model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# train_with_different_regularizations_NN(X_train, y_train, X_val, y_val, X_test, y_test, input_shape=(img_size, img_size, 1), num_classes=num_classes)
train_with_different_regularizations_CNN(X_train, y_train, X_val, y_val, X_test, y_test, input_shape=(img_size, img_size, 1), num_classes=num_classes)
