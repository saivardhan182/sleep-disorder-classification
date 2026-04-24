import tkinter
from tkinter import *
from tkinter import filedialog
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


from tensorflow.keras.models import Sequential


from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from  tensorflow.keras.optimizers import Adam
from  tensorflow.keras.callbacks import EarlyStopping
from tensorflow. keras.models import load_model


global dataset_path
global X_train, X_test, y_train, y_test

accuracy = []
precision = []
recall = []
fscore = []  
labels = ['None', 'Insomnia', 'Sleep Apnea']
algorithm = []
def uploaddataset():
    global dataset_path
    dataset_path = filedialog.askopenfilename(initialdir="Dataset")
    if dataset_path:
        text_widget.delete(1.0,  END)
        text_widget.insert(END, f"Dataset uploaded\n")
        sleep_df = pd.read_csv(dataset_path, keep_default_na=False, na_values=[''])
        text_widget.insert(END, sleep_df)
        
def preprocessdata():
    global dataset_path
    text_widget.delete(1.0,  END)
    sleep_df = pd.read_csv(dataset_path, keep_default_na=False, na_values=[''])
    
    sleep_df.duplicated().sum()
    sleep_df.drop(columns=['Person ID'], inplace=True)
    sleep_df[['Systolic', 'Diastolic']] = sleep_df['Blood Pressure'].str.split('/', expand=True).astype(int)
    sleep_df = sleep_df.drop(columns=['Blood Pressure'])

    label_encoder = LabelEncoder()
    sleep_df['Gender'] = label_encoder.fit_transform(sleep_df['Gender'])        #Gender
    value_counts = sleep_df['Gender'].value_counts()
    # Gender Pie Chart
    plt.figure(figsize=(5, 4))
    labels = label_encoder.inverse_transform(value_counts.index)
    total_count = value_counts.sum()
    explode = [1 if count < total_count * 0.1 else 0 for count in value_counts]
    plt.pie(value_counts, labels=labels, autopct="%0.2f", explode=explode, startangle=140)
    plt.legend(labels, loc="best")
    plt.axis('equal')
    sleep_df['Occupation'] = label_encoder.fit_transform(sleep_df['Occupation'])        #Occupation
    value_counts = sleep_df['Occupation'].value_counts()
    # Occupation Pie Chart
    plt.figure(figsize=(10, 7))
    labels = label_encoder.inverse_transform(value_counts.index)
    total_count = value_counts.sum()
    explode = [1 if count < total_count * 0.1 else 0 for count in value_counts]
    plt.pie(value_counts, labels=labels, autopct="%0.2f", explode=explode, startangle=140)
    plt.legend(labels, loc="best")
    plt.axis('equal')
    sleep_df['BMI Category'] = label_encoder.fit_transform(sleep_df['BMI Category'])        # BMI Category
    sleep_df['Sleep Disorder'] = label_encoder.fit_transform(sleep_df['Sleep Disorder'])    # Sleep Disorder
    value_counts = sleep_df['Sleep Disorder'].value_counts()
    text_widget.insert(END,'Dataset is prerocessed and below 3 classes are founnd in dataset for Sleep Disorder: \n')
    text_widget.insert(END, "Sleep Disorder Distribution:\n")
    text_widget.insert(END, "-" * 30 + "\n")
    for disorder, count in zip(label_encoder.inverse_transform(value_counts.index), value_counts):
        text_widget.insert(END, f"{disorder}: {count} occurrences\n")
    text_widget.insert(END, "-" * 30 + "\n")
    # Sleep Disorder Pie Chart
    plt.figure(figsize=(6, 6))
    plt.pie(value_counts, labels=label_encoder.inverse_transform(value_counts.index), autopct="%0.2f", startangle=140)
    plt.legend(label_encoder.inverse_transform(value_counts.index), loc="best")
    plt.axis('equal')
    plt.title("Distribution of Sleep Disorder Categories")
    plt.show()  # This will show all the plots together 
    sleep_df.to_csv('cleaned_sleep_data.csv')
    

def splitting():
    global X_train, X_test, y_train, y_test
    text_widget.delete(1.0,  END)
    sleep_df=pd.read_csv('cleaned_sleep_data.csv')
    sleep_df.drop(columns=['Unnamed: 0'], inplace=True)
    X = sleep_df.drop('Sleep Disorder', axis=1)
    y = sleep_df['Sleep Disorder']
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    total_size = len(sleep_df)
    text_widget.insert(END, "Dataset splitting completed:\n")
    text_widget.insert(END, f"Total dataset size: {total_size} samples\n")
    text_widget.insert(END, f"Training Set(80%): {len(X_train)} samples\n")
    text_widget.insert(END, f"Testing Set(20%): {len(X_test)} samples\n")

    
# Function to calculate metrics and store results
def calculateMetrics(algorithm_name, predict, testY, labels):
    p = precision_score(testY, predict, average='macro') * 100
    r = recall_score(testY, predict, average='macro') * 100
    f = f1_score(testY, predict, average='macro') * 100
    a = accuracy_score(testY, predict) * 100
    
    # Assuming you want to append the results at the end of the widget
    text_widget.delete(1.0, END)
    text_widget.insert(END, f"{algorithm_name} Accuracy: {a:.2f}%\n")
    text_widget.insert(END, f"{algorithm_name} Precision: {p:.2f}%\n")
    text_widget.insert(END, f"{algorithm_name} Recall: {r:.2f}%\n")
    text_widget.insert(END, f"{algorithm_name} F-Measure: {f:.2f}%\n")

    
    # Append metrics to global lists
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    algorithm.append(algorithm_name)

    # Confusion Matrix Plot
    conf_matrix = confusion_matrix(testY, predict)
    plt.figure(figsize=(5, 5))
    ax = sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, cmap="viridis", fmt="g")
    ax.set_ylim([0, len(labels)])
    plt.title(f"{algorithm_name} Confusion Matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()

# Define classifier functions
def knn():
    global X_train, X_test, y_train, y_test
    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    calculateMetrics('KNN', y_pred, y_test, labels)

def svm():
    global X_train, X_test, y_train, y_test
    clf = SVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    calculateMetrics('SVM', y_pred, y_test, labels)

def random_forest():
    global X_train, X_test, y_train, y_test
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    calculateMetrics('Random Forest', y_pred, y_test, labels)

def decisionTree():
    global X_train, X_test, y_train, y_test
    clf = DecisionTreeClassifier(criterion="entropy", random_state=100,
        max_depth=3, min_samples_leaf=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    calculateMetrics('Decision Tree', y_pred, y_test, labels)

def ann_model():
    global X_train, X_test, y_train, y_test
##    input_dim = X_train.shape[1] 
##    ANNmodel = Sequential()
##    # Input layer and first hidden layer
##    ANNmodel.add(Dense(units=64, activation='relu', input_dim=input_dim))
##    # Second hidden layer
##    ANNmodel.add(Dense(units=32, activation='relu'))
##    # Output layer for multi-class classification
##    ANNmodel.add(Dense(units=3, activation='softmax'))  # Three classes: Sleep Apnea, Insomnia, None
##    # Compile the model
##    ANNmodel.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    ANNmodel = load_model('sleep_disorder_ann_model.h5')
    ANNmodel.fit(X_train, y_train, epochs=100, batch_size=15)
    y_pred = np.argmax(ANNmodel.predict(X_test), axis=1)
    calculateMetrics('ANN', y_pred, y_test, labels)
##    ANNmodel.save('sleep_disorder_ann_model.h5')
    return ANNmodel


# Function to plot comparison graph
def graph():
    text_widget.delete(1.0, END)
    # Create a DataFrame for the metrics of all models
    df = pd.DataFrame([
        ['KNN', 'Precision', precision[0]],['KNN', 'Recall', recall[0]],['KNN', 'F1 Score', fscore[0]],['KNN', 'Accuracy', accuracy[0]],
        ['SVM', 'Precision', precision[1]],['SVM', 'Recall', recall[1]],['SVM', 'F1 Score', fscore[1]],['SVM', 'Accuracy', accuracy[1]],
        ['Decision Tree', 'Precision', precision[2]],['Decision Tree', 'Recall', recall[2]],['Decision Tree', 'F1 Score', fscore[2]],['Decision Tree', 'Accuracy', accuracy[2]],
        ['Random Forest', 'Precision', precision[3]],['Random Forest', 'Recall', recall[3]],['Random Forest', 'F1 Score', fscore[3]],['Random Forest', 'Accuracy', accuracy[3]],
        ['ANN Algorithm', 'Precision', precision[4]],['ANN Algorithm', 'Recall', recall[4]],['ANN Algorithm', 'F1 Score', fscore[4]],['ANN Algorithm', 'Accuracy', accuracy[4]]
    ], columns=['Parameters', 'Algorithms', 'Value'])
    
    # Pivot the dataframe for easier plotting
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.title("All Algorithms Performance Comparison")
    plt.ylabel('Scores (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def prediction():
    text_widget.delete(1.0, END)
    csv_file_path = filedialog.askopenfilename(initialdir="Dataset")
    input_data = pd.read_csv(csv_file_path)
    features = ['Age', 'Gender', 'Sleep Duration', 'Quality of Sleep', 
                'Physical Activity Level', 'Stress Level', 'BMI Category',
                'Systolic', 'Diastolic', 'Heart Rate', 'Daily Steps', 'Occupation']
    label_encoder = LabelEncoder()
    if 'Gender' in input_data:
        input_data['Gender'] = label_encoder.fit_transform(input_data['Gender'])
    if 'Occupation' in input_data:
        input_data['Occupation'] = label_encoder.fit_transform(input_data['Occupation'])
    if 'BMI Category' in input_data:
        input_data['BMI Category'] = label_encoder.fit_transform(input_data['BMI Category'])
    if 'Blood Pressure' in input_data:
        input_data[['Systolic', 'Diastolic']] = input_data['Blood Pressure'].str.split('/', expand=True)
        input_data[['Systolic', 'Diastolic']] = input_data[['Systolic', 'Diastolic']].astype(int)
        input_data.drop(columns=['Blood Pressure'], inplace=True)  # Drop the original Blood Pressure column
    input_data = input_data[features]
    scaler = StandardScaler()
    input_scaled = scaler.fit_transform(input_data)
    ANNmodel = load_model('sleep_disorder_ann_model.h5')
    predictions = ANNmodel.predict(input_scaled)
    class_labels = ['None', 'Sleep Apnea', 'Insomnia']
    predicted_classes = [class_labels[np.argmax(prediction)] for prediction in predictions]
    text_widget.insert(END, "Prediction using test data:- ")
    for idx, (input_row, prediction) in enumerate(zip(input_data.itertuples(index=False), predicted_classes), start=1):
        input_data_str = ', '.join(map(str, input_row))
        text_widget.insert(END, f"Input Data: {input_data_str}\nSleep Disorder Predicted as: {prediction}\n")
    


root = tkinter.Tk()
root.title("Applying Machine Learning Algorithms for the Classification of Sleep Disorders")
root.geometry("1250x700")
title = Label(root, text='Applying Machine Learning Algorithms for the Classification of Sleep Disorders', font = ('times', 16, 'bold'))
title.config(bg='gray24', fg='white')             
title.config(height=3, width=120)       
title.place(x=0,y=5)

# Text Widget for output
text_widget = Text(root, height=25, width=140)
text_widget.place(x = 50, y=200)

font1 = ('times', 12, 'bold')
# 1 Upload Dataset Button
upload_button = Button(root, text="Upload Dataset", font=font1, bg = "#7DDA58", fg="white", command = uploaddataset)
upload_button.place(x = 20, y=100)

# 2 Preprocess Button
preprocess_button = Button(root, text="Preprocess Data", font=font1, bg = "#7DDA58", fg="white", command = preprocessdata)
preprocess_button.place(x = 180, y=100)

# 3 Run Splitting Button
split_button = Button(root, text="Dataset Splitting", font=font1, bg = "#7DDA58",fg="white", command = splitting)
split_button.place(x = 350, y=100)

# 4 Run KNN Button
run_knn_button = Button(root, text="Train Existing KNN", font=font1, bg = "#4741D9", fg="white",command = knn)
run_knn_button.place(x = 520, y=100)

# 5 Run SVM Button
run_svm_button = Button(root, text="Train Existing SVM", font=font1, bg = "#4741D9", fg="white", command = svm)
run_svm_button.place(x = 720, y=100)

# 6 Run Decision Tree Button
run_DT_button = Button(root, text="Train Existing Decision Tree", bg = "#4741D9", fg="white", font=font1, command = decisionTree)
run_DT_button.place(x = 920, y=100)

# 7 Run Random Forest Button
run_rf_button = Button(root, text="Train Existing Random Forest", font=font1, bg = "#4741D9", fg="white", command = random_forest)
run_rf_button.place(x = 20, y=150)

# 8 Run ANN Button
run_ann_button = Button(root, text="Train Proposed ANN Algorithn", font=font1, bg = "#060270", fg="white", command = ann_model)
run_ann_button.place(x = 280, y=150)
                       
# 9 Run Comparison Graph Button
run_graph_button = Button(root, text="Comparison Graph", bg = "#7DDA58", fg="white", font=font1, command = graph)
run_graph_button.place(x = 540, y=150)

# 10 Run Prediction Button
pred_button = Button(root, text="Sleep Disorder Prediction", font=font1, command = prediction, bg = "#FF0B0D", fg="white")
pred_button.place(x = 720, y=150)

# Run the main loop
root.config(bg='#DFC57B')
root.mainloop()








    




    
    
