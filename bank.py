""" Taller Machine Learning , Dataset : Bank-Full.csv """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from warnings import simplefilter
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report, roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


simplefilter(action='ignore', category=FutureWarning)


def metricas_de_entrenamiento(model, x_train, x_test, y_train, y_test):
    kfold = KFold(n_splits=10)
    cvscores, trainscores = [], []
    for train, test in kfold.split(x_train, y_train):
        model.fit(x_train[train], y_train[train])
        scores = model.score(x_train[test], y_train[test])
        cvscores.append(scores)
        tscores = model.score(x_train[train], y_train[train])
        trainscores.append(tscores)
    y_pred = model.predict(x_test)
    accuracy_validation = np.mean(cvscores)
    accuracy_train = np.mean(trainscores)
    accuracy_test = accuracy_score(y_pred, y_test)
    return model, accuracy_validation, accuracy_test, y_pred, accuracy_train


tabla_modelos = pd.DataFrame(
    columns=[
        'Training Acc',
        'Validation Acc',
        'Test Acc',
        'Recall',
        'Precision',
        'F1-Score',
        'AUC'
    ]
)


def tabla_metricas_modelos(
        strmodel, acc_train, acc_validation, acc_test, matrixconfusion, AUC):
    TP = matrixconfusion[0, 0]
    FP = matrixconfusion[0, 1]
    FN = matrixconfusion[1, 0]
    # TN = matrixconfusion[1, 1]
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * ((precision * recall) / (precision + recall))

    tabla_modelos.loc[strmodel] = (
        [acc_train.round(2)] +
        [acc_validation.round(2)] +
        [acc_test.round(2)] +
        [recall.round(2)] +
        [precision.round(2)] +
        [f1_score.round(2)] + [AUC.round(2)]
    )

    return print(tabla_modelos.sort_values(by="AUC", ascending=False))


def matrix_de_confusion_auc(model, x_test, y_test, y_pred):
    matriz_confusion = confusion_matrix(y_test, y_pred)
    probs = model.predict_proba(x_test)
    probs = probs[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    AUC = roc_auc_score(y_test, probs)
    return matriz_confusion, AUC, fpr, tpr, probs


def matrix_curvas_roc(
        fpr, tpr, fpr2, tpr2, fpr3, tpr3, fpr4, tpr4, fpr5, tpr5):
    plt.figure(0).clf()
    plt.plot(fpr, tpr, color='red', label="Linear Regression")
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('Falsos Positivos')
    plt.ylabel('Verdaderos Positivos')

    plt.plot(fpr2, tpr2, color='blue', label="DecisionTree")
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('Falsos Positivos')
    plt.ylabel('Verdaderos Positivos')

    plt.plot(fpr3, tpr3, color='orange', label="Kneighbors")
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('Falsos Positivos')
    plt.ylabel('Verdaderos Positivos')

    plt.plot(fpr4, tpr4, color='black', label="RandomForest")
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('Falsos Positivos')
    plt.ylabel('Verdaderos Positivos')

    plt.plot(fpr5, tpr5, color='green', label="ExtraTrees")
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('Falsos Positivos')
    plt.ylabel('Verdaderos Positivos')
    plt.title('Receiver Operating Characteristic (ROC) Curve')

    plt.legend()


def mostrar_matrices_confusion(
        matrix_1, matrix_2, matrix_3, matrix_4, matrix_5):
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(3, 2, 1)
    sns.heatmap(data=matrix_1, ax=ax1).set_title('LogisticRegression')

    ax2 = fig.add_subplot(3, 2, 2)
    sns.heatmap(data=matrix_2, ax=ax2).set_title('DecisionTreeClassifier')

    ax3 = fig.add_subplot(3, 2, 3)
    sns.heatmap(data=matrix_3, ax=ax3).set_title('KNeighborsClassifier')

    ax4 = fig.add_subplot(3, 2, 4)
    sns.heatmap(data=matrix_4, ax=ax4).set_title('RandomForest')

    ax5 = fig.add_subplot(3, 2, 5)
    sns.heatmap(data=matrix_5, ax=ax5).set_title('ExtraTrees')


def mostrar_metricas(
        str_model, AUC, acc_validation,
        acc_test, y_test, y_pred, matriz_confusion):
    print('-' * 50 + '\n')
    print(str.upper(str_model))
    print('\n')
    print(matriz_confusion, '\n')
    print(f'Accuracy de validación: {acc_validation} ')
    print(f'Accuracy de test: {acc_test} ')
    print(classification_report(y_test, y_pred))
    print(f'AUC: {AUC} ')


url = 'bank-full.csv'
data = pd.read_csv(url)

rangos = [0, 8, 15, 18, 25, 40, 60, 100]
nombres = ['1', '2', '3', '4', '5', '6', '7']
data.age = pd.cut(data.age, rangos, labels=nombres)
data.drop(['job', 'marital', 'month'], axis=1, inplace=True)

data.default.replace(['no', 'yes'], [0, 1], inplace=True)
data.housing.replace(['no', 'yes'], [0, 1], inplace=True)
data.loan.replace(['no', 'yes'], [0, 1], inplace=True)
data.y.replace(['no', 'yes'], [0, 1], inplace=True)
data.education.replace(
    ['unknown', 'primary', 'secondary', 'tertiary'],
    [0, 1, 2, 3], inplace=True)
data.contact.replace(
    ['unknown', 'cellular', 'telephone'], [0, 1, 2], inplace=True)
data.poutcome.replace(
    ['unknown', 'failure', 'other', 'success'], [0, 1, 2, 3], inplace=True)

x = np.array(data.drop(['y'], 1))
y = np.array(data.y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Modelo 1
model = LogisticRegression()
model, acc_validation, acc_test, y_pred, acc_train = metricas_de_entrenamiento(
    model, x_train, x_test, y_train, y_test)
matrixconfusion1, AUC, fpr1, tpr1, probs = matrix_de_confusion_auc(
    model, x_test, y_test, y_pred)
mostrar_metricas('Linear Regression', AUC, acc_validation,
                 acc_test, y_test, y_pred, matrixconfusion1)
tabla_metricas_modelos('Linear Regression', acc_train,
                       acc_validation, acc_test, matrixconfusion1, AUC)

# Modelo 2
model = DecisionTreeClassifier()
model, acc_validation, acc_test, y_pred, acc_train = metricas_de_entrenamiento(
    model, x_train, x_test, y_train, y_test)
matrixconfusion2, AUC, fpr2, tpr2, probs = matrix_de_confusion_auc(
    model, x_test, y_test, y_pred)
mostrar_metricas('Decision Tree', AUC, acc_validation,
                 acc_test, y_test, y_pred, matrixconfusion2)
tabla_metricas_modelos('Decision Tree', acc_train,
                       acc_validation, acc_test, matrixconfusion2, AUC)

# Modelo 3
model = KNeighborsClassifier(n_neighbors=3)
model, acc_validation, acc_test, y_pred, acc_train = metricas_de_entrenamiento(
    model, x_train, x_test, y_train, y_test)
matrixconfusion3, AUC, fpr3, tpr3, probs = matrix_de_confusion_auc(
    model, x_test, y_test, y_pred)
mostrar_metricas('KNeighborns', AUC, acc_validation,
                 acc_test, y_test, y_pred, matrixconfusion3)
tabla_metricas_modelos('KNeighborns', acc_train,
                       acc_validation, acc_test, matrixconfusion3, AUC)

# Modelo 4
model = RandomForestClassifier()
model, acc_validation, acc_test, y_pred, acc_train = metricas_de_entrenamiento(
    model, x_train, x_test, y_train, y_test)
matrixconfusion4, AUC, fpr4, tpr4, probs = matrix_de_confusion_auc(
    model, x_test, y_test, y_pred)
mostrar_metricas('RandomForest', AUC, acc_validation,
                 acc_test, y_test, y_pred, matrixconfusion4)
tabla_metricas_modelos('RandomForest', acc_train,
                       acc_validation, acc_test, matrixconfusion4, AUC)

# Modelo 5
model = ExtraTreesClassifier(n_estimators=100, random_state=0)
model, acc_validation, acc_test, y_pred, acc_train = metricas_de_entrenamiento(
    model, x_train, x_test, y_train, y_test)
matrixconfusion5, AUC, fpr5, tpr5, probs = matrix_de_confusion_auc(
    model, x_test, y_test, y_pred)
mostrar_metricas('ExtraTrees', AUC, acc_validation,
                 acc_test, y_test, y_pred, matrixconfusion5)
tabla_metricas_modelos('ExtraTrees', acc_train,
                       acc_validation, acc_test, matrixconfusion5, AUC)

# Informacion de los modelos
mostrar_matrices_confusion(matrixconfusion1, matrixconfusion2,
                           matrixconfusion3, matrixconfusion4, matrixconfusion5)
matrix_curvas_roc(fpr1, tpr1, fpr2, tpr2, fpr3, tpr3, fpr4, tpr4, fpr5, tpr5)
