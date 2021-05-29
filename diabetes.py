""" Taller Machine Learning , Dataset : diabetes.csv """

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


def metricas_entrenamiento(model, x_train, x_test, y_train, y_test):
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


tabla_modelos = pd.DataFrame(columns=[
                             'Training Acc', 'Validation Acc', 'Test Acc', 'Recall', 'Precision', 'F1-Score', 'AUC'])


def tabla_metricas_modelos(strmodel, acc_train, acc_validation, acc_test, matrixconfusion, AUC):
    TP = matrixconfusion[0, 0]
    FP = matrixconfusion[0, 1]
    FN = matrixconfusion[1, 0]
    # TN = matrixconfusion[1, 1]
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1_score = 2 * ((Precision * Recall) / (Precision + Recall))

    tabla_modelos.loc[strmodel] = [acc_train.round(2)] + [acc_validation.round(2)] + [acc_test.round(
        2)] + [Recall.round(2)] + [Precision.round(2)] + [F1_score.round(2)] + [AUC.round(2)]

    return print(tabla_modelos.sort_values(by="AUC", ascending=False))


def matrix_confusion_AUC(model, x_test, y_test, y_pred):
    matriz_confusion = confusion_matrix(y_test, y_pred)
    probs = model.predict_proba(x_test)
    probs = probs[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    AUC = roc_auc_score(y_test, probs)
    return matriz_confusion, AUC, fpr, tpr


def curvas_roc_matrix(fpr, tpr, fpr2, tpr2, fpr3, tpr3, fpr4, tpr4, fpr5, tpr5):
    plt.figure(0).clf()
    plt.plot(fpr, tpr, color='red', label="Linear Regression")
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.plot(fpr2, tpr2, color='blue', label="DecisionTree")
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.plot(fpr3, tpr3, color='orange', label="Kneighbors")
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.plot(fpr4, tpr4, color='black', label="RandomForest")
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.plot(fpr5, tpr5, color='green', label="ExtraTrees")
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')

    plt.legend()


def matrices_confusion(matrix_confusion, matrix_confusion2, matrix_confusion3, matrix_confusion4, matrix_confusion5):
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(3, 2, 1)
    sns.heatmap(data=matrix_confusion, ax=ax1).set_title('LogisticRegression')

    ax2 = fig.add_subplot(3, 2, 2)
    sns.heatmap(data=matrix_confusion2, ax=ax2).set_title(
        'DecisionTreeClassifier')

    ax3 = fig.add_subplot(3, 2, 3)
    sns.heatmap(data=matrix_confusion3, ax=ax3).set_title(
        'KNeighborsClassifier')

    ax4 = fig.add_subplot(3, 2, 4)
    sns.heatmap(data=matrix_confusion4, ax=ax4).set_title('RandomForest')

    ax5 = fig.add_subplot(3, 2, 5)
    sns.heatmap(data=matrix_confusion5, ax=ax5).set_title('ExtraTrees')


def mostrar_matrices(str_model, AUC, acc_validation, acc_test, y_test, y_pred, matriz_confusion):
    print('-' * 50 + '\n')
    print(str.upper(str_model))
    print('\n')
    print(matriz_confusion, '\n')
    print(f'Accuracy de validación: {acc_validation} ')
    print(f'Accuracy de test: {acc_test} ')
    print(classification_report(y_test, y_pred))
    print(f'AUC: {AUC} ')


url = 'diabetes.csv'
data = pd.read_csv(url)

rangos = [20, 35, 50, 65, 82]
nombres = ['1', '2', '3', '4']
data.Age = pd.cut(data.Age, rangos, labels=nombres)

x = np.array(data.drop(['Outcome'], 1))
y = np.array(data.Outcome)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = LogisticRegression()
model, acc_validation, acc_test, y_pred, acc_train = metricas_entrenamiento(
    model, x_train, x_test, y_train, y_test)
matrixconfusion1, AUC, fpr1, tpr1 = matrix_confusion_AUC(
    model, x_test, y_test, y_pred)
mostrar_matrices('Linear Regression', AUC, acc_validation,
                 acc_test, y_test, y_pred, matrixconfusion1)
tabla_metricas_modelos('Linear Regression', acc_train,
                       acc_validation, acc_test, matrixconfusion1, AUC)


model = DecisionTreeClassifier()
model, acc_validation, acc_test, y_pred, acc_train = metricas_entrenamiento(
    model, x_train, x_test, y_train, y_test)
matrixconfusion2, AUC, fpr2, tpr2 = matrix_confusion_AUC(
    model, x_test, y_test, y_pred)
mostrar_matrices('Decision Tree', AUC, acc_validation,
                 acc_test, y_test, y_pred, matrixconfusion2)
tabla_metricas_modelos('Decision Tree', acc_train,
                       acc_validation, acc_test, matrixconfusion2, AUC)


model = KNeighborsClassifier(n_neighbors=3)
model, acc_validation, acc_test, y_pred, acc_train = metricas_entrenamiento(
    model, x_train, x_test, y_train, y_test)
matrixconfusion3, AUC, fpr3, tpr3 = matrix_confusion_AUC(
    model, x_test, y_test, y_pred)
mostrar_matrices('KNeighborns', AUC, acc_validation,
                 acc_test, y_test, y_pred, matrixconfusion3)
tabla_metricas_modelos('KNeighborns', acc_train,
                       acc_validation, acc_test, matrixconfusion3, AUC)


model = RandomForestClassifier()
model, acc_validation, acc_test, y_pred, acc_train = metricas_entrenamiento(
    model, x_train, x_test, y_train, y_test)
matrixconfusion4, AUC, fpr4, tpr4 = matrix_confusion_AUC(
    model, x_test, y_test, y_pred)
mostrar_matrices('RandomForest', AUC, acc_validation,
                 acc_test, y_test, y_pred, matrixconfusion4)
tabla_metricas_modelos('RandomForest', acc_train,
                       acc_validation, acc_test, matrixconfusion4, AUC)


model = ExtraTreesClassifier(n_estimators=100, random_state=0)
model, acc_validation, acc_test, y_pred, acc_train = metricas_entrenamiento(
    model, x_train, x_test, y_train, y_test)
matrixconfusion5, AUC, fpr5, tpr5 = matrix_confusion_AUC(
    model, x_test, y_test, y_pred)
mostrar_matrices('ExtraTrees', AUC, acc_validation,
                 acc_test, y_test, y_pred, matrixconfusion5)
tabla_metricas_modelos('ExtraTrees', acc_train,
                       acc_validation, acc_test, matrixconfusion5, AUC)


matrices_confusion(matrixconfusion1, matrixconfusion2,
                   matrixconfusion3, matrixconfusion4, matrixconfusion5)

curvas_roc_matrix(fpr1, tpr1, fpr2, tpr2, fpr3, tpr3, fpr4, tpr4, fpr5, tpr5)
