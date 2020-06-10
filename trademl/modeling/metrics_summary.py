from sklearn.metrics import (accuracy_score, confusion_matrix, recall_score,
                             precision_score, f1_score, classification_report,accuracy_score,
                             roc_curve)
import matplotlib.pyplot as plt

    
def display_mental_model_metrics(actual, pred):
    print(classification_report(y_true=actual, y_pred=pred))
    print("Confusion Matrix")
    print(confusion_matrix(actual, pred))
    print("Accuracy")
    print(accuracy_score(actual, pred))


def clf_metrics(fitted_model, X_train, X_test, y_train, y_test, avg='binary', prefix=''):
    """
    Show main matrics from classification: accuracy, precision, recall, 
    confusion matrix.
    
    Arguments:
        fitted_model {[type]} -- [description]
    """
    predictions_train = fitted_model.predict(X_train)
    predictions_test = fitted_model.predict(X_test)
    print(f'Confusion matrix train: \n{confusion_matrix(y_train, predictions_train)}')
    print(f'Confusion matrix test: \n{confusion_matrix(y_test, predictions_test)}')
    print(f'{prefix}accuracy_train: {accuracy_score(y_train, predictions_train)}')
    print(f'{prefix}accuracy_test: {accuracy_score(y_test, predictions_test)}')
    print(f"{prefix}recall_train: {recall_score(y_train, predictions_train, average=avg)}")
    print(f"{prefix}recall_test: {recall_score(y_test, predictions_test, average=avg)}")
    print(f"{prefix}precision_train: {precision_score(y_train, predictions_train, average=avg)}")
    print(f"{prefix}precisoin_test: {precision_score(y_test, predictions_test, average=avg)}")
    print(f"{prefix}f1_train: {f1_score(y_train, predictions_train, average=avg)}")
    print(f"{prefix}f1_test: {f1_score(y_test, predictions_test, average=avg)}")
    # if binary:
    # print(f'True values: ', y_test.iloc[:10].values)
    # print(f'Predictions: ', predictions_train[:10])


def plot_roc_curve(fitted_model, X_train, X_test, y_train, y_test, suffix=''):
    """
    Show main matrics from classification: accuracy, precision, recall, 
    confusion matrix.
    
    :param fitted_model: (sklearn.model) Estimated skelarn model.
    :param X_train: (pd.DataFrame) X train set data frame.
    """
    # train set
    y_pred_rf = fitted_model.predict_proba(X_train)[:, 1]
    fpr_rf, tpr_rf, _ = roc_curve(y_train, y_pred_rf)
    
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_rf, tpr_rf, label='Train set ')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(f'ROC curve {suffix}')
    plt.legend(loc='best')
    plt.savefig('train_roc.png')
    # plt.show()
    
    # test set
    y_pred_rf = fitted_model.predict_proba(X_test)[:, 1]
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
    
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_rf, tpr_rf, label='Test set ')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(f'ROC curve {suffix}')
    plt.legend(loc='best')
    plt.savefig('test_roc.png')
    # plt.show()
    