from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report

irisData = load_iris()

X = irisData.data
y = irisData.target

# Split data into training (70%) and testing (30%)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y,
    test_size = 0.3,
    random_state = 42
)
# Split training data into training (70%) and validation (30%)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size = 0.3,
    random_state = 42
)

print(len(y_train))

def get_best_para(X_train, X_val, y_train, y_val):
    C_range = [2.0 ** i for i in range(-2,13,1)]
    best_score = -1
    best_c = -10
    for C in C_range:
        svc =SVC(kernel = 'linear', C=C)
        
        model = OneVsOneClassifier(svc)
        model.fit(X_train, y_train)
        val_score = model.score(X_val, y_val)
        print(f"parameter: {C}")
        print(f"Validation Accuracy: {val_score:.4f}")
        
        if val_score > best_score:
            best_score = val_score
            best_c = C
            
    print(f"Best Parameter: {best_c}")
    print(F"Validation Accuracy: {best_score:.4f}")
    return best_c
    
def train_final_model(X_trainval, y_trainval, best_c):
    model = OneVsOneClassifier(SVC(kernel='linear', C=best_c))
    
    model.fit(X_trainval, y_trainval)
    print("Final model trained on train + validation set")
    return model
    
def evaluate_model(model, X_train, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
best_c = get_best_para(X_train, X_val, y_train, y_val)
final_model = train_final_model(X_train_full,y_train_full,best_c)
evaluate_model(final_model,X_test,y_test)