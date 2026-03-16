# ============================================
# CREDIT SCORING MODEL
# ============================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# ============================================
# LOAD DATA
# ============================================

def load_data():
    df = pd.read_csv("credit_scoring_dataset.csv")
    print("Dataset Loaded:", df.shape)
    return df


# ============================================
# PREPARE DATA
# ============================================

def prepare_data(df):

    X = df.drop("Creditworthy", axis=1)
    y = df["Creditworthy"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, X.columns


# ============================================
# TRAIN MODEL
# ============================================

def train_model(X_train, y_train):

    model = LogisticRegression()
    model.fit(X_train, y_train)

    return model


# ============================================
# EVALUATE MODEL
# ============================================

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,1]

    print("\nModel Performance")
    print("----------------------------")
    print("Accuracy :", round(accuracy_score(y_test, predictions),3))
    print("Precision:", round(precision_score(y_test, predictions),3))
    print("Recall   :", round(recall_score(y_test, predictions),3))
    print("F1 Score :", round(f1_score(y_test, predictions),3))
    print("ROC AUC  :", round(roc_auc_score(y_test, probs),3))


# ============================================
# PREDICT CREDIT
# ============================================

def predict_credit(model, scaler, columns, user_values):

    user_df = pd.DataFrame([user_values], columns=columns)

    user_scaled = scaler.transform(user_df)

    prediction = model.predict(user_scaled)[0]
    probability = model.predict_proba(user_scaled)[0]

    confidence = max(probability) * 100

    print("\nPrediction Result")
    print("----------------------------")

    if prediction == 1:
        print(f"Creditworthy (Confidence: {confidence:.1f}%)")
    else:
        print(f"Not Creditworthy (Confidence: {confidence:.1f}%)")


# ============================================
# MAIN
# ============================================

def main():

    df = load_data()

    X_train, X_test, y_train, y_test, scaler, columns = prepare_data(df)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)


    choice = input("\nDo you want to check your creditworthiness? (yes/no): ").lower()

    if choice == "yes":

        print("\nEnter your financial details:")

        age = float(input("Age: "))
        income = float(input("Income: "))
        debt = float(input("Debt: "))
        payment_history = float(input("Payment History (0-100): "))
        loan_amount = float(input("Loan Amount: "))

        debt_to_income = debt / income
        loan_to_income = loan_amount / income

        user_values = [
            age,
            income,
            debt,
            payment_history,
            loan_amount,
            debt_to_income,
            loan_to_income
        ]

        predict_credit(model, scaler, columns, user_values)

    else:
        print("Program finished.")


if __name__ == "__main__":
    main()