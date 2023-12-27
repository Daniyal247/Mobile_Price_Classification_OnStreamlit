import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import streamlit as st

def map_price_range(prediction):
    if prediction == 0:
        return "15k - 20k"
    elif prediction == 1:
        return "25k - 35k"
    elif prediction == 2:
        return "35k - 60k"
    elif prediction == 3:
        return "60k - 1Lac"
    else:
        return "Unknown Price Range"

def main():
    st.title("Mobile Price Range Prediction App")

    # Load the dataset
    data = pd.read_csv("C:/Users/romai/Documents/daniyal/Machine Learning/project/train.csv")

    # Determine relevant columns based on some criteria (you can customize this)
    relevant_columns = ['battery_power', 'blue', 'ram', 'touch_screen', 'wifi', 'price_range']

    # Display input fields for relevant columns
    input_fields = {}
    for column in relevant_columns[:-1]:  # Exclude the target column ('price_range')
        if column in ['blue', 'wifi', 'touch_screen']:
            # Allow the user to enter "yes" or "no" and convert to 1 or 0
            input_fields[column] = st.selectbox(f"Select value for {column}:", ["yes", "no"])
            input_fields[column] = 1 if input_fields[column] == "yes" else 0
        else:
            input_fields[column] = st.text_input(f"Enter value for {column}:", "")

    # Button to trigger the action
    if st.button("Predict Price Range"):
        # Check if all input fields are filled
        if all(value for value in input_fields.values()):
            # Create a DataFrame with the input data
            input_data = pd.DataFrame({column: [float(value)] if column not in ['blue', 'wifi', 'touch_screen'] else [value] for column, value in input_fields.items()})

            # Assuming the 'target' column is the target variable
            X = data[relevant_columns[:-1]]  # Exclude the target column ('price_range')
            y = data['price_range']

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=31)

            # Create models
            logistic_model = LogisticRegression()
            tree_model = DecisionTreeClassifier()
            forest_model = RandomForestClassifier()

            # Train models
            logistic_model.fit(X_train, y_train)
            tree_model.fit(X_train, y_train)
            forest_model.fit(X_train, y_train)

            # Make predictions
            logistic_prediction = logistic_model.predict(input_data)
            tree_prediction = tree_model.predict(input_data)
            forest_prediction = forest_model.predict(input_data)

            # Map predictions to price ranges
            logistic_output = map_price_range(logistic_prediction[0])
            tree_output = map_price_range(tree_prediction[0])
            forest_output = map_price_range(forest_prediction[0])

            # Display the predicted price range
            st.success(f"Logistic Regression Prediction: {logistic_output}")
            st.success(f"Decision Tree Prediction: {tree_output}")
            st.success(f"Random Forest Prediction: {forest_output}")

            # Display classification report
            st.subheader("Classification Report")
            st.text("Logistic Regression:")
            st.text(classification_report(y_test, logistic_model.predict(X_test)))
            st.text("Decision Tree:")
            st.text(classification_report(y_test, tree_model.predict(X_test)))
            st.text("Random Forest:")
            st.text(classification_report(y_test, forest_model.predict(X_test)))

        else:
            st.error("Please fill in all input fields.")

if __name__ == "__main__":
    main()
