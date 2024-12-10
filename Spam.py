import streamlit as st
import pickle

# Load the trained model and vectorizer
try:
    model = pickle.load(open("spam_classifier.pkl", "rb"))
    if isinstance(model, tuple):  # Handle tuple case
        st.warning("Model loaded as a tuple. Extracting the first element...")
        model = model[0]
except Exception as e:
    st.error(f"Error loading model: {e}")

try:
    cv = pickle.load(open("vectorizer.pkl", "rb"))
except Exception as e:
    st.error(f"Error loading vectorizer: {e}")
# Define the main function
def main():
    st.title("Email Spam Classification Application")
    st.write("This application uses a Machine Learning model to classify emails as Spam or Not Spam. Enter the email content below and click 'Classify' to see the result.")
    st.subheader("Classification of Emails")
    user_input = st.text_area("Enter an email to classify:", height=150)
    if st.button("Classify"):
        if user_input.strip():
            try:
                data = [user_input]
                vec = cv.transform(data)
                result = model.predict(vec)
                if result[0] == 0:
                    st.success("This is Not A Spam Email")
                else:
                    st.error("This is A Spam Email")
            except Exception as e:
                st.error(f"An error occurred during classification: {e}")
        else:
            st.warning("Please enter an email to classify.")

# Run the app
if __name__ == "__main__":
    main()


