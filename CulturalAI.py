import streamlit as st
import pandas as pd
import random
from faker import Faker
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
import openai
import prompts
from prompts import initial_context, append_to_context

# Streamlit page configuration
st.set_page_config(page_title='CulturaLink AI')

# Initialize Faker
fake = Faker()

# Streamlit sidebar for user inputs
with st.sidebar:
    st.title("🧠 CulturaLink AI")
    st.image("culturalink.png")
    st.sidebar.header('User Input Parameters')
    num_entries = st.sidebar.slider('Number of Entries', min_value=50, max_value=200, value=100)
    # Sidebar for User Guide and Feedback
    st.sidebar.title("User Guide")
    with st.sidebar.expander("How to Use This Chatbot"):
        st.markdown("""
        **Welcome to CulturaLink AI Chatbot!** Here's how to get started:
        - Ask questions related to global leadership effectiveness, cultural dynamics, and AI technologies.
        - Get insights about the features of CulturaLink AI and its applications.
        - For specific queries about the platform, simply type in your question.
        """)

    # Collecting User Feedback
    st.sidebar.title("Feedback")
    rating = st.sidebar.slider("Rate your experience", 1, 5, 3)
    if st.sidebar.button("Submit Rating"):
        st.sidebar.success(f"Thanks for rating us {rating} stars!")
        st.sidebar.markdown(
            "Do visit my [Github Repository](https://github.com/MohamedFarhun/CulturaLinkAI)"
        )

def generate_data(num_entries):
    data = {
        'Decision-Making Efficiency (days)': [random.randint(1, 60) for _ in range(num_entries)],
        'Team Engagement (score)': [random.uniform(1, 5) for _ in range(num_entries)],
        'Communication Effectiveness (score)': [random.uniform(1, 5) for _ in range(num_entries)],
        'Cultural Competency Metrics (score)': [random.uniform(1, 5) for _ in range(num_entries)],
        'Innovation Index (score)': [random.uniform(1, 5) for _ in range(num_entries)],
        'Workforce Analytics (score)': [random.uniform(1, 5) for _ in range(num_entries)],
        'Leadership Development Tracking (score)': [random.uniform(1, 5) for _ in range(num_entries)],
        'Employee Well-being Index (score)': [random.uniform(1, 5) for _ in range(num_entries)],
        'Collaboration Network Analysis (score)': [random.uniform(1, 5) for _ in range(num_entries)],
        'Skill Gap Analysis (score)': [random.uniform(1, 5) for _ in range(num_entries)],
        'Feedback and Recognition System (score)': [random.uniform(1, 5) for _ in range(num_entries)],
        'AI-Powered Decision Support (score)': [random.uniform(1, 5) for _ in range(num_entries)],
        'Global Trend Analysis (score)': [random.uniform(1, 5) for _ in range(num_entries)]
    }
    data['Team Leader'] = [fake.name() for _ in range(num_entries)]
    return pd.DataFrame(data)

def statistical_overview(df):
    return df.describe()

def logistic_regression_analysis(df):
    global model, scaler
    df['Effective Leader'] = df.iloc[:, 0:13].mean(axis=1) > 3.5
    X = df.iloc[:, 0:13]  # all score columns
    y = df['Effective Leader']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    # Returning the necessary values
    return X_train, X_test, y_train, y_test, y_pred

def predict_random_sample(model, scaler, features):
    random_sample = {feature: random.uniform(1, 5) if 'score' in feature else random.randint(1, 60)
                     for feature in features}
    sample_df = pd.DataFrame([random_sample])
    sample_scaled = scaler.transform(sample_df)
    prediction = model.predict(sample_scaled)
    prediction_proba = model.predict_proba(sample_scaled)
    prediction_result = 'Effective' if prediction[0] else 'Ineffective'
    probability_effective = prediction_proba[0][1]
    return sample_df, prediction_result, probability_effective

def format_classification_report(y_test, y_pred):
    report = classification_report(y_test, y_pred, output_dict=True)
    return pd.DataFrame(report).transpose()

def plot_confusion_matrix(conf_matrix):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    return plt

# Main app
st.title('👨🏻‍💻 CulturaLink AI')
st.write('This app generates and displays leadership-related data.')

# Data Generation
df = generate_data(num_entries)

# Displaying the data
st.header('Generated Data (First 5 Rows)')
st.write(df.head())  # Display only the first 5 rows

# Statistical Overview
st.header('Statistical Overview')
st.table(statistical_overview(df))

# Data Visualization and Statistical Overview
st.header('Data Visualization and Statistical Overview')

# Placeholder for visualization and statistical analysis
# Example: Histogram
st.subheader('Histogram')
selected_column = st.selectbox('Select Column for Histogram', df.columns)
plt.figure(figsize=(10, 4))
sns.histplot(df[selected_column], kde=True)
st.pyplot(plt)

# Logistic Regression Analysis
X_train, X_test, y_train, y_test, y_pred = logistic_regression_analysis(df)

# Function to predict effectiveness based on user input
def predict_effectiveness(user_input):
    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)
    return 'Effective' if prediction[0] else 'Ineffective', prediction_proba[0][1]

# Initialize a session state to store the history of user inputs and predictions
if 'input_history' not in st.session_state:
    st.session_state['input_history'] = []

# User Input for Prediction
st.header("User Input for Prediction")
# Add a text input for the name
user_name = st.text_input("Enter your name", "")
# Assuming each pair of features to be placed in the same line
input_pairs = [('Decision-Making Efficiency (days)', 'Team Engagement (score)'), 
               ('Communication Effectiveness (score)', 'Cultural Competency Metrics (score)'),
               ('Innovation Index (score)', 'Workforce Analytics (score)'),
               ('Leadership Development Tracking (score)', 'Employee Well-being Index (score)'),
               ('Collaboration Network Analysis (score)', 'Skill Gap Analysis (score)'),
               ('Feedback and Recognition System (score)', 'AI-Powered Decision Support (score)'),
              ]
# If you have an odd number of features, handle the last feature separately
last_feature = 'Global Trend Analysis (score)'  # Replace with your actual last feature name
user_input = {}
for feature1, feature2 in input_pairs:
    col1, col2 = st.columns(2)  # Create two columns
    with col1:
        if 'score' in feature1:
            user_input[feature1] = st.number_input(f"Enter {feature1}", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
        else:
            user_input[feature1] = st.number_input(f"Enter {feature1}", min_value=1, max_value=60, value=30, step=1)
    with col2:
        if 'score' in feature2:
            user_input[feature2] = st.number_input(f"Enter {feature2}", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
        else:
            user_input[feature2] = st.number_input(f"Enter {feature2}", min_value=1, max_value=60, value=30, step=1)

# Handle the last feature if you have an odd number of features
if len(df.columns) % 2 != 0:
    if 'score' in last_feature:
        user_input[last_feature] = st.number_input(f"Enter {last_feature}", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
    else:
        user_input[last_feature] = st.number_input(f"Enter {last_feature}", min_value=1, max_value=60, value=30, step=1)

if st.button('Predict Effectiveness'):
    if user_name:  # Check if the user has entered a name
        prediction_result, probability_effective = predict_effectiveness(user_input)
        # Append input data along with prediction and probability to the history
        st.session_state['input_history'].append(
            {'Name': user_name, **user_input, 'Prediction': prediction_result, 'Probability of Being Effective': probability_effective}
        )
        st.write(f"The person is predicted to be {prediction_result}")
        st.write(f"Probability of being Effective: {probability_effective:.2f}")
    else:
        st.warning("Please enter your name.")

# Display User Input History and Predictions
st.header("Input History and Predictions")
history_df = pd.DataFrame(st.session_state['input_history'])
st.write(history_df)

# Enhance the display of Logistic Regression Analysis
st.header('Logistic Regression Analysis')
st.write('Model Used: Logistic Regression')
accuracy = accuracy_score(y_test, y_pred)
st.write(f'Accuracy Score: {accuracy:.2f}')
# Formatted Classification Report
st.subheader('Classification Report')
report_df = format_classification_report(y_test, y_pred)
st.dataframe(report_df.style.format("{:.2f}"))
st.text('Confusion Matrix:')
conf_matrix = confusion_matrix(y_test, y_pred)
plt = plot_confusion_matrix(conf_matrix)
st.pyplot(plt)

# Function to display chat messages with styled boxes
def display_chat_message(role, message):
    if role == "user":
        st.markdown(f"<div style='padding: 10px; border-radius: 10px; background-color: #e1f5fe; margin-bottom: 10px;'>👤 <strong>You:</strong><br>{message}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='padding: 10px; border-radius: 10px; background-color: #f0f4c3; margin-bottom: 10px;'>🤖 <strong>CulturaLink AI:</strong><br>{message}</div>", unsafe_allow_html=True)

# CulturaLink AI Chatbot Section
st.title("🤖 CulturaLink AI Chatbot")

# Set OpenAI API Key
openai.api_key = st.secrets["openai_secret"]

# Initialize session state for chat messages
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": initial_context}]

# Chat Interface
with st.container():
    user_input = st.text_input("Your question to the CulturaLink AI Chatbot:")
        # Display sample questions and handle click
    if st.button("Send") and user_input:
        # Append user message with context
        messages_with_context = [{"role": "system", "content": initial_context}]
        messages_with_context += st.session_state.messages
        messages_with_context.append({"role": "user", "content": user_input})

        # Generate response from OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages_with_context
        )
        bot_response = response.choices[0].message["content"]
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

        # Display the latest user message and bot response
        display_chat_message("user", user_input)
        display_chat_message("assistant", bot_response)
        
    st.subheader("Sample Questions")
    sample_questions = [
        "How does CulturaLink AI enhance global leadership?",
        "What technologies are used in CulturaLink AI for language translation?",
        "Can CulturaLink AI integrate with existing HR systems?"
    ]
    for question in sample_questions:
        if st.button(question):
            # Simulate sending this question to the chatbot
            st.session_state.messages.append({"role": "user", "content": question})

            # Generate response from OpenAI
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=st.session_state.messages
            )
            bot_response = response.choices[0].message["content"]
            st.session_state.messages.append({"role": "assistant", "content": bot_response})

            # Display the question and bot response
            display_chat_message("user", question)
            display_chat_message("assistant", bot_response)
