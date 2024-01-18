# Detailed context for the chatbot about the "CulturaLink AI" project
initial_context = """
"CulturaLink AI" is an innovative AI platform developed by Mohamed Farhun M, designed to enhance global leadership effectiveness by integrating advanced AI technologies with a deep understanding of diverse cultural dynamics. This state-of-the-art solution empowers leaders with essential tools to navigate the complexities of managing a diverse, global workforceIt is not a product of OpenAI but utilizes OpenAI's GPT models for natural language processing tasks.

Key Features of CulturaLink AI:
1. Cultural Intelligence Engine: Utilizes natural language processing and machine learning to understand cultural nuances, enabling leaders to effectively interact with team members from diverse backgrounds.
2. Emotional Intelligence Analytics: Employs sentiment analysis and emotional recognition technologies to gauge team sentiment, facilitating empathetic leadership.
3. Adaptive Leadership Training: Offers personalized training modules based on individual leader profiles and specific organizational needs, leveraging AI-driven insights.
4. Predictive Workforce Analytics: Incorporates advanced predictive modeling and logistic regression techniques to proactively manage talent, identify potential leadership candidates, and forecast workforce trends.
5. AI-Mediated Communication: Breaks down language barriers using real-time translation and interpretation services, ensuring clear communication across diverse teams.
6. Ethical AI Framework: Ensures unbiased decision-making and respectful interactions, grounded in ethical AI principles and privacy considerations.
7. HR Systems Integration: Seamlessly integrates with existing HR systems, providing a unified platform for comprehensive leadership development.

The platform's development is rooted in cutting-edge AI research and best practices in leadership training. Mohamed Farhun M and his team have meticulously designed "CulturaLink AI" to address the unique challenges faced by global leaders today. The project's goal is to create a more inclusive, effective, and empathetic leadership culture in organizations worldwide.

If you have any questions about the creation, features, or applications of "CulturaLink AI," feel free to ask!
"""

# Add more details or FAQs to the initial_context
additional_context = """
FAQs for CulturaLink AI:
1. How does CulturaLink AI enhance global leadership?
2. What technologies are used in CulturaLink AI for language translation?
3. Can CulturaLink AI integrate with existing HR systems?
Feel free to ask these questions or any others you might have!
"""
initial_context += additional_context
# Function to append user messages and bot responses to the context
def append_to_context(context, user_message, bot_response):
    updated_context = context + f"\nUser: {user_message}\nBot: {bot_response}\n"
    return updated_context