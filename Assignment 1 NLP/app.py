import streamlit as st
import ollama

# Set up the web page title
st.set_page_config(page_title="Local AI Chatbot", page_icon="🤖")
st.title("🤖 Local AI with Ollama & Streamlit")

# Dropdown sidebar to choose between your models
selected_model = st.sidebar.selectbox(
    "Choose a model:",
    ["gemma4", "medgemma:4b"] # <-- Cleanly maps to your exact downloaded local tags
)

# Initialize chat history in Streamlit session memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if user_prompt := st.chat_input("Type your message here..."):
    # Display the user's message in the app window
    with st.chat_message("user"):
        st.markdown(user_prompt)
    
    # Save user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # Generate assistant response from Ollama
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Stream the text output smoothly just like ChatGPT
        response_stream = ollama.chat(
            model=selected_model,
            messages=st.session_state.messages,
            stream=True
        )
            
        for chunk in response_stream:
            full_response += chunk['message']['content']
            message_placeholder.markdown(full_response + "▌")
            
        message_placeholder.markdown(full_response)
        
    # Save assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
