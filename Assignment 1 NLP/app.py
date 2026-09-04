import streamlit as st
import ollama

# --- Page Setup ---
st.set_page_config(page_title="Cebuano MedAI", page_icon="🏥", layout="centered")
st.title("🏥 Cebuano Healthcare Assistant")
st.markdown("Pangutan-a kini nga AI bahin sa imong panglawas gamit ang **Cebuano**.")

# --- Define Your Exact Local Models ---
TRANSLATION_MODEL = "gemma4"
MEDICAL_MODEL = "medgemma:4b"

# --- Core Translation & Clinical Logic ---

def translate_ceb_to_eng(ceb_text):
    """Uses Gemma 4 to translate the Cebuano query into English."""
    prompt = f"Translate this Cebuano text directly into plain English. Only return the translation, nothing else:\n\n\"{ceb_text}\""
    response = ollama.generate(model=TRANSLATION_MODEL, prompt=prompt)
    return response['response'].strip()

def get_medical_response(eng_query):
    """Uses MedGemma:4b to process clinical context and generate an English response."""
    system_prompt = "You are a professional, empathetic, and knowledgeable clinical medical assistant. Provide accurate medical insights based on standard medical guidelines."
    response = ollama.generate(model=MEDICAL_MODEL, system=system_prompt, prompt=eng_query)
    return response['response'].strip()

def translate_eng_to_ceb(eng_text):
    """Uses Gemma 4 to translate the medical English response back to fluent Cebuano."""
    prompt = f"Translate the following medical advice accurately and naturally into fluent Cebuano. Maintain an empathetic tone. Only return the Cebuano translation:\n\n\"{eng_text}\""
    response = ollama.generate(model=TRANSLATION_MODEL, prompt=prompt)
    return response['response'].strip()

# --- Streamlit Chat History Setup ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat history (in Cebuano)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Handle Chat Input ---
if user_ceb_input := st.chat_input("Isulat ang imong pangutana dinhi (e.g., Unsaon pag-ayo sa hilanat?)..."):
    
    # 1. Display user input immediately in the GUI
    with st.chat_message("user"):
        st.markdown(user_ceb_input)
    st.session_state.messages.append({"role": "user", "content": user_ceb_input})

    # 2. Run the pipeline with a clean visual loader
    with st.chat_message("assistant"):
        with st.status("Nagproseso sa imong pangutana (Processing)...", expanded=True) as status:
            
            # Step A: Translate Cebuano Input -> English
            status.write("🔄 Naghubad sa English (Gemma 4)...")
            english_query = translate_ceb_to_eng(user_ceb_input)
            
            # Step B: Get Clinical Insight -> MedGemma:4b
            status.write("🩺 Nagkonsulta sa MedGemma...")
            english_medical_reply = get_medical_response(english_query)
            
            # Step C: Translate English Reply -> Cebuano Output
            status.write("🔄 Naghubad og balik sa Cebuano (Gemma 4)...")
            cebuano_final_reply = translate_eng_to_ceb(english_medical_reply)
            
            status.update(label="Nahuman na! (Complete)", state="complete", expanded=False)
        
        # 3. Display the final Cebuano response to the user
        st.markdown(cebuano_final_reply)
        st.session_state.messages.append({"role": "assistant", "content": cebuano_final_reply})
        
        # 4. Expandable debug logs for evaluation
        with st.expander("🔍 Tan-awa ang Translation Log (English Logs)"):
            st.caption(f"**Translated English Query:** {english_query}")
            st.caption(f"**Raw MedGemma English Advice:** {english_medical_reply}")

# --- Medical Disclaimer ---
st.markdown("---")
st.caption("""
    ⚠️ **Disclaimer:** Kini nga aplikasyon alang lamang sa impormasyon ug edukasyon. Dili kini kapuli sa propesyonal nga tambag sa doktor, diagnosis, o pagtambal. Pakunsolta kanunay sa imong doktor alang sa bisan unsang medikal nga kabalaka.
""")
