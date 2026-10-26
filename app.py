
import os
import streamlit as st
from dotenv import load_dotenv
from google import genai

load_dotenv(dotenv_path=".env")

st.set_page_config(
    page_title="Gemini 4 Argon Hackathon",
    page_icon="🤖",
    layout="wide"
)

st.title("Gemini 4 Argon Hackathon")
st.write("AI-powered reasoning and analysis assistant")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("Gemini API key not configured.")
    st.stop()

prompt = st.text_area(
    "What would you like Gemini to analyze?",
    placeholder="Enter a question or problem...",
    height=150
)

thinking_level = st.selectbox(
    "Reasoning depth",
    ["low", "medium", "high"],
    index=1
)

if st.button("Analyze with Gemini 4 Argon"):
    if not prompt.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Analyzing..."):
                with genai.Client(api_key=api_key) as client:
                    response = client.interactions.create(
                        model="gemini-4-argon",
                        input=prompt,
                        generation_config={
                            "thinking_level": thinking_level
                        }
                    )

            st.subheader("Analysis")
            st.markdown(
                response.output_text or
                "No text response returned."
            )

        except Exception as error:
            st.error(f"Gemini API error: {error}")

