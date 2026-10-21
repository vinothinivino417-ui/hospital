"""Hospital appointment enquiry chatbot using Streamlit, LangChain, and Hugging Face."""

import os
from datetime import date
from urllib.parse import quote

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


load_dotenv()
api_key=os.getenv("HF_TOKEN")
st.write("HF_TOKEN FOUND:", bool(api_key))
MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
HOSPITAL_WHATSAPP = "917010123503"
HOSPITAL_PHONE = "7010123503"

DOCTORS = [
    {
        "name": "Dr. Ananya Sharma",
        "specialty": "General Medicine",
        "days": "Monday to Saturday",
        "slots": ["09:00 AM", "10:00 AM", "11:00 AM", "04:00 PM"],
    },
    {
        "name": "Dr. Karthik Raj",
        "specialty": "Cardiology",
        "days": "Monday, Wednesday, Friday",
        "slots": ["10:30 AM", "12:00 PM", "03:30 PM"],
    },
    {
        "name": "Dr. Meera Nair",
        "specialty": "Dermatology",
        "days": "Tuesday to Saturday",
        "slots": ["11:00 AM", "12:30 PM", "05:00 PM"],
    },
    {
        "name": "Dr. Arjun Kumar",
        "specialty": "Orthopaedics",
        "days": "Monday to Friday",
        "slots": ["09:30 AM", "01:00 PM", "04:30 PM"],
    },
]
DOCTOR_OPTIONS = {f"{doctor['name']} — {doctor['specialty']}": doctor for doctor in DOCTORS}
SCHEDULE_CONTEXT = "\n".join(
    f"- {doctor['name']} ({doctor['specialty']}): {doctor['days']}; "
    f"slots: {', '.join(doctor['slots'])}."
    for doctor in DOCTORS
)
WELCOME_MESSAGE = (
    "Hello! I am MediConnect, your hospital appointment assistant. "
    "How can I help with your appointment today?"
)
SYSTEM_PROMPT = """You are MediConnect, a hospital appointment enquiry assistant with
the communication style of an experienced doctor. Help users with appointment
enquiries, departments, doctor-specialty selection, visiting hours, appointment
preparation, and hospital contact guidance. Be kind, short, and clear.

Do not diagnose conditions, prescribe medicines, or give treatment instructions.
For a medical emergency, tell the user to call local emergency services or visit
the nearest emergency department immediately. If hospital-specific details are
unknown, say that the user should confirm them directly with the hospital.

Current project schedule:
""" + SCHEDULE_CONTEXT


def initialise_memory() -> None:
    """Set up short-term conversation memory for this browser session."""
    st.session_state.setdefault("messages", [AIMessage(content=WELCOME_MESSAGE)])
    st.session_state.setdefault("latest_booking", None)


@st.cache_resource(show_spinner=False)
def get_chat_model(temperature: float, max_tokens: int, api_key: str) -> ChatHuggingFace:
    """Create and cache the LangChain Hugging Face model for the selected settings."""
    endpoint = HuggingFaceEndpoint(
        repo_id=MODEL_ID,
        task="text-generation",
        provider="auto",
        huggingfacehub_api_token=api_key,
        temperature=temperature,
        max_new_tokens=max_tokens,
        do_sample=temperature > 0,
        return_full_text=False,
    )
    return ChatHuggingFace(llm=endpoint)


def clear_conversation() -> None:
    st.session_state.messages = [AIMessage(content=WELCOME_MESSAGE)]


def to_text(content: object) -> str:
    """Return readable text from a LangChain model response."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


st.set_page_config(
    page_title="MediConnect | Appointment assistant",
    page_icon=":material/local_hospital:",
    layout="wide",
)

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 50% 3%, rgba(214, 221, 255, .95), transparent 36rem),
            linear-gradient(135deg, #eef1ff 0%, #f9fbff 50%, #eef7ff 100%);
    }
    [data-testid="stSidebar"] > div:first-child { background: rgba(255, 255, 255, .92); }
    .block-container { max-width: 970px; padding-top: 2.25rem; padding-bottom: 7rem; }
    .hero {
        text-align: center; padding: 1.2rem 1rem 2.1rem;
    }
    .hero h1 { color: #18245c; font-size: clamp(2rem, 4vw, 3.3rem); margin-bottom: .4rem; }
    .hero p { color: #687299; font-size: 1.05rem; margin: 0 auto; max-width: 42rem; }
    .hero-badge {
        display: inline-block; padding: .35rem .7rem; margin-bottom: .75rem;
        color: #4739c7; background: #eeedff; border-radius: 999px; font-size: .82rem;
        font-weight: 600;
    }
    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, .8); border: 1px solid #e2e6ff;
        border-radius: 18px; padding: .45rem .85rem; margin-bottom: .7rem;
        box-shadow: 0 8px 24px rgba(55, 71, 180, .05);
    }
    [data-testid="stChatInput"] { border-radius: 16px; }
    [data-testid="stForm"] {
        background: rgba(255, 255, 255, .84); border: 1px solid #e2e6ff;
        border-radius: 18px; padding: .9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

initialise_memory()

with st.sidebar:
    st.title(":material/local_hospital: MediConnect")
    st.caption("Hospital appointment assistant")
    st.space("small")
    st.subheader("Response settings")
    temperature = st.slider(
        "Temperature",
        0.0,
        1.0,
        0.3,
        0.1,
        help="Lower values make replies more focused.",
    )
    max_tokens = st.slider(
        "Response token limit",
        64,
        1024,
        300,
        32,
        help="Maximum length of one answer.",
    )
    st.button(
        "New conversation",
        icon=":material/add_comment:",
        on_click=clear_conversation,
        width="stretch",
    )
    st.space("medium")
    st.caption("For emergencies, call local emergency services immediately.")

st.markdown(
    """
    <div class="hero">
      <div class="hero-badge">:material/health_and_safety: Appointment support</div>
      <h1>How can we help today?</h1>
      <p>Ask about hospital appointments, departments, specialist visits, or what to bring.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

booking_col, schedule_col = st.columns([1.15, 0.85], gap="large")

with schedule_col:
    with st.container(border=True):
        st.subheader("Doctors & schedules", anchor=False)
        for doctor in DOCTORS:
            st.markdown(f"**{doctor['name']}**")
            st.caption(f"{doctor['specialty']} · {doctor['days']}")
            st.caption("Available: " + " · ".join(doctor["slots"]))
        st.caption(f"Hospital contact: +91 {HOSPITAL_PHONE}")

with booking_col:
    st.subheader("Book an appointment", anchor=False)
    with st.form("appointment_form"):
        patient_name = st.text_input("Patient name", placeholder="Enter full name")
        patient_phone = st.text_input("Mobile number", placeholder="Enter mobile number")
        doctor_label = st.selectbox("Choose a doctor", list(DOCTOR_OPTIONS))
        selected_doctor = DOCTOR_OPTIONS[doctor_label]
        appointment_date = st.date_input(
            "Appointment date",
            min_value=date.today(),
            format="DD/MM/YYYY",
        )
        appointment_time = st.selectbox("Available time", selected_doctor["slots"])
        reason = st.text_input("Reason for visit", placeholder="Optional")
        book_now = st.form_submit_button(
            "Book appointment",
            icon=":material/calendar_add_on:",
            type="primary",
            width="stretch",
        )

    if book_now:
        if not patient_name.strip() or not patient_phone.strip():
            st.warning("Please enter the patient name and mobile number.")
        else:
            booking = {
                "name": patient_name.strip(),
                "phone": patient_phone.strip(),
                "doctor": selected_doctor,
                "date": appointment_date.strftime("%d %B %Y"),
                "time": appointment_time,
                "reason": reason.strip() or "Not specified",
            }
            st.session_state.latest_booking = booking

    if st.session_state.latest_booking:
        booking = st.session_state.latest_booking
        confirmation = (
            "New appointment enquiry\n"
            f"Patient: {booking['name']}\n"
            f"Mobile: {booking['phone']}\n"
            f"Doctor: {booking['doctor']['name']} ({booking['doctor']['specialty']})\n"
            f"Date: {booking['date']}\n"
            f"Time: {booking['time']}\n"
            f"Reason: {booking['reason']}"
        )
        whatsapp_url = f"https://wa.me/{HOSPITAL_WHATSAPP}?text={quote(confirmation)}"
        st.success(
            f"Appointment request saved for {booking['date']} at {booking['time']}."
        )
        st.link_button(
            "Send booking to hospital WhatsApp",
            whatsapp_url,
            icon=":material/chat:",
            type="primary",
            width="stretch",
        )
        st.caption("WhatsApp opens with the booking message. Tap Send to notify the hospital.")

for message in st.session_state.messages:
    role = "assistant" if isinstance(message, AIMessage) else "user"
    avatar = ":material/local_hospital:" if role == "assistant" else ":material/person:"
    with st.chat_message(role, avatar=avatar):
        st.write(message.content)

prompt = st.chat_input("Message MediConnect...", submit_mode="disable")

if prompt:
    st.session_state.messages.append(HumanMessage(content=prompt))

    with st.chat_message("user", avatar=":material/person:"):
        st.write(prompt)

    with st.chat_message("assistant", avatar=":material/local_hospital:"):
        api_key = os.getenv("HF_TOKEN")
        if not api_key or api_key == "hf_your_hugging_face_api_key_here":
            st.error("Add your Hugging Face API key to the .env file, then restart the app.")
        else:
            try:
                with st.spinner("Preparing a helpful answer..."):
                    model = get_chat_model(temperature, max_tokens, api_key)
                    response = model.invoke(
                        [SystemMessage(content=SYSTEM_PROMPT), *st.session_state.messages]
                    )
                    answer = to_text(response.content)

                st.write(answer)
                st.session_state.messages.append(AIMessage(content=answer))
            except Exception as error:
                st.error(f"Unable to get a response: {error}")

