# MediConnect

MediConnect is a Streamlit hospital appointment assistant. It displays doctor schedules, collects appointment request details, and offers a chat interface for general appointment enquiries.

## Features

- View doctors, specialties, visiting days, available time slots, and the hospital phone number.
- Prepare an appointment request by entering the patient's name and mobile number, choosing a doctor and date/time, and optionally adding a reason for the visit.
- Open WhatsApp with the request details pre-filled so the patient can review and send them to the hospital.
- Ask appointment-related questions in a chat powered by a Hugging Face model through LangChain.
- Adjust chat response temperature and maximum response length in the sidebar.
- Start a new chat conversation without clearing the appointment request.

## Requirements

- Python
- A Hugging Face access token to use the chat feature

The doctor list, schedules, contact number, and model ID are currently defined in `app.py`.

## Setup

Open PowerShell in the project directory and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create a `.env` file in the project root and add your Hugging Face token:

```text
HF_TOKEN=your_hugging_face_access_token
```

Keep the token private. Do not commit your `.env` file or share its contents.

## Run the app

From the project directory, with the virtual environment activated:

```powershell
streamlit run app.py
```

Streamlit will print a local URL in the terminal, typically `http://localhost:8501`.

## Using the app

1. Review doctor availability in **Doctors & schedules**.
2. Fill out **Book an appointment** and submit the form.
3. Select **Send booking to hospital WhatsApp** to open WhatsApp with the request pre-filled. The request is not sent until you send it in WhatsApp.
4. Enter a question in the chat. Chat responses require a valid `HF_TOKEN`; the appointment form and schedule can be used without it.

## Important notes

- Appointment requests are held in the current Streamlit session and are not saved to a database or sent automatically.
- Doctor schedules and contact details are sample values maintained in `app.py`. Confirm them with the hospital before relying on them.
- The chat is for general appointment guidance, not diagnosis, prescriptions, or treatment instructions. For an emergency, contact local emergency services or go to the nearest emergency department.
