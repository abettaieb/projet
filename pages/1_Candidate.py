import streamlit as st
import re


# === PROMPT CONFIG ===
AGENT_SYSTEM_PROMPTS = {
    "Interview Question Generator": {
        "prompt": (
            "You are an HR expert tasked with designing a diverse set of technical interview questions. "
            "Given a list of job requirements, generate a combination of multiple-choice (QCM) and open-ended (written) questions. "
            "Each question must test a specific knowledge area relevant to the requirements. "
            "Format MCQs as:\n"
            "Q: ...\nA) ...\nB) ...\nC) ...\nD) ...\nAnswer: X\n\n"
            "Format written questions as:\n"
            "Q: ...\nType: written\n\n"
            "Include at least 2 written questions if possible."
        ),
        "inputs": ["Job Requirements"],
        "output_description": "A set of MCQ and written questions for candidate assessment."
    }
}
from Admin import AGENT_SYSTEM_PROMPTS, call_agent

def call_agent(system_prompt, input_variables, input_values):
    from groq import Groq
    client = Groq(api_key=st.secrets["groc_api_key"])

    messages = [{"role": "system", "content": system_prompt}]
    for var in input_variables:
        messages.append({
            "role": "user",
            "content": f"{var}: {input_values[var]}"
        })

    chat_completion = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=messages,
    )
    return chat_completion.choices[0].message.content.strip()



st.set_page_config(page_title="Candidate Interview", layout="wide", initial_sidebar_state="collapsed")
st.title("🎤 Candidate Interview")

job_desc = st.text_area("Paste job requirements here:")

def parse_questions(text):
    qcm_pattern = re.compile(r"Q:\s*(.*?)\nA\)\s*(.*?)\nB\)\s*(.*?)\nC\)\s*(.*?)\nD\)\s*(.*?)\nAnswer:\s*([A-D])", re.DOTALL)
    written_pattern = re.compile(r"Q:\s*(.*?)\nType:\s*written", re.DOTALL)

    questions = []

    for m in qcm_pattern.findall(text):
        q, a, b, c, d, ans = m
        questions.append({
            "question": q.strip(),
            "options": [a.strip(), b.strip(), c.strip(), d.strip()],
            "answer": ans.strip().upper(),
            "type": "mcq"
        })

    for m in written_pattern.findall(text):
        questions.append({
            "question": m.strip(),
            "type": "written"
        })

    return questions

if st.button("Start Interview") and job_desc.strip():
    with st.spinner("Generating questions..."):
        prompt_template = AGENT_SYSTEM_PROMPTS.get("Interview Question Generator")
        try:
            response = call_agent(
                system_prompt=prompt_template["prompt"],
                input_variables=prompt_template["inputs"],
                input_values={"Job Requirements": job_desc}
            )
        except Exception as e:
            st.error(f"⚠️ Error calling the Interview Question Generator: {e}")
            st.stop()

    if not response or "Q:" not in response:
        st.error("❌ No questions generated. Try using more technical job requirements.")
    else:
        questions = parse_questions(response)
        if not questions:
            st.error("⚠️ Could not parse the generated content into questions.")
        else:
            st.markdown("### 📝 Interview Questions")

            if "answers" not in st.session_state:
                st.session_state.answers = [None] * len(questions)

            with st.form("interview_form"):
                for idx, q in enumerate(questions):
                    st.markdown(f"**Q{idx + 1}. {q['question']}**")
                    if q["type"] == "mcq":
                        st.session_state.answers[idx] = st.radio(
                            label="",
                            options=["A", "B", "C", "D"],
                            format_func=lambda x: f"{x}) {q['options'][ord(x) - 65]}",
                            key=f"q_{idx}"
                        )
                    else:
                        st.session_state.answers[idx] = st.text_area("Your answer:", key=f"q_{idx}")

                submitted = st.form_submit_button("Submit")

            if submitted:
                score = sum(
                    1 for i, q in enumerate(questions)
                    if q["type"] == "mcq" and st.session_state.answers[i] == q["answer"]
                )
                total_mcq = sum(1 for q in questions if q["type"] == "mcq")
                st.success(f"✅ Your MCQ Score: {score}/{total_mcq}")

                if any(q["type"] == "written" for q in questions):
                    st.markdown("### ✍️ Written Answers")
                    for i, q in enumerate(questions):
                        if q["type"] == "written":
                            st.markdown(f"**Q{i + 1}:** {q['question']}")
                            st.markdown(f"📝 Your answer: _{st.session_state.answers[i]}_")

else:
    st.info("👈 Paste job requirements and click Start Interview.")

