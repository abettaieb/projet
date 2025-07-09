
import streamlit as st
import re
import streamlit.components.v1 as components

# === Role Selector (Enhancement) ===
if "user_role" not in st.session_state:
    st.session_state.user_role = st.radio("🔐 Select your role", ["Admin", "Candidate"], key="user_role_selector")

# === Filter agent options based on role ===
if st.session_state.user_role == "Admin":
    agent_options = ["Job Description Writer", "Candidate Screener", "CV-to-Requirements Matcher"]
else:
    agent_options = ["Interview Question Generator"]

# Placeholder for selected agent
selected_agent_name = st.selectbox("🧠 Choose an Agent", agent_options)

# Simulated response input (normally this would be returned by your LLM)
response = st.text_area("📝 LLM Response Output", height=300)

# === Updated Prompt for Interview Question Generator ===
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

# === Enhanced Modal Logic for Interview Question Generator ===
if selected_agent_name == "Interview Question Generator":
    def parse_questions(text):
        qcm_pattern = re.compile(
            r"Q:\s*(.*?)\nA\)\s*(.*?)\nB\)\s*(.*?)\nC\)\s*(.*?)\nD\)\s*(.*?)\nAnswer:\s*([A-D])",
            re.DOTALL,
        )
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

    questions = parse_questions(response)

    if not questions:
        st.error("❌ Could not parse any questions. Please check the LLM output format.")
        st.write(response)
    else:
        if "show_modal" not in st.session_state:
            st.session_state.show_modal = False
        if "answers" not in st.session_state:
            st.session_state.answers = [None] * len(questions)
        if "submitted" not in st.session_state:
            st.session_state.submitted = False

        def open_modal():
            st.session_state.show_modal = True
            st.session_state.answers = [None] * len(questions)
            st.session_state.submitted = False

        def close_modal():
            st.session_state.show_modal = False

        st.button("📝 Passer l'entretien", on_click=open_modal, key="open_modal_btn")

        if st.session_state.show_modal:
            modal_style = """
            <style>
            .modal-bg {position: fixed; top:0; left:0; width:100vw; height:100vh; background:rgba(0,0,0,0.4); z-index:1000;}
            .modal {position: fixed; top:50%; left:50%; transform:translate(-50%,-50%); background:white; padding:2rem; border-radius:1rem; box-shadow:0 0 30px #0003; z-index:1001; min-width:350px; max-width:90vw; max-height:90vh; overflow:auto;}
            .modal h3 {margin-top:0;}
            .modal .close {position:absolute; top:1rem; right:1.5rem; font-size:1.5rem; cursor:pointer; color:#888;}
            </style>
            """
            components.html(modal_style, height=0)

            with st.container():
                st.markdown('<div class="modal-bg"></div>', unsafe_allow_html=True)
                st.markdown('<div class="modal">', unsafe_allow_html=True)
                st.markdown('<span class="close" onclick="window.parent.postMessage('close_modal', '*')">×</span>', unsafe_allow_html=True)
                st.markdown("<h3>📝 Entretien technique - QCM & Questions écrites</h3>", unsafe_allow_html=True)

                with st.form("interview_form"):
                    for idx, q in enumerate(questions):
                        st.markdown(f"**Q{idx+1}. {q['question']}**")
                        if q["type"] == "mcq":
                            st.radio(
                                label="",
                                options=["A", "B", "C", "D"],
                                format_func=lambda x: f"{x}) {q['options'][ord(x)-65]}",
                                key=f"answer_{idx}",
                                index=(ord(st.session_state.answers[idx])-65) if st.session_state.answers[idx] else 0
                            )
                        elif q["type"] == "written":
                            st.text_area("Votre réponse :", key=f"answer_{idx}")

                    submitted = st.form_submit_button("Valider mes réponses")
                    if submitted:
                        st.session_state.answers = [st.session_state.get(f"answer_{i}") for i in range(len(questions))]
                        st.session_state.submitted = True

                if st.session_state.submitted:
                    score = sum(
                        1 for i, q in enumerate(questions)
                        if q.get("type") == "mcq" and st.session_state.answers[i] == q["answer"]
                    )
                    total_mcq = sum(1 for q in questions if q.get("type") == "mcq")
                    st.markdown(f"<h4>Résultat : {score} / {total_mcq} corrects (MCQ)</h4>", unsafe_allow_html=True)
                    if score == total_mcq and total_mcq > 0:
                        st.balloons()

                    if any(q["type"] == "written" for q in questions):
                        st.markdown("📝 **Réponses écrites :**")
                        for i, q in enumerate(questions):
                            if q["type"] == "written":
                                st.markdown(f"**Q{i+1}**: {q['question']}")
                                st.markdown(f"✍️ Réponse: _{st.session_state.answers[i]}_")

                st.button("Fermer", on_click=close_modal, key="close_modal_btn")
                st.markdown('</div>', unsafe_allow_html=True)
