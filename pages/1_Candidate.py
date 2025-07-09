import streamlit as st

st.set_page_config(page_title="Candidate Interview", layout="wide", initial_sidebar_state="collapsed")

st.title("🎤 Candidate Interview")

job_desc = st.text_area("Paste job requirements here:")

if st.button("Start Interview"):
    st.write("Here the questions will be displayed...")
    # --- QCM Modal/Popup for Interview Question Generator ---
                            if selected_agent_name == "Interview Question Generator":
                                import re
                                def parse_qcm_questions(text):
                                    # Parse QCM questions from LLM output
                                    pattern = re.compile(r"Q:\s*(.*?)\nA\)\s*(.*?)\nB\)\s*(.*?)\nC\)\s*(.*?)\nD\)\s*(.*?)\nAnswer:\s*([A-D])", re.DOTALL)
                                    matches = pattern.findall(text)
                                    questions = []
                                    for m in matches:
                                        q, a, b, c, d, ans = m
                                        questions.append({
                                            "question": q.strip(),
                                            "options": [a.strip(), b.strip(), c.strip(), d.strip()],
                                            "answer": ans.strip().upper()
                                        })
                                    return questions

                                qcm_questions = parse_qcm_questions(response)
                                if not qcm_questions or len(qcm_questions) < 1:
                                    st.error("❌ Could not parse QCM questions. Please check the LLM output format.")
                                    st.write(response)
                                else:
                                    # Modal state
                                    if "show_qcm_modal" not in st.session_state:
                                        st.session_state.show_qcm_modal = False
                                    if "qcm_answers" not in st.session_state:
                                        st.session_state.qcm_answers = [None]*len(qcm_questions)
                                    if "qcm_submitted" not in st.session_state:
                                        st.session_state.qcm_submitted = False

                                    def open_qcm_modal():
                                        st.session_state.show_qcm_modal = True
                                        st.session_state.qcm_answers = [None]*len(qcm_questions)
                                        st.session_state.qcm_submitted = False

                                    def close_qcm_modal():
                                        st.session_state.show_qcm_modal = False

                                    st.button("📝 Passer le QCM (Quiz)", on_click=open_qcm_modal, key="open_qcm_modal_btn")

                                    if st.session_state.show_qcm_modal:
                                        import streamlit.components.v1 as components
                                        # Overlay style for modal
                                        modal_style = """
                                        <style>
                                        .qcm-modal-bg {position: fixed; top:0; left:0; width:100vw; height:100vh; background:rgba(0,0,0,0.4); z-index:1000;}
                                        .qcm-modal {position: fixed; top:50%; left:50%; transform:translate(-50%,-50%); background:white; padding:2rem; border-radius:1rem; box-shadow:0 0 30px #0003; z-index:1001; min-width:350px; max-width:90vw; max-height:90vh; overflow:auto;}
                                        .qcm-modal h3 {margin-top:0;}
                                        .qcm-modal .qcm-close {position:absolute; top:1rem; right:1.5rem; font-size:1.5rem; cursor:pointer; color:#888;}
                                        </style>
                                        """
                                        components.html(modal_style, height=0)
                                        # Modal content
                                        with st.container():
                                            st.markdown('<div class="qcm-modal-bg"></div>', unsafe_allow_html=True)
                                            st.markdown('<div class="qcm-modal">', unsafe_allow_html=True)
                                            st.markdown('<span class="qcm-close" onclick="window.parent.postMessage(\'close_qcm_modal\', \'*\')">×</span>', unsafe_allow_html=True)
                                            st.markdown("<h3>📝 QCM - Quiz d'entretien</h3>", unsafe_allow_html=True)
                                            with st.form("qcm_form"):
                                                for idx, q in enumerate(qcm_questions):
                                                    st.markdown(f"**Q{idx+1}. {q['question']}**")
                                                    st.radio(
                                                        label="",
                                                        options=["A", "B", "C", "D"],
                                                        format_func=lambda x: f"{x}) {q['options'][ord(x)-65]}",
                                                        key=f"qcm_answer_{idx}",
                                                        index=(ord(st.session_state.qcm_answers[idx])-65) if st.session_state.qcm_answers[idx] else 0
                                                    )
                                                submitted = st.form_submit_button("Valider mes réponses")
                                                if submitted:
                                                    answers = [st.session_state.get(f"qcm_answer_{i}") for i in range(len(qcm_questions))]
                                                    st.session_state.qcm_answers = answers
                                                    st.session_state.qcm_submitted = True
                                            if st.session_state.qcm_submitted:
                                                score = sum(1 for i, q in enumerate(qcm_questions) if st.session_state.qcm_answers[i] == q['answer'])
                                                st.markdown(f"<h4>Résultat: {score} / {len(qcm_questions)} corrects</h4>", unsafe_allow_html=True)
                                                if score == len(qcm_questions):
                                                    st.balloons()
                                                else:
                                                    st.info("Essayez à nouveau pour améliorer votre score !")
                                            st.button("Fermer", on_click=close_qcm_modal, key="close_qcm_modal_btn")
                                            st.markdown('</div>', unsafe_allow_html=True)

                            else:
                                st.write(response)
                            st.success(f"✅ {selected_agent_name} completed successfully!")
                        else:
                            st.error("❌ No response received from the agent")
                    except Exception as e:
                        st.error(f"❌ Error running {selected_agent_name}: {str(e)}")
            else:
                st.warning("⚠️ Missing required inputs. Please provide all required fields.")
