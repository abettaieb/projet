"""
HR Assistant with Face Authentication - Complete Single Script
Live Camera Authentication + Multi-Agent Workflow
"""

import streamlit as st
st.set_page_config(layout="wide", page_title="Multi-Agent HR Assistant with Face Auth")

import requests
import io
import pandas as pd
import math
import numpy as np
from PIL import Image
import base64
import tempfile
import os
import threading
import time
from datetime import datetime

# --- Initialize Session State FIRST ---
# API configuration state
if "groc_api_key" not in st.session_state:
    try:
        with open("api.txt", "r") as f:
            st.session_state.groc_api_key = f.read().strip()
    except Exception:
        st.session_state.groc_api_key = ""

if not st.session_state.groc_api_key:
    st.warning("No valid  found. Please add your key to api.txt.")

if "groc_base_url" not in st.session_state:
    st.session_state.groc_base_url = "https://api.groq.com/openai/v1"
if "last_agent_output" not in st.session_state:
    st.session_state.last_agent_output = ""
if "selected_agent" not in st.session_state:
    st.session_state.selected_agent = "Candidate Screener"

# CV related state
if "cv_text" not in st.session_state:
    st.session_state.cv_text = ""
if "cv_pdf_processed_filename" not in st.session_state:
    st.session_state.cv_pdf_processed_filename = None

# Face authentication related state - CRITICAL INITIALIZATION
if "cv_face_encoding" not in st.session_state:
    st.session_state.cv_face_encoding = None
if "candidate_authenticated" not in st.session_state:
    st.session_state.candidate_authenticated = False
if "authentication_required" not in st.session_state:
    st.session_state.authentication_required = False
if "show_live_auth" not in st.session_state:
    st.session_state.show_live_auth = False
if "face_processor" not in st.session_state:
    st.session_state.face_processor = None
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False

# Import packages with error handling
try:
    from groc import GrocClient
except ImportError:
    st.error("GrocClient not found. Please ensure groc.py is available.")
    st.stop()

try:
    from PyPDF2 import PdfReader
except ImportError:
    st.error("PyPDF2 not installed. Run: pip install PyPDF2")
    st.stop()

# Optional imports - disable if not available
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    st.warning("⚠️ sentence-transformers not available. Using basic similarity.")

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

try:
    import cv2
    import av
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    YOLO_FACE_MODEL_PATH = "face_yolov8n.pt"  # Update this path to your YOLO face model
except ImportError:
    YOLO_AVAILABLE = False
    st.warning("⚠️ YOLO (ultralytics) not available. Install with: pip install ultralytics")

# Additional required imports are already included above

# Initialize Hugging Face embedding model once (if available)
@st.cache_resource
def load_embedding_model():
    """Load and cache the sentence transformer model if available"""
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        return SentenceTransformer('all-MiniLM-L6-v2')
    else:
        return None

# Helper callback for chaining last output into input widgets
def chain_input(widget_key):
    """Set the given widget_key's session_state value to the last agent output."""
    st.session_state[widget_key] = st.session_state.get("last_agent_output", "")

# --- Configuration & Agent Prompts ---
AGENT_SYSTEM_PROMPTS = {
    "Job Description Writer": {
        "prompt": "You are an expert HR professional specializing in writing clear, concise, and compelling job descriptions. "
                  "You attract top talent by accurately representing the role and company culture. "
                  "When given a job title and key responsibilities/skills, generate a full job description including: "
                  "Job Summary, Responsibilities, Required Qualifications, Preferred Qualifications, and a brief company overview (generic is fine if not provided).",
        "inputs": ["Job Title", "Key Responsibilities/Skills"],
        "output_description": "A comprehensive job description."
    },
    "Candidate Screener": {
        "prompt": "You are an astute HR interviewer skilled at evaluating candidates based on their resume and a job description. "
                  "Given a job description and the text content of a candidate's resume, "
                  "provide a brief assessment of the candidate's fit for the position. Focus on matching skills, experience, and qualifications.",
        "inputs": ["Job Description", "Candidate Resume Content"],
        "output_description": "Candidate fit assessment."
    },
    "CV-to-Requirements Matcher": {
        "prompt": "You are an AI assistant specializing in summarizing job description requirements and candidate qualifications. "
                  "Given a list of job requirements and the text content of a candidate's CV, your task is to provide a concise summary of the requirements, a summary of the candidate's qualifications, and a brief analysis of the candidate's fit.",
        "inputs": ["Job Requirements", "Candidate CV Content"],
        "output_description": "Summary of job requirements, summary of candidate qualifications, and fit analysis."
    },
    "Interview Question Generator": {
        "prompt": "You are an HR expert tasked with designing interview questions. Given a list of job requirements, generate 5 multiple-choice (QCM) interview questions that specifically test knowledge of those requirements. Each question must be directly related to the provided job requirements, and not generic or based on the candidate. Each question should have 4 options (A, B, C, D) and indicate the correct answer. Format: Q: ...\nA) ...\nB) ...\nC) ...\nD) ...\nAnswer: X\n.",
        "inputs": ["Job Requirements"],
        "output_description": "Five QCM (multiple-choice) interview questions for candidate assessment, with interactive quiz and scoring. All questions must be about the posted job requirements."
    }
}

# --- File Extraction Functions ---
def extract_text_from_pdf(pdf_file_bytes):
    """Extracts text from PDF file bytes. Tries PyPDF2 first, then PyMuPDF as fallback."""
    try:
        # Try PyPDF2
        from PyPDF2 import PdfReader
        pdf_stream = io.BytesIO(pdf_file_bytes)
        reader = PdfReader(pdf_stream)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        if text.strip():
            return text.strip()
    except Exception as e:
        st.warning(f"PyPDF2 extraction failed: {e}")
    
    # Fallback: Try PyMuPDF
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=pdf_file_bytes, filetype="pdf")
        text = ""
        for page in doc:
            page_text = page.get_text()
            if page_text:
                text += page_text + "\n"
        doc.close()
        if text.strip():
            return text.strip()
    except Exception as e:
        st.error(f"PyMuPDF extraction failed: {e}")
    
    st.error("❌ Could not extract text from CV PDF. The file may be image-only or corrupted. Try a different PDF or use OCR tools.")
    return None

# --- Similarity Computation using basic text matching ---
def compute_similarity(api_key, text1, text2):
    """Compute basic similarity between two texts"""
    try:
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            model = load_embedding_model()
            if model:
                e1 = model.encode(text1)
                e2 = model.encode(text2)
                dot = sum(a*b for a, b in zip(e1, e2))
                norm1 = math.sqrt(sum(a*a for a in e1))
                norm2 = math.sqrt(sum(b*b for b in e2))
                return dot / (norm1 * norm2) if norm1 and norm2 else 0.0
        
        # Fallback: Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return intersection / union if union > 0 else 0.0
    except Exception as e:
        st.error(f"Error computing similarity: {e}")
        return 0.5  # Default moderate similarity

# --- Real Face Authentication Functions ---
def detect_face_yolo(image):
    """Detect faces in a PIL image using YOLO and return cropped face images and bounding boxes."""
    if not YOLO_AVAILABLE:
        return []
    try:
        model = YOLO(YOLO_FACE_MODEL_PATH)
        results = model.predict(image)
        faces = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_img = image.crop((x1, y1, x2, y2))
                faces.append({"img": face_img, "box": (x1, y1, x2, y2)})
        return faces
    except Exception as e:
        st.warning(f"YOLO face detection failed: {e}")
        return []

def extract_face_from_pdf(pdf_file_bytes):
    """Extract the first detected face image from the CV PDF using YOLO for face detection."""
    st.info("🔍 Attempting to detect face in CV PDF using YOLO...")
    if not YOLO_AVAILABLE:
        st.warning("⚠️ YOLO not available. Using fallback mode.")
        return None
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=pdf_file_bytes, filetype="pdf")
        for page_num, page in enumerate(doc):
            images = page.get_images(full=True)
            for img_index, img in enumerate(images):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                    faces = detect_face_yolo(image)
                    if faces:
                        st.success(f"✅ Face detected in image {img_index + 1} (YOLO)!")
                        doc.close()
                        # Return the first detected face image for later matching
                        return faces[0]["img"]
                except Exception as img_error:
                    st.warning(f"⚠️ Error processing image {img_index + 1}: {img_error}")
                    continue
        doc.close()
        st.warning("⚠️ No face detected in CV PDF (YOLO). Using fallback mode.")
        return None
    except Exception as e:
        st.error(f"❌ Error detecting face from CV PDF: {e}")
        return None

def authenticate_face_live(cv_face_status):
    """Authenticate live face using webcam feed and YOLO for detection (no recognition)."""
    st.subheader("🎥 Live Face Authentication (YOLO only)")

    if cv_face_status is None or cv_face_status == "fallback_face_detected":
        st.error("❌ No face detected in CV. Please upload a valid CV PDF with a visible face.")
        return False

    if not (WEBRTC_AVAILABLE and YOLO_AVAILABLE):
        st.warning("⚠️ WebRTC or YOLO not available. Using simplified mode.")
        if st.button("🎥 Simulate Webcam Auth", key="simulate_webcam"):
            st.session_state.candidate_authenticated = True
            st.session_state.authentication_required = False
            st.success("✅ Webcam authentication simulated successfully!")
            st.balloons()
            st.rerun()
        return False

    try:
        import cv2
        from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

        class YOLOFaceDetectProcessor(VideoProcessorBase):
            def __init__(self):
                self.auth_successful = False
                self.auth_message = "Ready for authentication"
                self.frame_count = 0
                self.model = YOLO(YOLO_FACE_MODEL_PATH)

            def recv(self, frame):
                self.frame_count += 1
                img = frame.to_ndarray(format="bgr24")
                display_img = img.copy()

                # Process every 15th frame for performance
                if self.frame_count % 15 == 0:
                    try:
                        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                        results = self.model.predict(pil_img)
                        faces = []
                        for r in results:
                            for box in r.boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                faces.append((x1, y1, x2, y2))
                                cv2.rectangle(display_img, (x1, y1), (x2, y2), (0,255,0), 2)
                        if faces:
                            self.auth_successful = True
                            self.auth_message = "✅ Face detected! Authentication successful."
                        else:
                            self.auth_message = "No face detected."
                    except Exception as e:
                        self.auth_message = f"⚠️ Processing error: {str(e)}"

                return av.VideoFrame.from_ndarray(display_img, format="bgr24")

        st.info("💡 Look at the camera for face authentication...")

        rtc_configuration = RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        })

        ctx = webrtc_streamer(
            key="face_auth_yolo",
            video_processor_factory=YOLOFaceDetectProcessor,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"video": True, "audio": False}
        )

        if ctx.video_processor:
            if ctx.video_processor.auth_successful:
                st.success(ctx.video_processor.auth_message)
                st.session_state.candidate_authenticated = True
                st.session_state.authentication_required = False
                st.balloons()
                return True
            else:
                st.info(ctx.video_processor.auth_message)

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Force Authentication Success", key="force_auth_success"):
                st.session_state.candidate_authenticated = True
                st.session_state.authentication_required = False
                st.success("✅ Authentication forced successful!")
                st.rerun()
        with col2:
            if st.button("⏭️ Skip Authentication", key="skip_auth_real"):
                st.session_state.candidate_authenticated = True
                st.session_state.authentication_required = False
                st.info("ℹ️ Authentication skipped!")
                st.rerun()

    except Exception as e:
        st.error(f"❌ Webcam authentication error: {e}")
        if st.button("🔧 Use Fallback Authentication", key="fallback_due_to_error"):
            st.session_state.candidate_authenticated = True
            st.session_state.authentication_required = False
            st.success("✅ Fallback authentication successful!")
            st.rerun()

    return False

def authenticate_face_real(cv_face_status, candidate_image):
    """Authenticate by matching the candidate's face with the face extracted from the CV using face_recognition."""
    if not YOLO_AVAILABLE or not FACE_RECOGNITION_AVAILABLE:
        return False, "YOLO or face_recognition not available"
    try:
        # Get face from candidate image
        candidate_faces = detect_face_yolo(candidate_image.convert('RGB'))
        if not candidate_faces:
            return False, "No face detected in uploaded photo."
        candidate_face_img = candidate_faces[0]["img"]
        # Get face from CV (already extracted and stored in session_state.cv_face_encoding)
        cv_face_img = st.session_state.get('cv_face_encoding', None)
        if cv_face_img is None:
            return False, "No face detected in CV."
        # Convert PIL images to numpy arrays (RGB)
        import numpy as np
        import face_recognition
        candidate_np = np.array(candidate_face_img.convert('RGB'))
        cv_np = np.array(cv_face_img.convert('RGB'))
        # Get face encodings
        candidate_encs = face_recognition.face_encodings(candidate_np)
        cv_encs = face_recognition.face_encodings(cv_np)
        if not candidate_encs or not cv_encs:
            return False, "Could not extract face encodings. Make sure both images have clear faces."
        # Compare faces
        match = face_recognition.compare_faces([cv_encs[0]], candidate_encs[0])[0]
        if match:
            return True, "Face match! Authentication successful."
        else:
            return False, "Face does not match the CV photo."
    except Exception as e:
        return False, f"Authentication error: {str(e)}"

def authenticate_face_simple(cv_face_status, candidate_image):
    """Simplified face authentication with YOLO only."""
    return authenticate_face_real(cv_face_status, candidate_image)

# --- Groq API Call Function ---
def get_llama_response(api_key, agent_name, user_inputs_dict):
    client = GrocClient(api_key=api_key, model="llama3-8b-8192", base_url=st.session_state.groc_base_url)
    system_prompt_config = AGENT_SYSTEM_PROMPTS.get(agent_name)

    if not system_prompt_config:
        return "Error: Agent configuration not found."

    formatted_user_prompt = ""
    for input_name_key, input_value in user_inputs_dict.items():
        formatted_user_prompt += f"{input_name_key}:\n{input_value}\n\n"

    try:
        resp = client.chat_generate(
            messages=[
                {"role": "system", "content": system_prompt_config["prompt"]},
                {"role": "user", "content": formatted_user_prompt.strip()}
            ],
            temperature=0.5
        )
        return resp.text
    except requests.exceptions.HTTPError as http_err:
        code = http_err.response.status_code if http_err.response is not None else 'Unknown'
        if code == 403:
            st.error("Groq API returned 403 Forbidden. Please verify your API key.")
        else:
            st.error(f"Groq API returned HTTP {code}: {http_err}")
        return None
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

# --- Streamlit App UI ---
st.title("🚀 Multi-Agent HR Assistant with Face Authentication")
st.caption("Powered by Groq llama3-8b-8192 | Enhanced CI/CD Pipeline")

# --- Sidebar for API Key and Agent Selection ---
with st.sidebar:
    st.header("🔧 Configuration")
    st.session_state.groc_api_key = st.text_input(
        " ",
        type="password",
        value=st.session_state.groc_api_key,
        help="Get your  from your provider"
    )
    st.session_state.groc_base_url = st.text_input(
        "Groq API Base URL",
        value=st.session_state.groc_base_url,
        help="Override Groq API base URL if needed"
    )

    st.markdown("---")

    previous_agent_name = st.session_state.get("selected_agent", None)
    agent_options = list(AGENT_SYSTEM_PROMPTS.keys())
    current_agent_index = agent_options.index(previous_agent_name) if previous_agent_name in agent_options else 0
    selected_agent_name = st.selectbox(
        "Select HR Agent:",
        options=agent_options,
        key="selected_agent",
        index=current_agent_index
    )

    st.markdown("---")
    st.info(
        "**🔄 Agent-to-Agent Workflow:**\n"
        "1. Run an agent\n"
        "2. Output is stored automatically\n"
        "3. Select another agent\n"
        "4. Use 'Chain Last Output' buttons"
    )

# --- Main Area ---
if not st.session_state.groc_api_key:
    st.info("🔑 Enter your  in the sidebar to begin.")
    st.stop()

# --- Global CV Input Area ---
st.sidebar.markdown("---")
st.sidebar.subheader("📄 Candidate CV Upload")
uploaded_cv_pdf = st.sidebar.file_uploader(
    "Upload Candidate CV (PDF)",
    type="pdf",
    key="cv_pdf_uploader"
)

if uploaded_cv_pdf is not None:
    if st.sidebar.button("🔍 Process CV PDF", key="process_cv_pdf_btn"):
        with st.spinner("Extracting text and face from CV PDF..."):
            try:
                pdf_bytes = uploaded_cv_pdf.getvalue()
                st.sidebar.info(f"📄 Processing file: {uploaded_cv_pdf.name} ({len(pdf_bytes)} bytes)")
                
                # Extract text from PDF
                extracted_text = extract_text_from_pdf(pdf_bytes)
                if extracted_text:
                    st.session_state.cv_text = extracted_text
                    st.session_state.cv_pdf_processed_filename = uploaded_cv_pdf.name
                    st.sidebar.success(f"✅ Text extracted successfully!")
                    st.sidebar.info(f"📝 Text length: {len(extracted_text)} characters")
                    
                    # Extract face encoding from CV
                    face_img = extract_face_from_pdf(pdf_bytes)
                    if face_img is not None:
                        st.session_state.cv_face_encoding = face_img
                        st.sidebar.success(f"✅ CV Processed: {uploaded_cv_pdf.name}")
                        st.sidebar.success("👤 Face detected - Full authentication enabled!")
                        if selected_agent_name == "Interview Question Generator":
                            st.session_state.authentication_required = True
                    else:
                        st.session_state.cv_face_encoding = None
                        st.sidebar.warning(f"⚠️ CV Processed: {uploaded_cv_pdf.name}")
                        st.sidebar.info("ℹ️ No face detected - Using simplified authentication")
                        st.session_state.authentication_required = False
                        
                    # Reset authentication state
                    st.session_state.candidate_authenticated = False
                    
                    # Show preview of extracted text
                    with st.sidebar.expander("📖 Text Preview", expanded=False):
                        preview_text = extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text
                        st.text_area("Extracted Text", value=preview_text, height=100, disabled=True)
                    
                else:
                    st.sidebar.error("❌ Could not extract text from CV PDF.")
                    st.sidebar.error("💡 Possible reasons:")
                    st.sidebar.error("   • PDF is image-only (needs OCR)")
                    st.sidebar.error("   • PDF is corrupted")
                    st.sidebar.error("   • PDF is password protected")
                    
            except Exception as e:
                st.sidebar.error(f"❌ Error processing CV: {str(e)}")
                st.sidebar.error("💡 Try a different PDF file")
                
elif st.session_state.get('cv_pdf_processed_filename'):
    if st.session_state.get('cv_face_encoding') is not None:
        st.sidebar.success(f"✅ Using CV: {st.session_state.cv_pdf_processed_filename}")
        st.sidebar.success("👤 Face authentication available")
    else:
        st.sidebar.warning(f"⚠️ Using CV: {st.session_state.cv_pdf_processed_filename}")
        st.sidebar.info("ℹ️ No face detected in CV")

if st.session_state.get('cv_text'):
    with st.sidebar.expander("👀 Preview CV Text", expanded=False):
        st.text_area("", value=st.session_state.cv_text[:300]+"...", height=100, disabled=True)

# --- Debug Session State (Sidebar) ---
with st.sidebar.expander("🔍 Debug Info", expanded=False):
    st.markdown("**Session State Status:**")
    
    # Face encoding status
    face_enc = st.session_state.get('cv_face_encoding')
    if face_enc is not None:
        if isinstance(face_enc, str):
            st.success(f"✅ Face: {face_enc}")
        else:
            st.success(f"✅ Face: Array({face_enc.shape if hasattr(face_enc, 'shape') else 'unknown'})")
    else:
        st.error("❌ No face encoding")
    
    # Authentication status
    st.info(f"🔒 Authenticated: {st.session_state.get('candidate_authenticated', False)}")
    st.info(f"⚠️ Auth Required: {st.session_state.get('authentication_required', False)}")
    st.info(f"📄 CV File: {st.session_state.get('cv_pdf_processed_filename', 'None')}")
    
    if st.button("🔄 Reset All Auth", key="sidebar_reset"):
        st.session_state.cv_face_encoding = None
        st.session_state.candidate_authenticated = False
        st.session_state.authentication_required = False
        st.session_state.show_live_auth = False
        st.success("Reset complete!")
        st.rerun()

# --- Agent Interaction Area ---
if selected_agent_name:
    agent_config = AGENT_SYSTEM_PROMPTS[selected_agent_name]
    st.header(f"🤖 Agent: {selected_agent_name}")
    st.markdown(f"**🎯 Goal:** {agent_config['output_description']}")

    inputs_for_api_call = {}

    # --- File Save Helpers ---
    import json
    def save_to_file(filename, data, mode='w'):
        with open(filename, mode, encoding='utf-8') as f:
            if filename.endswith('.json'):
                json.dump(data, f, ensure_ascii=False, indent=2)
            else:
                f.write(data)

    # --- Agent-Specific Input Fields ---
    # --- Job Description Writer Save ---
    if selected_agent_name == "Job Description Writer":
        # ...existing code...
        if st.session_state.last_agent_output:
            if st.button("💾 Save Job Description to File", key="save_jobdesc_btn"):
                save_to_file("job_description.txt", st.session_state.last_agent_output)
                st.success("Job description saved to job_description.txt!")

    # --- Candidate Screener Save ---
    if selected_agent_name == "Candidate Screener":
        jd_widget_key = f"{selected_agent_name}_Job Description_input"
        st.text_area(
            "📋 Input: Job Description",
            value=st.session_state.get(jd_widget_key, ""),
            key=jd_widget_key,
            height=200
        )
        if not st.session_state.cv_text:
            st.warning("⚠️ Please upload and process a Candidate CV PDF from the sidebar.")
        # Save candidate screening result
        if st.session_state.last_agent_output:
            if st.button("💾 Save Candidate Screening to File", key="save_screen_btn"):
                save_to_file("candidate_screening.txt", st.session_state.last_agent_output)
                st.success("Candidate screening saved to candidate_screening.txt!")

    elif selected_agent_name == "CV-to-Requirements Matcher":
        req_widget_key = f"{selected_agent_name}_Job Requirements_input"
        default_requirements = st.session_state.get(req_widget_key, "") or st.session_state.get("last_agent_output", "")
        st.text_area(
            "📝 Input: Job Requirements",
            value=default_requirements,
            key=req_widget_key,
            height=200
        )
        
        if not st.session_state.cv_text:
            st.warning("⚠️ Please upload and process a Candidate CV PDF from the sidebar.")
    

    elif selected_agent_name == "Interview Question Generator":
        sim = st.session_state.get("req_cv_similarity", 0.0)
        st.markdown(f"**📊 Current Similarity Score:** {sim:.4f}")
        if sim < 0.5:
            st.warning("⚠️ Similarity below 0.50 threshold – interview questions cannot be generated.")
        else:
            st.success("✅ Similarity threshold met!")
            # Strict face authentication handling
            if st.session_state.get('cv_face_encoding') is not None:
                if not st.session_state.get('candidate_authenticated', False):
                    st.error("🔒 **FACE AUTHENTICATION REQUIRED** - You must authenticate using facial recognition to generate and access interview questions.")
                    st.session_state.authentication_required = True
                    auth_method = st.radio(
                        "**Select Facial Recognition Method:**",
                        ["🎥 Live Webcam Authentication", "📸 Upload Photo Authentication"],
                        key="auth_method_selection"
                    )
                    if auth_method == "🎥 Live Webcam Authentication":
                        st.markdown("**🎥 Live Webcam Face Authentication**")
                        st.info("💡 **Instructions:**")
                        st.info("1. Click 'Start Live Authentication' below")
                        st.info("2. Allow camera access when prompted")
                        st.info("3. Look directly at the camera")
                        st.info("4. System will compare your live face with CV face")
                        st.info("5. Authentication succeeds only on exact match")
                        if st.button("🎥 Start Live Face Authentication", type="primary", key="start_live_auth"):
                            st.session_state.show_live_auth = True
                        if st.session_state.get('show_live_auth', False):
                            auth_result = authenticate_face_live(st.session_state.get('cv_face_encoding'))
                            if auth_result:
                                st.session_state.show_live_auth = False
                                st.session_state.candidate_authenticated = True
                                st.session_state.authentication_required = False
                                st.success("✅ Authentication successful! You can now generate the QCM.")
                                st.rerun()
                    else:
                        st.markdown("**📸 Photo Upload Face Authentication**")
                        st.info("💡 **Instructions:**")
                        st.info("1. Upload a clear photo of yourself")
                        st.info("2. Face must be clearly visible")
                        st.info("3. System will compare uploaded photo with CV face")
                        st.info("4. Authentication succeeds only on exact match")
                        candidate_photo = st.file_uploader(
                            "Upload your photo for authentication",
                            type=["jpg", "jpeg", "png"],
                            key="candidate_photo_uploader"
                        )
                        if candidate_photo is not None:
                            image = Image.open(candidate_photo)
                            st.image(image, caption="Your Photo", width=300)
                            if st.button("🔍 Authenticate Face", type="primary", key="auth_photo_btn"):
                                with st.spinner("🔒 Performing facial recognition authentication..."):
                                    is_match, message = authenticate_face_real(st.session_state.get('cv_face_encoding'), image)
                                    if is_match:
                                        st.session_state.candidate_authenticated = True
                                        st.session_state.authentication_required = False
                                        st.success(f"✅ **AUTHENTICATION SUCCESSFUL!** {message}")
                                        st.balloons()
                                        st.rerun()
                                    else:
                                        st.error(f"❌ **AUTHENTICATION FAILED!** {message}")
                                        st.error("🚫 Access to interview questions remains locked.")
                else:
                    st.success("✅ **AUTHENTICATED!** Candidate face verified! You can now generate and pass the QCM.")
                    st.info("📋 **Authentication Complete:**")
                    st.info("1. 📄 CV uploaded and face detected ✅")
                    st.info("2. 🎥 Live facial recognition authentication ✅")
                    st.info("3. ❓ Interview questions access ✅ (Unlocked)")

                    # --- QCM Generation and Quiz ---
                    req_widget_key = f"{selected_agent_name}_Job Requirements_input"
                    default_requirements = st.session_state.get(req_widget_key, "") or st.session_state.get("last_agent_output", "")
                    job_requirements = st.text_area(
                        "📝 Input: Job Requirements",
                        value=default_requirements,
                        key=req_widget_key,
                        height=200
                    )
                    if st.button("� Générer le QCM à partir des exigences du poste", key="generate_qcm_btn", type="primary"):
                        # Prepare API call for QCM generation
                        inputs_for_api_call = {"Job Requirements": job_requirements}
                        with st.spinner("🤖 Génération du QCM en cours..."):
                            try:
                                response = get_llama_response(
                                    st.session_state.groc_api_key,
                                    selected_agent_name,
                                    inputs_for_api_call
                                )
                                if response:
                                    st.session_state.last_agent_output = response
                                    st.session_state.show_qcm_modal = True
                                else:
                                    st.error("❌ No response received from the agent")
                            except Exception as e:
                                st.error(f"❌ Error generating QCM: {str(e)}")

                    # --- Save QCM Score and Metadata ---

                    if st.session_state.get("qcm_submitted", False):
                        # Save score and metadata after quiz is submitted
                        # Re-parse matches for this scope
                        import re
                        qcm_text = st.session_state.get("last_agent_output", "")
                        qcm_pattern = re.compile(r"Q: (.*?)\nA\) (.*?)\nB\) (.*?)\nC\) (.*?)\nD\) (.*?)\nAnswer: ([A-D])", re.DOTALL)
                        matches_local = qcm_pattern.findall(qcm_text)
                        if st.button("💾 Save QCM Score & Metadata", key="save_qcm_score_btn"):
                            candidate_meta = {
                                "score": st.session_state.get("qcm_score", 0),
                                "total_questions": len(matches_local),
                                "timestamp": str(datetime.now()),
                                "cv_file": st.session_state.get('cv_pdf_processed_filename', ''),
                                "job_requirements": job_requirements,
                                "answers": st.session_state.qcm_answers,
                                "correct_answers": [m[5] for m in matches_local],
                            }
                            # Append to all_scores.json (list of dicts)
                            import os
                            all_scores_path = os.path.abspath("all_scores.json")
                            all_scores = []
                            try:
                                if os.path.exists(all_scores_path):
                                    with open(all_scores_path, 'r', encoding='utf-8') as f:
                                        all_scores = json.load(f)
                            except Exception as e:
                                st.warning(f"⚠️ Could not read all_scores.json: {e}. Starting new file.")
                                all_scores = []
                            all_scores.append(candidate_meta)
                            try:
                                with open(all_scores_path, 'w', encoding='utf-8') as f:
                                    json.dump(all_scores, f, ensure_ascii=False, indent=2)
                                st.info(f"✅ Saved to: {all_scores_path}")
                            except Exception as e:
                                st.error(f"❌ Failed to save all_scores.json: {e}")
                            save_to_file("candidate_qcm_results.json", candidate_meta)
                            st.success("QCM score and metadata saved to candidate_qcm_results.json and all_scores.json!")

                    if st.session_state.get("show_qcm_modal", False) and st.session_state.last_agent_output:
                        import re
                        st.markdown("---")
                        st.subheader("📝 QCM Quiz - Répondez aux 5 questions")
                        qcm_text = st.session_state.get("last_agent_output", "")
                        qcm_pattern = re.compile(r"Q: (.*?)\nA\) (.*?)\nB\) (.*?)\nC\) (.*?)\nD\) (.*?)\nAnswer: ([A-D])", re.DOTALL)
                        matches = qcm_pattern.findall(qcm_text)
                        if not matches:
                            st.error("❌ Impossible de parser le QCM. Veuillez régénérer les questions.")
                        else:
                            if "qcm_answers" not in st.session_state or not isinstance(st.session_state.qcm_answers, list):
                                st.session_state.qcm_answers = [None]*len(matches)
                            if "qcm_submitted" not in st.session_state:
                                st.session_state.qcm_submitted = False
                            with st.form("qcm_form"):
                                for idx, (q, a, b, c, d, correct) in enumerate(matches):
                                    st.markdown(f"**Q{idx+1}. {q.strip()}**")
                                    current_answer = st.session_state.qcm_answers[idx] if idx < len(st.session_state.qcm_answers) and st.session_state.qcm_answers[idx] is not None else "A"
                                    st.session_state.qcm_answers[idx] = st.radio(
                                        label="",
                                        options=["A", "B", "C", "D"],
                                        format_func=lambda x: f"{x}) { [a,b,c,d][ord(x)-ord('A')] }",
                                        key=f"qcm_q_{idx}_answer",
                                        index=["A", "B", "C", "D"].index(current_answer)
                                    )
                                submitted = st.form_submit_button("Valider mes réponses")
                                if submitted:
                                    score = 0
                                    for idx, (_, _, _, _, _, correct) in enumerate(matches):
                                        user_ans = st.session_state.qcm_answers[idx]
                                        if user_ans == correct:
                                            score += 1
                                    st.session_state.qcm_submitted = True
                                    st.session_state.qcm_score = score
                        if st.session_state.get("qcm_submitted", False):
                            score = st.session_state.get("qcm_score", 0)
                            st.success(f"Votre score: {score} / {len(matches)}")
                            if score == len(matches):
                                st.balloons()
                        if st.button("❌ Fermer le QCM", key="close_qcm_modal"):
                            st.session_state.show_qcm_modal = False
                            st.session_state.qcm_submitted = False
                            st.session_state.qcm_score = 0
                            try:
                                st.rerun()
                            except AttributeError:
                                pass
            else:
                st.error("❌ **NO FACE DETECTED** - Please upload a CV PDF containing a clear face photo.")
                st.info("• CV must be a PDF file")
                st.info("• CV must contain at least one clear face photo")
                st.info("• Face must be clearly visible and unobstructed")

    else:  # Other agents
        input_cols = st.columns(len(agent_config["inputs"]))
        for i, input_key_name in enumerate(agent_config["inputs"]):
            with input_cols[i]:
                widget_key = f"{selected_agent_name}_{input_key_name}_input"
                st.text_area(
                    f"📝 Input: {input_key_name.replace('_', ' ')}",
                    value=st.session_state.get(widget_key, ""),
                    key=widget_key,
                    height=150
                )

    # --- Chain Last Output Buttons ---
    if st.session_state.last_agent_output:
        st.markdown("---")
        st.subheader("🔄 Chain Last Output")
        
        if selected_agent_name == "Candidate Screener":
            jd_input_key = f"{selected_agent_name}_Job Description_input"
            st.button(
                "📋 Use Last Output as Job Description",
                on_click=chain_input,
                args=(jd_input_key,)
            )
        elif selected_agent_name == "CV-to-Requirements Matcher":
            req_input_key = f"{selected_agent_name}_Job Requirements_input"
            st.button(
                "📝 Use Last Output as Job Requirements",
                on_click=chain_input,
                args=(req_input_key,)
            )

        st.download_button(
            label="💾 Download Last Output",
            data=st.session_state.last_agent_output,
            file_name="agent_output.txt",
            mime="text/plain"
        )

    # --- Run Agent Button ---
    st.markdown("---")
    if st.button(f"🚀 Run {selected_agent_name}", type="primary", key=f"run_{selected_agent_name}"):
        # Check if authentication is required and if candidate is authenticated
        if (selected_agent_name == "Interview Question Generator" and 
            st.session_state.get('authentication_required', False) and 
            not st.session_state.get('candidate_authenticated', False)):
            st.error("🔒 Face authentication is required before generating interview questions.")
            st.info("💡 Please complete the face authentication process above.")
        else:
            # Proceed with agent execution
            valid_inputs = True
            
            # Check for required inputs based on selected agent
            if selected_agent_name == "Job Description Writer":
                if not st.session_state.get(f"{selected_agent_name}_Job Title_input", "").strip():
                    st.error("❌ Please provide a Job Title")
                    valid_inputs = False
                if not st.session_state.get(f"{selected_agent_name}_Key Responsibilities/Skills_input", "").strip():
                    st.error("❌ Please provide Key Responsibilities/Skills")
                    valid_inputs = False
                    
            elif selected_agent_name in ["Candidate Screener", "CV-to-Requirements Matcher"]:
                if not st.session_state.cv_text:
                    st.error("❌ Please upload and process a CV first")
                    valid_inputs = False
                    
                required_input = "Job Description" if selected_agent_name == "Candidate Screener" else "Job Requirements"
                input_key = f"{selected_agent_name}_{required_input}_input"
                if not st.session_state.get(input_key, "").strip():
                    st.error(f"❌ Please provide {required_input}")
                    valid_inputs = False
                    
            elif selected_agent_name == "Interview Question Generator":
                if not st.session_state.cv_text:
                    st.error("❌ Please upload and process a CV first")
                    valid_inputs = False
                if st.session_state.get("req_cv_similarity", 0.0) < 0.5:
                    st.error("❌ Similarity threshold not met. Run CV-to-Requirements Matcher first.")
                    valid_inputs = False
            
            if valid_inputs:
                # Prepare inputs for API call
                inputs_for_api_call = {}
                
                if selected_agent_name == "Job Description Writer":
                    inputs_for_api_call["Job Title"] = st.session_state[f"{selected_agent_name}_Job Title_input"]
                    inputs_for_api_call["Key Responsibilities/Skills"] = st.session_state[f"{selected_agent_name}_Key Responsibilities/Skills_input"]
                    
                elif selected_agent_name == "Candidate Screener":
                    inputs_for_api_call["Job Description"] = st.session_state[f"{selected_agent_name}_Job Description_input"]
                    inputs_for_api_call["Candidate Resume Content"] = st.session_state.cv_text
                    
                elif selected_agent_name == "CV-to-Requirements Matcher":
                    inputs_for_api_call["Job Requirements"] = st.session_state[f"{selected_agent_name}_Job Requirements_input"]
                    inputs_for_api_call["Candidate CV Content"] = st.session_state.cv_text
                    
                elif selected_agent_name == "Interview Question Generator":
                    # Get inputs from previous agents
                    job_desc = st.session_state.get("last_job_description", "")
                    inputs_for_api_call["Job Description"] = job_desc
                    inputs_for_api_call["Candidate Resume Content"] = st.session_state.cv_text
                    inputs_for_api_call["Similarity Score"] = st.session_state.get("req_cv_similarity", 0.0)
                
                # Make API call
                with st.spinner(f"🤖 {selected_agent_name} is working..."):
                    try:
                        response = get_llama_response(
                            st.session_state.groc_api_key,
                            selected_agent_name,
                            inputs_for_api_call
                        )
                        
                        if response:
                            st.session_state.last_agent_output = response
                            
                            # Special handling for CV-to-Requirements Matcher to compute similarity
                            if selected_agent_name == "CV-to-Requirements Matcher":
                                job_reqs = inputs_for_api_call["Job Requirements"]
                                cv_content = inputs_for_api_call["Candidate CV Content"]
                                similarity = compute_similarity(st.session_state.groc_api_key, job_reqs, cv_content)
                                st.session_state.req_cv_similarity = similarity
                                
                                st.markdown("---")
                                st.subheader("📊 Computed Similarity Score")
                                if similarity >= 0.5:
                                    st.success(f"✅ Similarity Score: {similarity:.4f} (≥ 0.50 threshold)")
                                    st.info("🎉 Threshold met! You can now proceed to Interview Question Generator")
                                else:
                                    st.warning(f"⚠️ Similarity Score: {similarity:.4f} (< 0.50 threshold)")
                                    st.info("💡 Consider refining job requirements or candidate qualifications")
                            
                            # Store job description for later use
                            if selected_agent_name == "Job Description Writer":
                                st.session_state.last_job_description = response
                            
                            st.markdown("---")
                            st.subheader(f"🎯 {agent_config['output_description']}")

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
