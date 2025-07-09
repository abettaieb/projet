# 🚀 HR Assistant with Face Authentication

A comprehensive HR Assistant application built with Streamlit, featuring multi-agent AI workflows and facial recognition authentication for secure interview processes.

## ✨ Features

- **Multi-Agent AI System**: Four specialized AI agents for different HR tasks
- **Face Authentication**: Secure facial recognition for candidate verification
- **PDF CV Processing**: Extract text and faces from uploaded CV PDFs
- **Interview Question Generation**: AI-powered question creation based on job requirements
- **Real-time Webcam Integration**: Live face authentication using WebRTC
- **Similarity Matching**: CV-to-job requirements matching with confidence scoring

## 🤖 AI Agents

1. **Job Description Writer**: Creates comprehensive job descriptions
2. **Candidate Screener**: Evaluates candidate fit for positions
3. **CV-to-Requirements Matcher**: Analyzes compatibility between CVs and job requirements
4. **Interview Question Generator**: Generates targeted interview questions (requires face authentication)

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **AI/ML**: Groq API (Llama 3), Sentence Transformers
- **Computer Vision**: OpenCV, face_recognition, WebRTC
- **PDF Processing**: PyPDF2, PyMuPDF
- **Deployment**: Docker support

## 📋 Prerequisites

- Python 3.11+
- Webcam (for face authentication)
- Groq API key

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/AhmedAbdelhedi899/projet_final.git
cd projet_final
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up API Key
- Add your Groq API key to `api.txt`
- Or the app will use the default key

### 4. Run the Application
```bash
streamlit run tentative_with_auth.py
```

### 5. Access the Application
Open your browser and go to: `http://localhost:8501`

## 🔐 Face Authentication

The application includes sophisticated face authentication for the Interview Question Generator:

- **CV Face Extraction**: Automatically extracts faces from uploaded CV PDFs
- **Live Webcam Authentication**: Real-time face matching with very lenient thresholds (15% similarity)
- **Photo Upload Authentication**: Alternative authentication via photo upload
- **Fallback Options**: Quick confirmation buttons for accessibility

## 📁 Project Structure

```
projet_final/
├── tentative_with_auth.py    # Main application
├── groc.py                   # Groq API client
├── api.txt                   # API key file
├── requirements.txt          # Python dependencies
├── cv_prof_pierre_bernard.pdf # Test CV file
├── image.jpg                 # Test face image
├── Dockerfile               # Container configuration
└── README.md               # This file
```

## 🐳 Docker Deployment

Build and run with Docker:

```bash
# Build the image
docker build -t hr-assistant .

# Run the container
docker run -p 8501:8501 hr-assistant
```

## 💡 Usage Guide

### Basic Workflow

1. **Start with Job Description Writer**: Create a comprehensive job description
2. **Upload Candidate CV**: Process PDF to extract text and face encoding
3. **Use Candidate Screener**: Get initial assessment of candidate fit
4. **Run CV-to-Requirements Matcher**: Calculate similarity scores
5. **Generate Interview Questions**: Requires face authentication (similarity ≥ 0.5)

### Face Authentication Steps

1. Upload a CV PDF containing a clear face photo
2. Navigate to "Interview Question Generator"
3. Choose authentication method:
   - **Live Webcam**: Real-time face detection and matching
   - **Photo Upload**: Upload a current photo for comparison
   - **Quick Confirm**: Manual confirmation option
4. System accepts minimal resemblance (15% similarity threshold)

## 🔧 Configuration

### API Configuration
- **Groq API**: Configure in `api.txt` or through the sidebar
- **Models**: Uses Llama 3-8B-8192 for text generation
- **Embeddings**: Sentence Transformers for similarity calculation

### Face Recognition Settings
- **Tolerance**: Very lenient (15% similarity required)
- **Detection Model**: HOG for speed, CNN for accuracy
- **Processing**: Every 10th frame for real-time performance

## 🛡️ Security Features

- **Face Authentication**: Prevents unauthorized access to interview questions
- **API Key Protection**: Secure API key management
- **Session Management**: Streamlit session state for user data

## 🚨 Troubleshooting

### Common Issues

1. **Face Recognition Not Working**:
   - Ensure good lighting for webcam
   - Use clear, front-facing photos
   - Try the "Quick Confirm" option as fallback

2. **CV Processing Fails**:
   - Ensure PDF contains extractable text (not image-only)
   - Try different PDF files
   - Check for password protection

3. **API Errors**:
   - Verify Groq API key is valid
   - Check internet connection
   - Monitor API rate limits

### Performance Tips

- Use well-lit environment for face authentication
- Upload clear, high-quality CV PDFs
- Ensure stable internet connection for AI processing

## 📚 Dependencies

See `requirements.txt` for complete list. Key dependencies:

- `streamlit>=1.28.0` - Web application framework
- `face_recognition>=1.3.0` - Facial recognition library
- `opencv-python>=4.8.0` - Computer vision processing
- `streamlit-webrtc>=0.47.0` - Real-time webcam integration
- `sentence-transformers>=2.2.0` - Text similarity calculation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Ahmed Abdelhedi**
- GitHub: [@AhmedAbdelhedi899](https://github.com/AhmedAbdelhedi899)

## 🙏 Acknowledgments

- Groq for fast AI inference
- Streamlit team for the excellent framework
- Face recognition library contributors
- Open source community

---

**Note**: This application is designed for educational and demonstration purposes. For production use, consider additional security measures and compliance requirements.
