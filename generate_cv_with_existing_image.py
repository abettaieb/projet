from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import os
from PIL import Image as PILImage

def create_cv_with_existing_image():
    """Create a CV using the existing image.jpg file"""
      # Check if image exists
    if not os.path.exists("image.jpg"):
        print("Error: image.jpg not found in current directory")
        return
    
    # Create PDF document
    doc = SimpleDocTemplate(
        "cv_prof_pierre_bernard.pdf",
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18
    )
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        spaceBefore=20,
        textColor=colors.darkblue,
        borderWidth=1,
        borderColor=colors.darkblue,
        borderPadding=5
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=6,
        alignment=TA_JUSTIFY
    )
    
    # Build the document content
    story = []
      # Header with image and basic info
    try:
        # Resize image if needed
        img = PILImage.open("image.jpg")
        img.thumbnail((150, 150), PILImage.Resampling.LANCZOS)
        img.save("temp_profile.jpg", "JPEG")
        
        # Create header table with image and contact info
        profile_img = Image("temp_profile.jpg", width=120, height=120)
        
        contact_info = Paragraph("""
        <b>Dr. Pierre BERNARD</b><br/>
        Senior Computer Science Professor & AI Research Lead<br/>
        Phone: +33 6 15 67 89 12<br/>
        Email: pierre.bernard@universite.fr<br/>
        LinkedIn: linkedin.com/in/pierre-bernard-cs<br/>
        GitHub: github.com/pierre-bernard-ai<br/>
        Address: 456 Technology Avenue, 69001 Lyon, France
        """, normal_style)
        
        header_data = [[profile_img, contact_info]]
        header_table = Table(header_data, colWidths=[2*inch, 4*inch])
        header_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),        ]))
        
        story.append(header_table)
        story.append(Spacer(1, 20))
        
    except Exception as e:
        print(f"Error processing image: {e}")
        # Fallback without image
        story.append(Paragraph("Dr. Pierre BERNARD", title_style))
        story.append(Paragraph("Senior Computer Science Professor", styles['Heading2']))
        story.append(Spacer(1, 12))
    
    # Professional Summary
    story.append(Paragraph("PROFESSIONAL PROFILE", heading_style))
    story.append(Paragraph(
        "Experienced Computer Science Professor with over 15 years in higher education and research. "
        "Specialist in Artificial Intelligence, Machine Learning, Software Engineering, and Database Systems. "
        "Recognized for innovative teaching methodologies and ability to inspire students in cutting-edge technology. "
        "Published researcher with expertise in Deep Learning, Natural Language Processing, and Computer Vision. "
        "Strong background in industry collaboration and practical application of theoretical concepts.",
        normal_style    ))
    
    # Education
    story.append(Paragraph("EDUCATION", heading_style))
    story.append(Paragraph("<b>Ph.D. in Computer Science</b> - Pierre and Marie Curie University, Paris (2015)", normal_style))
    story.append(Paragraph("Dissertation: 'Applications of Artificial Intelligence in Educational Technologies'", normal_style))
    story.append(Paragraph("Advisors: Prof. Jean Dubois & Prof. Marie Leclerc", normal_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>Master of Science in Computer Science</b> - École Normale Supérieure de Lyon (2012)", normal_style))
    story.append(Paragraph("Specialization: Machine Learning and Data Mining, GPA: 18.5/20", normal_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>Bachelor of Science in Computer Science</b> - Claude Bernard University Lyon 1 (2010)", normal_style))
    story.append(Paragraph("Magna Cum Laude, Minor in Mathematics", normal_style))
    
    # Certifications
    story.append(Paragraph("PROFESSIONAL CERTIFICATIONS", heading_style))
    story.append(Paragraph("• <b>Certified TensorFlow Developer</b> - Google (2023)", normal_style))
    story.append(Paragraph("• <b>AWS Solutions Architect Professional</b> - Amazon Web Services (2022)", normal_style))
    story.append(Paragraph("• <b>Microsoft Azure AI Engineer Associate</b> - Microsoft (2021)", normal_style))
    story.append(Paragraph("• <b>Deep Learning Specialization</b> - Stanford University/Coursera (2020)", normal_style))
    story.append(Paragraph("• <b>Kubernetes Certified Application Developer</b> - CNCF (2019)", normal_style))
    
    # Professional Experience
    story.append(Paragraph("PROFESSIONAL EXPERIENCE", heading_style))
    
    story.append(Paragraph("<b>Senior Professor of Computer Science</b> - University of Technology Lyon (2018-Present)", normal_style))
    story.append(Paragraph("• Teaching: Advanced Programming, AI/ML, Database Systems, Software Engineering (250h/year)", normal_style))
    story.append(Paragraph("• Research: Machine Learning, Natural Language Processing, Computer Vision", normal_style))
    story.append(Paragraph("• Supervision: 25+ Master's and PhD students in AI and Software Engineering", normal_style))
    story.append(Paragraph("• Leadership: Head of Computer Science Department (2021-Present)", normal_style))
    story.append(Paragraph("• Industry Partnerships: Collaborated with Google, Microsoft, and IBM on AI research projects", normal_style))
    
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Associate Professor</b> - University Paris-Saclay (2015-2018)", normal_style))
    story.append(Paragraph("• Courses: Algorithms, Data Structures, Software Engineering, Web Development", normal_style))
    story.append(Paragraph("• Development: Created new Data Science and Machine Learning curriculum", normal_style))
    story.append(Paragraph("• Innovation: Implemented project-based learning with industry partners", normal_style))
    story.append(Paragraph("• Research: Published 15+ papers in top-tier conferences (ICML, NeurIPS, AAAI)", normal_style))
    
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Software Engineer</b> - Tech Solutions Inc., Paris (2012-2015)", normal_style))
    story.append(Paragraph("• Full-stack development using Python, Java, JavaScript, and cloud technologies", normal_style))
    story.append(Paragraph("• Led development of AI-powered recommendation systems serving 1M+ users", normal_style))
    story.append(Paragraph("• DevOps: Implemented CI/CD pipelines, containerization, and cloud deployment", normal_style))
    
    # Technical Skills
    story.append(Paragraph("TECHNICAL SKILLS", heading_style))
    
    skills_data = [
        ["Programming Languages:", "Python, Java, C++, JavaScript, TypeScript, Go, Rust, R, MATLAB"],
        ["AI/ML Frameworks:", "TensorFlow, PyTorch, Keras, Scikit-learn, OpenCV, Hugging Face"],
        ["Web Technologies:", "React, Node.js, Django, Flask, HTML5, CSS3, REST APIs, GraphQL"],
        ["Databases:", "PostgreSQL, MongoDB, Redis, Elasticsearch, Neo4j, Apache Cassandra"],
        ["Cloud Platforms:", "AWS, Google Cloud, Microsoft Azure, Docker, Kubernetes"],
        ["DevOps Tools:", "Git, Jenkins, GitHub Actions, Terraform, Ansible, Prometheus"],
        ["Specialized Areas:", "Natural Language Processing, Computer Vision, Deep Learning, MLOps"]
    ]
    
    skills_table = Table(skills_data, colWidths=[2*inch, 4*inch])
    skills_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    
    story.append(skills_table)
    
    # Research & Publications
    story.append(Paragraph("RESEARCH & PUBLICATIONS", heading_style))
    story.append(Paragraph("<b>Selected Publications (h-index: 28, 1500+ citations):</b>", normal_style))
    story.append(Paragraph("• Bernard, P. et al. 'Deep Learning for Educational Content Recommendation' - ICML 2023", normal_style))
    story.append(Paragraph("• Bernard, P. 'Transformer-based Models for Code Generation' - NeurIPS 2022", normal_style))
    story.append(Paragraph("• Bernard, P. et al. 'Federated Learning in Educational Systems' - AAAI 2021", normal_style))
    story.append(Paragraph("• Bernard, P. 'Adaptive AI Tutoring Systems' - Nature Machine Intelligence 2020", normal_style))
    
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Research Grants:</b>", normal_style))
    story.append(Paragraph("• EU Horizon 2020: AI for Education (€2.5M, Principal Investigator, 2020-2024)", normal_style))
    story.append(Paragraph("• French National Research Agency: Adaptive Learning Systems (€800K, 2019-2022)", normal_style))
    
    # Achievements & Awards
    story.append(Paragraph("HONORS & ACHIEVEMENTS", heading_style))
    story.append(Paragraph("• <b>Excellence in Teaching Award</b> - University of Lyon (2022, 2023)", normal_style))
    story.append(Paragraph("• <b>Outstanding Research Award</b> - French Computer Science Society (2021)", normal_style))
    story.append(Paragraph("• <b>Best Paper Award</b> - International Conference on AI in Education (2020)", normal_style))
    story.append(Paragraph("• <b>Featured Speaker</b> - International AI Conference, MIT (2021, 2023)", normal_style))
    story.append(Paragraph("• <b>Editorial Board Member</b> - Journal of AI in Education (2019-Present)", normal_style))
    story.append(Paragraph("• <b>Program Committee</b> - NeurIPS, ICML, AAAI (2018-Present)", normal_style))
    
    # Projects
    story.append(Paragraph("NOTABLE PROJECTS", heading_style))
    story.append(Paragraph("<b>AI-Powered Code Assistant</b> - Open Source Project (2023-Present)", normal_style))
    story.append(Paragraph("• Developed an intelligent code completion system using transformer models", normal_style))
    story.append(Paragraph("• 10,000+ GitHub stars, integrated into VS Code extension with 50K+ downloads", normal_style))
    
    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>EduAI Platform</b> - Educational Technology Startup (2022-Present)", normal_style))
    story.append(Paragraph("• Co-founded startup developing AI-powered personalized learning platforms", normal_style))
    story.append(Paragraph("• Secured €1.2M in Series A funding, serving 100+ educational institutions", normal_style))
    
    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>MLOps Pipeline Framework</b> - Industry Collaboration (2021-2022)", normal_style))
    story.append(Paragraph("• Built enterprise-grade MLOps platform for automated model deployment", normal_style))
    story.append(Paragraph("• Deployed across 5 Fortune 500 companies, handling 1M+ daily predictions", normal_style))
    
    # Professional Activities
    story.append(Paragraph("PROFESSIONAL ACTIVITIES", heading_style))
    story.append(Paragraph("• <b>IEEE Senior Member</b> - Institute of Electrical and Electronics Engineers", normal_style))
    story.append(Paragraph("• <b>ACM Distinguished Scientist</b> - Association for Computing Machinery", normal_style))
    story.append(Paragraph("• <b>Reviewer</b> - Nature, Science, PNAS, JMLR, TPAMI (100+ papers reviewed)", normal_style))
    story.append(Paragraph("• <b>Conference Chair</b> - European Conference on AI in Education (2024)", normal_style))
    story.append(Paragraph("• <b>Industry Advisory Board</b> - Google AI Education Initiative", normal_style))
    story.append(Paragraph("• <b>Technical Consultant</b> - UNESCO AI in Education Global Report", normal_style))
    
    # Build PDF
    doc.build(story)
    
    # Clean up temporary file
    if os.path.exists("temp_profile.jpg"):
        os.remove("temp_profile.jpg")
    
    print("Enhanced CV for Dr. Pierre BERNARD created successfully: 'cv_prof_pierre_bernard.pdf'")

if __name__ == "__main__":
    create_cv_with_existing_image()