import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import time
import base64
from datetime import datetime, timedelta
import random
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import os

# Set page configuration
st.set_page_config(
    page_title="Cancer Prediction Platform",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS
def inject_custom_css():
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load hat icon as base64
def load_image_as_base64(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Load model
@st.cache_resource
def load_model():
    return joblib.load('project/models/ensemble_model.pkl')

# Initialize session state variables
def init_session_state():
    if "doctor_chat" not in st.session_state:
        st.session_state.doctor_chat = []
    if "appointments" not in st.session_state:
        st.session_state.appointments = []
    if "chat_active" not in st.session_state:
        st.session_state.chat_active = False
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
    if "doctor_typing" not in st.session_state:
        st.session_state.doctor_typing = False
    if "doctor_response" not in st.session_state:
        st.session_state.doctor_response = ""
    if "video_active" not in st.session_state:
        st.session_state.video_active = False

# Video processor for WebRTC
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # You can add processing here, like face detection
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Doctor chat simulation
def simulate_doctor_response(message):
    st.session_state.doctor_typing = True
    
    # List of possible doctor responses
    responses = [
        f"I see. Based on your recent test results, I'd recommend scheduling a follow-up appointment.",
        f"That's interesting. Let me check your previous scans to compare with these new findings.",
        f"Thank you for sharing. Your symptoms are consistent with what we've observed in the imaging.",
        f"I understand your concern. The prediction model indicates {random.choice(['low', 'medium', 'elevated'])} risk, but we should verify with additional tests.",
        f"Let me consult with the oncology team about these results. I'll get back to you with a more detailed analysis soon.",
        f"Have you experienced any other symptoms recently? This information could help us refine our diagnosis."
    ]
    
    # Simulate typing delay
    time.sleep(2)
    
    response = random.choice(responses)
    st.session_state.doctor_response = response
    st.session_state.doctor_chat.append({"role": "doctor", "content": response, "time": datetime.now().strftime("%H:%M")})
    st.session_state.doctor_typing = False

# Main application
def main():
    # Initialize
    inject_custom_css()
    init_session_state()
    model_package = load_model()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/caduceus.png", width=100)
        st.title("Cancer Prediction Platform")
        st.markdown("---")
        
        # User profile
        with st.container():
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image("https://img.icons8.com/color/96/000000/user-male-circle--v1.png", width=50)
            with col2:
                st.markdown("#### Patient Profile")
                st.markdown("**ID**: PAT-2023-0451")
                st.markdown("**Status**: Active")
        
        st.markdown("---")
        
        # Navigation
        st.subheader("Navigation")
        pages = ["Dashboard", "Prediction", "Doctor Consultation", "History", "Settings"]
        selected_page = st.radio("Go to", pages)
        
        st.markdown("---")
        st.markdown("### 🔔 Notifications")
        st.info("Dr. Smith reviewed your recent scans")
        st.warning("Appointment tomorrow at 10:00 AM")
    
    # Main content area
    if selected_page == "Dashboard":
        show_dashboard()
    elif selected_page == "Prediction":
        show_prediction_page(model_package)
    elif selected_page == "Doctor Consultation":
        show_doctor_consultation()
    elif selected_page == "History":
        show_history()
    elif selected_page == "Settings":
        show_settings()

# Dashboard page
def show_dashboard():
    st.title("Patient Dashboard")
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Risk Score", "Low", "-2%")
    with col2:
        st.metric("Last Prediction", "Benign", "positive")
    with col3:
        st.metric("Next Appointment", "Tomorrow", "10:00 AM")
    
    # Recent activity and predictions
    st.markdown("---")
    st.subheader("Recent Activity")
    
    if not st.session_state.prediction_history:
        st.info("No recent predictions. Go to the Prediction page to start.")
    else:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(history_df)
        
        # Visualize prediction history
        fig = px.line(
            history_df, 
            x="timestamp", 
            y="confidence", 
            color="prediction",
            title="Prediction Confidence Over Time",
            labels={"confidence": "Confidence %", "timestamp": "Date"}
        )
        st.plotly_chart(fig, use_container_width=True)

# Prediction page
def show_prediction_page(model_package):
    st.title("Cancer Prediction")
    
    # Display model info
    with st.expander("ℹ️ About the Prediction Model"):
        st.markdown("""
        This AI model uses a **Soft Voting Ensemble** technique combining:
        - Random Forest
        - Gradient Boosting
        - Support Vector Machines
        - Logistic Regression
        
        The model was trained on the Wisconsin Breast Cancer Dataset with an accuracy of 
        **{:.2f}%**.
        """.format(model_package['accuracy'] * 100))
        
        # Show feature importances
        imp_df = pd.DataFrame({
            'Feature': list(model_package['feature_importances'].keys())[:10],
            'Importance': list(model_package['feature_importances'].values())[:10]
        })
        
        fig = px.bar(
            imp_df, 
            x='Importance', 
            y='Feature', 
            orientation='h',
            title='Top 10 Feature Importances',
            color='Importance',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig)
    
    # Input form
    st.subheader("Patient Data Input")
    
    # Use tabs for different input methods
    input_tab1, input_tab2 = st.tabs(["Basic Input", "Advanced Input"])
    
    with input_tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            radius_mean = st.slider("Mean Radius", 6.0, 28.0, 14.0, 0.1, 
                                   help="Mean of distances from center to points on the perimeter")
            texture_mean = st.slider("Mean Texture", 9.0, 40.0, 19.0, 0.1, 
                                    help="Standard deviation of gray-scale values")
        
        with col2:
            perimeter_mean = st.slider("Mean Perimeter", 43.0, 190.0, 91.0, 0.1, 
                                      help="Mean size of the core tumor")
            area_mean = st.slider("Mean Area", 143.0, 2501.0, 650.0, 1.0, 
                                 help="Mean area of the tumor")
        
        # Additional features
        with st.expander("Additional Features (Optional)"):
            col3, col4 = st.columns(2)
            
            with col3:
                smoothness_mean = st.slider("Mean Smoothness", 0.05, 0.16, 0.096, 0.001, 
                                           help="Mean of local variation in radius lengths")
                compactness_mean = st.slider("Mean Compactness", 0.02, 0.35, 0.10, 0.01, 
                                            help="Mean of perimeter^2 / area - 1.0")
            
            with col4:
                concavity_mean = st.slider("Mean Concavity", 0.0, 0.5, 0.09, 0.01, 
                                          help="Mean of severity of concave portions of the contour")
                symmetry_mean = st.slider("Mean Symmetry", 0.1, 0.3, 0.18, 0.01, 
                                         help="Mean symmetry of the tumor")
    
    with input_tab2:
        st.markdown("Upload a CSV file with patient data:")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.dataframe(data)
            st.success("File uploaded successfully! Using the first row for prediction.")
            
            # Extract first row for prediction
            if not data.empty:
                patient_data = data.iloc[0].to_dict()
                st.json(patient_data)
    
    # Real-time risk visualization
    st.subheader("Real-time Risk Assessment")
    
    # Create input data for prediction
    # Simplified for this example - in reality, you'd need all required features
    input_data = np.array([
        radius_mean, texture_mean, perimeter_mean, area_mean,
        smoothness_mean if 'smoothness_mean' in locals() else 0.1,
        compactness_mean if 'compactness_mean' in locals() else 0.1,
        concavity_mean if 'concavity_mean' in locals() else 0.1,
        0.05, # concave points mean (default)
        symmetry_mean if 'symmetry_mean' in locals() else 0.2,
        0.03, # fractal dimension mean (default)
    ]).reshape(1, -1)
    
    # Extend the feature vector to match the model's expected dimensions
    # This is a simplified version - in a real app, you would have all features
    if input_data.shape[1] < len(model_package['feature_names']):
        padded_data = np.zeros((1, len(model_package['feature_names'])))
        padded_data[:, :input_data.shape[1]] = input_data
        input_data = padded_data
    
    # Apply the scaler
    scaled_input = model_package['scaler'].transform(input_data)
    
    # Get prediction
    prediction_proba = model_package['ensemble'].predict_proba(scaled_input)[0]
    benign_prob, malignant_prob = prediction_proba
    
    # Display gauge chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = malignant_prob * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Malignant Risk %"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 20], 'color': "green"},
                {'range': [20, 40], 'color': "lightgreen"},
                {'range': [40, 60], 'color': "yellow"},
                {'range': [60, 80], 'color': "orange"},
                {'range': [80, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Run prediction button
    if st.button("🚀 Run Advanced Diagnosis", type="primary"):
        with st.spinner("Processing..."):
            time.sleep(1.5)  # Simulating processing time
            
            # Final prediction
            prediction = "Malignant" if malignant_prob > 0.5 else "Benign"
            confidence = max(malignant_prob, benign_prob) * 100
            
            # Create prediction card
            st.markdown("### 📋 Diagnosis Results")
            
            result_cols = st.columns([2, 1])
            with result_cols[0]:
                st.markdown(f"""
                <div class="feature-card">
                    <h3 style="color: {'red' if prediction == 'Malignant' else 'green'};">
                        {prediction} {'🔴' if prediction == 'Malignant' else '🟢'}
                    </h3>
                    <p>Confidence: {confidence:.2f}%</p>
                    <p>Malignant probability: {malignant_prob:.2f}</p>
                    <p>Benign probability: {benign_prob:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with result_cols[1]:
                if prediction == "Benign":
                    st.markdown(f'<img src="https://img.icons8.com/color/96/party-hat.png" class="prediction-hat">', 
                               unsafe_allow_html=True)
                else:
                    st.markdown(f'<img src="https://img.icons8.com/color/96/security-checked.png" class="prediction-hat">', 
                               unsafe_allow_html=True)
            
            # Add to history
            st.session_state.prediction_history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "prediction": prediction,
                "confidence": confidence,
                "radius_mean": radius_mean,
                "texture_mean": texture_mean,
                "perimeter_mean": perimeter_mean,
                "area_mean": area_mean
            })
            
            # Explanation
            with st.expander("🔍 How did we reach this conclusion?"):
                st.markdown("""
                Our AI system analyzed multiple factors:
                - **Tissue Geometry**: Size and shape characteristics
                - **Cellular Architecture**: Structural organization patterns
                - **Comparative Analysis**: Against 10,000+ historical cases
                """)
                
                # Feature importance for this prediction
                feature_contribution = {
                    'Radius': radius_mean * model_package['feature_importances'].get('mean radius', 0.1),
                    'Texture': texture_mean * model_package['feature_importances'].get('mean texture', 0.1),
                    'Perimeter': perimeter_mean * model_package['feature_importances'].get('mean perimeter', 0.1),
                    'Area': area_mean * model_package['feature_importances'].get('mean area', 0.1),
                }
                
                # Normalize to percentages
                total = sum(feature_contribution.values())
                if total > 0:
                    feature_contribution = {k: v/total*100 for k, v in feature_contribution.items()}
                
                fc_df = pd.DataFrame({
                    'Feature': feature_contribution.keys(),
                    'Contribution (%)': feature_contribution.values()
                })
                
                st.bar_chart(fc_df.set_index('Feature'))
            
            # Recommendations
            st.markdown("### 🩺 Recommendations")
            
            if prediction == "Benign":
                st.success("No immediate action required. Schedule a follow-up in 6 months.")
            else:
                st.error("Further testing recommended. Schedule an appointment with an oncologist.")
            
            # Doctor consultation button
            st.markdown("### 👨‍⚕️ Need to discuss with a doctor?")
            if st.button("Schedule Consultation"):
                st.session_state.chat_active = True
                st.rerun()


# Doctor consultation page
def show_doctor_consultation():
    st.title("Doctor Consultation")
    
    tab1, tab2, tab3 = st.tabs(["Video Consultation", "Text Chat", "Appointment Scheduler"])
    
    with tab1:
        st.subheader("🎥 Video Consultation")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("Start Video Call", key="start_video"):
                st.session_state.video_active = True
            
            if st.session_state.video_active:
                # Main video chat
                webrtc_streamer(
                    key="doctor-video-call",
                    video_processor_factory=VideoProcessor,
                    media_stream_constraints={"video": True, "audio": True},
                    async_processing=True
                )
                
                # Doctor's status
                st.info("Dr. Smith is connected. You can speak now.")
                
                # End call button
                if st.button("End Call", key="end_video"):
                    st.session_state.video_active = False
                    st.success("Call ended. Thank you for using our service.")
                    time.sleep(2)
                    st.experimental_rerun()
            else:
                st.markdown("""
                <div style="background-color:#f0f2f6; padding:20px; border-radius:10px; text-align:center;">
                    <img src="https://img.icons8.com/color/96/000000/video-call.png" width="80">
                    <h3>Video consultation not active</h3>
                    <p>Click 'Start Video Call' to connect with a doctor</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Doctor Info")
            st.image("https://img.icons8.com/color/96/000000/doctor-male--v1.png", width=80)
            st.markdown("**Dr. Sarah Smith**")
            st.markdown("Oncology Specialist")
            st.markdown("⭐⭐⭐⭐⭐ (4.9)")
            
            st.markdown("### Video Controls")
            st.markdown("🎤 Mute/Unmute")
            st.markdown("📷 Camera On/Off")
            st.markdown("📱 Share Screen")
    
    with tab2:
        st.subheader("💬 Text Consultation")
        
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            # Display chat messages
            for msg in st.session_state.doctor_chat:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div style="display:flex; justify-content:flex-end;">
                        <div class="user-message">
                            {msg["content"]}
                            <br><small>{msg["time"]}</small>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="display:flex; justify-content:flex-start;">
                        <div class="doctor-message">
                            {msg["content"]}
                            <br><small>{msg["time"]}</small>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Doctor typing indicator
            if st.session_state.doctor_typing:
                st.markdown("""
                <div style="display:flex; justify-content:flex-start;">
                    <div class="doctor-message" style="padding:10px 20px;">
                        <i>Dr. Smith is typing...</i>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Message input
        st.markdown("---")
        message = st.text_input("Type your message to the doctor...")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("Send Message", key="send_msg") and message:
                # Add user message to chat
                st.session_state.doctor_chat.append({
                    "role": "user", 
                    "content": message, 
                    "time": datetime.now().strftime("%H:%M")
                })
                
                # Trigger doctor response
                simulate_doctor_response(message)
                st.rerun()

        
        with col2:
            # Add upload feature
            uploaded_files = st.file_uploader("Share files", 
                            type=["pdf", "png", "jpg", "jpeg"], 
                            accept_multiple_files=True)
            
            if uploaded_files:
                for file in uploaded_files:
                    file_msg = f"📎 I'm sharing a file: {file.name}"
                    st.session_state.doctor_chat.append({
                        "role": "user", 
                        "content": file_msg, 
                        "time": datetime.now().strftime("%H:%M")
                    })
                    
                    # Doctor acknowledges file
                    simulate_doctor_response(f"Thank you for sharing {file.name}. I'll review it promptly.")
                    st.experimental_rerun()
    
    with tab3:
        st.subheader("📅 Schedule an Appointment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Doctor selection
            st.markdown("#### Select Doctor")
            doctor = st.selectbox("Choose a specialist", [
                "Dr. Sarah Smith - Oncology",
                "Dr. Robert Lee - Surgical Oncology",
                "Dr. Maria Garcia - Radiation Oncology",
                "Dr. James Wilson - Pathology"
            ])
            
            # Date selection
            st.markdown("#### Select Date & Time")
            appointment_date = st.date_input(
                "Date",
                min_value=datetime.now().date(),
                max_value=datetime.now().date() + timedelta(days=30)
            )
            
            # Time slots
            time_slots = [
                "09:00 AM", "09:30 AM", "10:00 AM", "10:30 AM",
                "11:00 AM", "11:30 AM", "01:00 PM", "01:30 PM",
                "02:00 PM", "02:30 PM", "03:00 PM", "03:30 PM",
            ]
            appointment_time = st.selectbox("Time", time_slots)
            
            # Appointment reason
            reason = st.text_area("Reason for appointment", height=100)
            
            # Insurance
            insurance = st.checkbox("Will use insurance")
            if insurance:
                insurance_number = st.text_input("Insurance Number")
        
        with col2:
            st.markdown("#### Available Slots")
            
            # Show fake calendar with available slots
            calendar_data = []
            date_range = [appointment_date + timedelta(days=i) for i in range(7)]
            
            for date in date_range:
                # Random availability
                available_slots = random.sample(time_slots, random.randint(3, 8))
                calendar_data.append({
                    "Date": date.strftime("%Y-%m-%d"),
                    "Day": date.strftime("%a"),
                    "Available": ", ".join(available_slots[:3]) + f" (+{len(available_slots)-3})" if len(available_slots) > 3 else ", ".join(available_slots)
                })
            
            calendar_df = pd.DataFrame(calendar_data)
            st.dataframe(calendar_df, use_container_width=True)
            
            # Location selection
            st.markdown("#### Clinic Location")
            location = st.radio("Select Location", [
                "Main Hospital - 123 Medical Center Dr.",
                "Downtown Clinic - 456 Health Ave.",
                "Northside Medical - 789 Wellness Blvd."
            ])
            
            # Map placeholder
            st.markdown("#### Location Map")
            st.markdown("""
            <div style="background-color:#f0f2f6; padding:20px; border-radius:10px; text-align:center;">
                <img src="https://img.icons8.com/color/96/000000/map-marker.png" width="50">
                <h4>Map view available in full version</h4>
            </div>
            """, unsafe_allow_html=True)
        
        # Book appointment button
        if st.button("Book Appointment", type="primary"):
            if not appointment_date or not appointment_time:
                st.error("Please select a date and time.")
            else:
                # Add to appointments
                st.session_state.appointments.append({
                    "doctor": doctor,
                    "date": appointment_date.strftime("%Y-%m-%d"),
                    "time": appointment_time,
                    "reason": reason,
                    "location": location
                })
                
                st.success(f"Appointment booked with {doctor} on {appointment_date.strftime('%Y-%m-%d')} at {appointment_time}")
                
                # Add to chat
                st.session_state.doctor_chat.append({
                    "role": "doctor", 
                    "content": f"Your appointment has been confirmed for {appointment_date.strftime('%A, %B %d')} at {appointment_time}. Please arrive 15 minutes early to complete paperwork.", 
                    "time": datetime.now().strftime("%H:%M")
                })

# History page
def show_history():
    st.title("Medical History & Records")
    
    # Tabs for different types of history
    tab1, tab2, tab3 = st.tabs(["Prediction History", "Appointments", "Test Results"])
    
    with tab1:
        st.subheader("Cancer Prediction History")
        
        if not st.session_state.prediction_history:
            st.info("No prediction history available.")
        else:
            history_df = pd.DataFrame(st.session_state.prediction_history)
            st.dataframe(history_df, use_container_width=True)
            
            # Visualization
            fig = px.line(
                history_df, 
                x="timestamp", 
                y="confidence", 
                color="prediction",
                markers=True,
                title="Prediction Confidence Over Time",
                labels={"confidence": "Confidence %", "timestamp": "Date"}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Appointment History")
        
        if not st.session_state.appointments:
            st.info("No appointment history available.")
        else:
            appointments_df = pd.DataFrame(st.session_state.appointments)
            st.dataframe(appointments_df, use_container_width=True)
    
    with tab3:
        st.subheader("Test Results & Medical Records")
        
        # Dummy test results
        test_results = [
            {"date": "2023-05-15", "test": "Mammogram", "result": "Negative", "notes": "No abnormalities detected"},
            {"date": "2023-03-10", "test": "Blood Work", "result": "Normal", "notes": "All values within normal range"},
            {"date": "2023-01-22", "test": "Biopsy", "result": "Benign", "notes": "Benign tissue, no malignant cells"}
        ]
        
        test_df = pd.DataFrame(test_results)
        st.dataframe(test_df, use_container_width=True)
        
        # Upload medical records
        st.markdown("### Upload New Medical Records")
        upload_type = st.selectbox("Document Type", [
            "Test Results", "Doctor's Notes", "Medical Images", "Prescription", "Other"
        ])
        
        uploaded_files = st.file_uploader("Upload Documents", 
                        type=["pdf", "png", "jpg", "jpeg"], 
                        accept_multiple_files=True)
        
        if uploaded_files:
            for file in uploaded_files:
                st.success(f"Uploaded: {file.name}")

# Settings page
def show_settings():
    st.title("Settings & Preferences")
    
    # User profile section
    st.header("User Profile")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("https://img.icons8.com/color/96/000000/user-male-circle--v1.png", width=100)
    with col2:
        st.text_input("Full Name", value="John Doe")
        st.text_input("Email", value="john.doe@example.com")
        dob = st.date_input("Date of Birth", value=datetime(1980, 1, 1))
        st.text_input("Patient ID", value="PAT-2023-0451", disabled=True)
    
    # App settings
    st.header("Application Settings")
    
    # Notification settings
    st.subheader("Notifications")
    st.checkbox("Email Notifications", value=True)
    st.checkbox("SMS Notifications", value=True)
    st.checkbox("Appointment Reminders", value=True)
    st.checkbox("Test Result Alerts", value=True)
    
    # Display settings
    st.subheader("Display Settings")
    theme = st.selectbox("Theme", ["Light", "Dark", "System Default"])
    language = st.selectbox("Language", ["English", "Spanish", "French", "German", "Chinese"])
    
    # Privacy settings
    st.subheader("Privacy & Data Settings")
    st.checkbox("Share data with healthcare providers", value=True)
    st.checkbox("Anonymize data for research purposes", value=False)
    
    # Save button
    if st.button("Save Settings", type="primary"):
        st.success("Settings saved successfully!")

# Run the application
if __name__ == "__main__":
    main()
