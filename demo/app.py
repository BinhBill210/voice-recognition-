"""
Streamlit demo app for Speech Emotion Recognition.
Allows users to upload audio files or record from microphone for emotion prediction.
"""

# IMPORTANT: Set environment variables BEFORE any other imports
import os
os.environ['NUMBA_CACHE_DIR'] = '/tmp'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'  # Critical for macOS

import streamlit as st
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import io
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Import config first (doesn't load heavy libraries)
from config import (
    EMOTION_NAMES, EMOTION_FULL_NAMES,
    SAMPLE_RATE, FINAL_MODELS_DIR
)

# Lazy imports for heavy modules (will be imported when needed)
EmotionPredictor = None
AudioRecorder = None

# Page configuration
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #555;
        margin-bottom: 2rem;
    }
    .emotion-card {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained model (cached)."""
    try:
        # Lazy import to avoid loading TensorFlow at startup
        from predict import EmotionPredictor as EP
        
        model_files = list(FINAL_MODELS_DIR.glob('*.keras'))
        if not model_files:
            return None
        latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
        return EP(latest_model)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def plot_probabilities(probabilities: dict) -> plt.Figure:
    """Create a bar plot of emotion probabilities."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    emotions = list(probabilities.keys())
    probs = list(probabilities.values())
    colors = plt.cm.viridis(np.linspace(0, 1, len(emotions)))
    
    bars = ax.barh(emotions, probs, color=colors)
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_title('Emotion Probabilities', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])
    
    # Add value labels on bars
    for bar, prob in zip(bars, probs):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                f'{prob:.1%}', ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    return fig


def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown('<p class="main-header">🎤 Speech Emotion Recognition</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Detect emotions from speech using deep learning</p>', unsafe_allow_html=True)
    
    # Load model
    with st.spinner('Loading model...'):
        predictor = load_model()
    
    if predictor is None:
        st.error("❌ No trained model found. Please train a model first.")
        st.stop()
    
    st.success("✅ Model loaded successfully!")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This app uses a Convolutional Neural Network (CNN) to recognize emotions from speech.
        
        **Emotions recognized:**
        - 😠 Anger (ANG)
        - 😊 Happiness (HAP)
        - 😢 Sadness (SAD)
        - 😐 Neutral (NEU)
        - 🤢 Disgust (DIS)
        - 😨 Fear (FEA)
        """)
        
        st.header("📊 Model Info")
        st.write(f"**Model:** {predictor.model_path.name}")
        st.write(f"**Sample Rate:** {SAMPLE_RATE} Hz")
        st.write(f"**Classes:** {len(EMOTION_NAMES)}")
        
        st.header("🎯 Instructions")
        st.write("""
        1. Choose input method (Upload or Record)
        2. Provide audio input
        3. Click 'Predict Emotion'
        4. View results and probabilities
        """)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📁 Upload Audio", "🎙️ Record Audio", "📊 Batch Processing"])
    
    # Tab 1: Upload Audio
    with tab1:
        st.header("Upload Audio File")
        
        uploaded_file = st.file_uploader(
            "Choose a WAV file",
            type=['wav'],
            help="Upload a WAV audio file (preferably 3-5 seconds)"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            temp_path = Path("temp_upload.wav")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            
            # Display audio player
            st.audio(str(temp_path), format='audio/wav')
            
            # Predict button
            if st.button("🔮 Predict Emotion", key="predict_upload"):
                with st.spinner('Analyzing audio...'):
                    try:
                        result = predictor.predict(temp_path)
                        
                        # Display results
                        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                        
                        col1, col2 = st.columns([2, 3])
                        
                        with col1:
                            st.metric(
                                label="Predicted Emotion",
                                value=f"{result['predicted_emotion']} - {result['emotion_full_name']}",
                                delta=f"Confidence: {result['confidence']:.1%}"
                            )
                        
                        with col2:
                            # Top-3 predictions
                            st.subheader("Top 3 Predictions")
                            for i, pred in enumerate(result['top_k_predictions'], 1):
                                st.write(f"{i}. **{pred['emotion']}** - {pred['emotion_full_name']}: {pred['confidence']:.1%}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Plot probabilities
                        st.subheader("📊 All Emotion Probabilities")
                        fig = plot_probabilities(result['all_probabilities'])
                        st.pyplot(fig)
                        
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                    finally:
                        # Clean up temp file
                        if temp_path.exists():
                            temp_path.unlink()
    
    # Tab 2: Record Audio
    with tab2:
        st.header("Record Audio from Microphone")
        
        st.warning("⚠️ Note: Recording functionality requires proper browser permissions and may not work in all environments.")
        
        duration = st.slider(
            "Recording Duration (seconds)",
            min_value=1.0,
            max_value=10.0,
            value=3.0,
            step=0.5
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔴 Start Recording", key="record"):
                st.info(f"Recording for {duration} seconds...")
                
                try:
                    # Lazy import
                    from record import AudioRecorder as AR
                    recorder = AR(sample_rate=SAMPLE_RATE)
                    audio = recorder.record(duration=duration)
                    
                    # Save recording
                    record_path = Path("temp_recording.wav")
                    recorder.save(record_path, audio)
                    
                    st.success("✅ Recording complete!")
                    st.audio(str(record_path), format='audio/wav')
                    
                    # Store in session state
                    st.session_state.recorded_audio = record_path
                    
                except Exception as e:
                    st.error(f"❌ Recording error: {e}")
        
        with col2:
            if st.button("🔮 Predict from Recording", key="predict_record"):
                if 'recorded_audio' not in st.session_state:
                    st.warning("⚠️ Please record audio first!")
                else:
                    with st.spinner('Analyzing audio...'):
                        try:
                            result = predictor.predict(st.session_state.recorded_audio)
                            
                            # Display results (same as Tab 1)
                            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                            
                            st.metric(
                                label="Predicted Emotion",
                                value=f"{result['predicted_emotion']} - {result['emotion_full_name']}",
                                delta=f"Confidence: {result['confidence']:.1%}"
                            )
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Plot
                            fig = plot_probabilities(result['all_probabilities'])
                            st.pyplot(fig)
                            
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
    
    # Tab 3: Batch Processing
    with tab3:
        st.header("Batch Audio Processing")
        
        uploaded_files = st.file_uploader(
            "Upload multiple WAV files",
            type=['wav'],
            accept_multiple_files=True,
            help="Upload multiple audio files for batch prediction"
        )
        
        if uploaded_files:
            st.write(f"📂 Uploaded {len(uploaded_files)} files")
            
            if st.button("🔮 Predict All", key="predict_batch"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = []
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                    
                    # Save temp file
                    temp_path = Path(f"temp_batch_{i}.wav")
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.read())
                    
                    try:
                        result = predictor.predict(temp_path)
                        result['filename'] = uploaded_file.name
                        results.append(result)
                    except Exception as e:
                        st.warning(f"Error processing {uploaded_file.name}: {e}")
                    finally:
                        if temp_path.exists():
                            temp_path.unlink()
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text("✅ All files processed!")
                
                # Display results table
                st.subheader("📋 Batch Results")
                
                import pandas as pd
                df = pd.DataFrame([
                    {
                        'File': r['filename'],
                        'Emotion': r['predicted_emotion'],
                        'Full Name': r['emotion_full_name'],
                        'Confidence': f"{r['confidence']:.1%}"
                    }
                    for r in results
                ])
                
                st.dataframe(df, use_container_width=True)
                
                # Summary statistics
                st.subheader("📊 Summary Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Files", len(results))
                
                with col2:
                    avg_confidence = np.mean([r['confidence'] for r in results])
                    st.metric("Average Confidence", f"{avg_confidence:.1%}")
                
                with col3:
                    emotion_counts = pd.Series([r['predicted_emotion'] for r in results]).value_counts()
                    st.metric("Most Common", emotion_counts.index[0])
                
                # Emotion distribution
                st.subheader("📊 Emotion Distribution")
                fig, ax = plt.subplots(figsize=(10, 6))
                emotion_counts.plot(kind='bar', ax=ax, color='skyblue')
                ax.set_xlabel('Emotion')
                ax.set_ylabel('Count')
                ax.set_title('Distribution of Predicted Emotions')
                plt.tight_layout()
                st.pyplot(fig)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Built with ❤️ using Streamlit and TensorFlow</p>
        <p>Speech Emotion Recognition • CREMA-D Dataset</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

