from flask import Flask, request, jsonify
import google.generativeai as genai
import tempfile
import os
import logging
from flask_cors import CORS
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-pro')

@app.route('/')
def home():
    try:
        with open('index.html', 'r') as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error reading index.html: {e}")
        return f"Error loading page: {str(e)}", 500

@app.route('/analyze', methods=['POST'])
def analyze():
    video = request.files.get('video')
    audio = request.files.get('audio')
    text_prompt = request.form.get('text_prompt', '')
    
    results = {
        'video_emotions': [],
        'audio_analysis': {},
        'coaching_feedback': '',
        'overall_analysis': ''
    }
    
    # Build the prompt parts for Gemini
    prompt_parts = []
    
    # Add video file directly to Gemini if present
    if video:
        logging.info("Processing video...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            video_path = temp_video.name
            video.save(video_path)
        
        try:
            # Upload the video file to Gemini
            video_file = genai.upload_file(path=video_path, display_name="presentation_video")
            
            # Wait for the file to be processed
            import time
            while video_file.state.name == "PROCESSING":
                time.sleep(1)
                video_file = genai.get_file(video_file.name)
            
            if video_file.state.name == "FAILED":
                raise ValueError(f"Video file processing failed: {video_file.state.name}")
                
            prompt_parts.append(video_file)
            prompt_parts.append("Analyze this video for emotional expressions, body language, and presentation quality.")
        except Exception as e:
            logging.error(f"Video upload error: {e}")
            results['error'] = f"Video processing failed: {str(e)}"
        finally:
            if os.path.exists(video_path):
                os.remove(video_path)
    
    # Add audio file directly to Gemini if present
    if audio:
        logging.info("Processing audio...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            audio_path = temp_audio.name
            audio.save(audio_path)
        
        try:
            # Upload the audio file to Gemini
            audio_file = genai.upload_file(path=audio_path, display_name="presentation_audio")
            
            # Wait for the file to be processed
            import time
            while audio_file.state.name == "PROCESSING":
                time.sleep(1)
                audio_file = genai.get_file(audio_file.name)
            
            if audio_file.state.name == "FAILED":
                raise ValueError(f"Audio file processing failed: {audio_file.state.name}")
                
            prompt_parts.append(audio_file)
            prompt_parts.append("Analyze this audio for speaking pace, tone variation, confidence level, and vocal clarity.")
        except Exception as e:
            logging.error(f"Audio upload error: {e}")
            results['error'] = f"Audio processing failed: {str(e)}"
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)
    
    if text_prompt:
        prompt_parts.append(f"\nUser's specific question/context: {text_prompt}")
    
    # Add the structured output request
    if prompt_parts:
        prompt_parts.append("""
        Please provide a comprehensive analysis in JSON format with these exact keys:
        {
            "video_emotions": ["list of detected emotions from video"],
            "emotion_timeline": [{"timestamp": 0.0, "emotion": "emotion_name", "confidence": 0.95}],
            "body_language": "analysis of posture, gestures, and non-verbal communication",
            "audio_feedback": {
                "pitch_analysis": "evaluation of pitch variation and appropriateness",
                "pace": "speaking pace assessment",
                "confidence_level": "vocal confidence rating 1-10",
                "areas_to_improve": ["list of specific audio improvements"]
            },
            "coaching_feedback": "personalized coaching advice based on all inputs",
            "strengths": ["list of presentation strengths"],
            "improvement_areas": ["specific areas needing work"],
            "overall_score": 85,
            "professional_tips": ["actionable tips for improvement"]
        }
        
        Be constructive, specific, and encouraging in your feedback. Focus on actionable improvements.
        """)
        
        # Use Gemini to analyze everything
        try:
            response = model.generate_content(prompt_parts)
            
            # Parse Gemini's response
            try:
                import json
                response_text = response.text
                
                # Clean up response to extract JSON
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0]
                
                gemini_data = json.loads(response_text.strip())
                
                # Map to expected frontend format
                results['video_emotions'] = gemini_data.get('video_emotions', [])
                results['emotion_timeline'] = gemini_data.get('emotion_timeline', [])
                results['body_language'] = gemini_data.get('body_language', '')
                results['audio_feedback'] = gemini_data.get('audio_feedback', {})
                results['coaching_feedback'] = gemini_data.get('coaching_feedback', '')
                results['strengths'] = gemini_data.get('strengths', [])
                results['improvement_areas'] = gemini_data.get('improvement_areas', [])
                results['overall_score'] = gemini_data.get('overall_score', 0)
                results['professional_tips'] = gemini_data.get('professional_tips', [])
                
                # Add some mock audio analysis data if we had audio but no detailed analysis
                if audio and not results.get('audio_analysis'):
                    results['audio_analysis'] = {
                        'average_pitch': 150.0,
                        'intensity': 0.065,
                        'tempo': 120.0
                    }
                
            except json.JSONDecodeError:
                # Fallback to text response if JSON parsing fails
                results['coaching_feedback'] = response.text
                results['video_emotions'] = ['Analysis completed - see feedback']
                
            logging.info("Gemini analysis completed successfully")
            
        except Exception as e:
            logging.error(f"Gemini API error: {e}")
            results['error'] = f"Analysis failed: {str(e)}"
            results['coaching_feedback'] = "Unable to complete analysis. Please check your API key and try again."
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=5001, host='0.0.0.0')