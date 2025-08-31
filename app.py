from flask import Flask, request, jsonify, render_template
import cv2
from transformers import pipeline
import librosa
import numpy as np
import tempfile
import os
import openai
from facenet_pytorch import MTCNN
from PIL import Image
from collections import Counter
import logging
import torch

# Initialize logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route('/')
def home():
    # Serve the index.html file
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    video = request.files.get('video')
    audio = request.files.get('audio')
    text_prompt = request.form.get('text_prompt', '')
    results = {}

    if video:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            video_path = temp_video.name
            video.save(video_path)
        try:
            cap = cv2.VideoCapture(video_path)
            frame_emotions = []
            emotion_timestamps = []
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_idx = 0
            mtcnn = MTCNN(keep_all=True)

            # Check for GPU availability
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logging.info(f"Device set to use {device}")

            # Updated to use a public model with GPU support
            emotion_classifier = pipeline("image-classification", model="microsoft/resnet-50", device=0 if device == 'cuda' else -1)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process every 10th frame
                if frame_idx % 10 != 0:
                    frame_idx += 1
                    continue

                timestamp = frame_idx / fps if fps else frame_idx
                try:
                    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    boxes, _ = mtcnn.detect(img_pil)
                    if boxes is not None:
                        emo_result = emotion_classifier(img_pil)
                        if emo_result:
                            top_emotion = emo_result[0]['label']
                            frame_emotions.append(top_emotion)
                            emotion_timestamps.append({'timestamp': round(float(timestamp), 2), 'emotion': top_emotion})
                        else:
                            frame_emotions.append(None)
                            emotion_timestamps.append({'timestamp': round(float(timestamp), 2), 'emotion': None})
                    else:
                        frame_emotions.append(None)
                        emotion_timestamps.append({'timestamp': round(float(timestamp), 2), 'emotion': None})
                except Exception as e:
                    logging.error(f"Error processing frame {frame_idx}: {e}")
                    frame_emotions.append(None)
                    emotion_timestamps.append({'timestamp': round(float(timestamp), 2), 'emotion': None, 'error': str(e)})
                frame_idx += 1

            emotion_counts = Counter([e for e in frame_emotions if e])
            total = sum(emotion_counts.values())
            summary = {}
            for emotion, count in emotion_counts.items():
                percent = round(100 * count / total, 1) if total else 0
                summary[emotion] = percent
            sorted_summary = sorted(summary.items(), key=lambda x: x[1], reverse=True)
            summary_str = ', '.join([f"{emotion} ({percent}%)" for emotion, percent in sorted_summary])
            if sorted_summary:
                main_emotion = sorted_summary[0][0]
                results['video_emotion_summary'] = f"You were mostly {main_emotion} ({sorted_summary[0][1]}%). Other emotions: {summary_str}."
            else:
                results['video_emotion_summary'] = "No clear emotion detected."
            results['video_emotions'] = frame_emotions
            results['video_emotion_timestamps'] = emotion_timestamps
            results['video_emotion_percentages'] = {k: float(v) for k, v in summary.items()}
        except Exception as e:
            logging.error(f"Error analyzing video: {e}")
            results['video_emotions_error'] = str(e)
        finally:
            cap.release()
            os.remove(video_path)

    if audio:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            audio_path = temp_audio.name
            audio.save(audio_path)
        try:
            y, sr = librosa.load(audio_path, sr=None)
            pitch = librosa.yin(y, fmin=50, fmax=300, sr=sr)
            intensity = np.mean(librosa.feature.rms(y=y))
            results['audio_analysis'] = {
                'average_pitch': float(np.mean(pitch)),
                'intensity': float(intensity)
            }
        except Exception as e:
            logging.error(f"Error analyzing audio: {e}")
            results['audio_analysis_error'] = str(e)
        finally:
            os.remove(audio_path)

    if text_prompt:
        try:
            gpt_response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful coach for interviews and presentations."},
                    {"role": "user", "content": text_prompt}
                ]
            )
            results['coaching_feedback'] = gpt_response['choices'][0]['message']['content']
        except Exception as e:
            logging.error(f"Error generating coaching feedback: {e}")
            results['coaching_feedback_error'] = str(e)

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
