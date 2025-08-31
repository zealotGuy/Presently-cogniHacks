# Presently: Your AI Communication Helper
Presently is an AI-powered communication coach designed to help you become a more confident and effective speaker. Whether you're preparing for a job interview, a public speech, or a class presentation, our app provides real-time feedback on your performance, analyzing your expressions, tone, and content to guide you to success.

## Key Features
* Video/Audio Analysis: Presently analyzes your facial expressions, voice tone, voice volume, and spoken content
* Feedback: The app provides feedback based on your audio level/intensity, confidence, clarity, and effectiveness

## How it was built
Presently was built using a combination of powerful technologies to create an intelligent coaching experience
* Backend: We used Flask to create a robust and manageable backend server
* Emotion Detection: We implemented emotion detection using Pytorch, allowing us to analyze facial expressions
* AI Model: The API key utilizes openai's gpt 4 as our core AI model to provide human-like feedback
* Frontend: The user interface was built using HTML and JSON for a responsive experience

## Some Challenges
Building a file uploading multimodal AI application wasn't easy. Our biggest challenge was integrating emotion detection with audio analysis. This required careful synchronization to make sure the feedback was accurate, while maintaining a smooth user experience as well. We also encountered a recurring error during development that took significant time to debug (an error we told the program to send if there were any issues).

## Accomplishments
* Ccombined computer vision and audio analysis to create a feedback system
* Integrated a Flask backend with a responsive HTML frontend
* Developed a working prototype that demonstrates the power of a large language model for personalized coaching

## Next Steps
* Real-Time feedback (live feedback while talking)
* Increased Runtime speed
* Progress Tracker
