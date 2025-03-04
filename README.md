This repository contains a Flask API to deploy an ELMo-based model.

## 1. Install Dependencies
Run: pip install -r requirements.txt

## 2. Start the Flask App
Run: python app.py

## 3. Make Predictions
Send a POST request: curl -X POST "http://localhost:5000/predict" -H "Content-Type: application/json" -d '{"text": "Hello, how are you?"}'

If using an EC2 server, replace 'localhost' with the public IP.

## 4. Deploy on AWS EC2
To deploy, follow these steps:
1. Launch an EC2 instance
2. Install Python and dependencies
3. Clone this repo
4. Run `gunicorn -w 4 -b 0.0.0.0:5000 app:app`
