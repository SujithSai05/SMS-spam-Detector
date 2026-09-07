from flask import Flask, request, render_template_string
from src.predict import predict_message

app = Flask(__name__)

HTML = '''
<!doctype html>
<html>
<head>
    <title>SMS Spam Detector</title>
    <style>
        * { box-sizing: border-box; }
        body {
            margin: 0;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 50%, #38bdf8 100%);
            color: #e2e8f0;
            padding: 24px;
        }
        .card {
            width: 100%;
            max-width: 760px;
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 22px;
            backdrop-filter: blur(10px);
            box-shadow: 0 20px 60px rgba(15, 23, 42, 0.45);
            padding: 32px;
        }
        .header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 20px;
        }
        h1 {
            margin: 0;
            font-size: 2rem;
            font-weight: 700;
        }
        .badge {
            background: rgba(59, 130, 246, 0.2);
            color: #bfdbfe;
            border: 1px solid rgba(147, 197, 253, 0.4);
            border-radius: 999px;
            padding: 8px 14px;
            font-size: 0.8rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        textarea {
            width: 100%;
            min-height: 150px;
            resize: vertical;
            border: 1px solid rgba(148, 163, 184, 0.3);
            border-radius: 16px;
            background: rgba(15, 23, 42, 0.55);
            color: #f8fafc;
            padding: 16px 18px;
            font-size: 1rem;
            line-height: 1.6;
            outline: none;
        }
        textarea:focus {
            border-color: #7dd3fc;
            box-shadow: 0 0 0 3px rgba(125, 211, 252, 0.18);
        }
        .actions {
            margin-top: 18px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 14px;
            flex-wrap: wrap;
        }
        button {
            background: linear-gradient(135deg, #38bdf8 0%, #2563eb 100%);
            border: none;
            color: white;
            padding: 14px 22px;
            border-radius: 12px;
            cursor: pointer;
            font-size: 1rem;
            font-weight: 600;
            transition: transform 0.2s ease;
        }
        button:hover {
            transform: translateY(-2px);
        }
        .status {
            min-height: 64px;
            margin-top: 24px;
            border-radius: 16px;
            padding: 18px 20px;
            background: rgba(15, 23, 42, 0.5);
            border: 1px solid rgba(148, 163, 184, 0.2);
        }
        .result-text {
            font-size: 1.5rem;
            font-weight: 700;
            margin: 0 0 6px;
        }
        .confidence {
            margin: 0;
            color: #cbd5e1;
            font-size: 0.98rem;
        }
        .spam { color: #fca5a5; }
        .ham { color: #86efac; }
        .neutral { color: #cbd5e1; }
    </style>
</head>
<body>
    <div class="card">
        <div class="header">
            <h1>SMS Spam Detector</h1>
            <span class="badge">NLP Model</span>
        </div>

        <form method="post">
            <textarea name="message" placeholder="Type an SMS message here...">{{ message }}</textarea>
            <div class="actions">
                <button type="submit">Check message</button>
            </div>
        </form>

        {% if result %}
            <div class="status">
                <p class="result-text {% if 'Spam' in result %}spam{% elif 'Ham' in result %}ham{% else %}neutral{% endif %}">Result: {{ result }}</p>
                <p class="confidence">Confidence: {{ confidence }}</p>
            </div>
        {% endif %}
    </div>
</body>
</html>
'''


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    confidence = None
    message = ''

    if request.method == 'POST':
        message = request.form.get('message', '')
        try:
            label, conf = predict_message(message)
            result = 'Spam' if label == 'spam' else 'Ham'
            confidence = f'{conf:.2%}'
        except FileNotFoundError as exc:
            result = str(exc)
            confidence = 'Train the model first'

    return render_template_string(HTML, result=result, confidence=confidence, message=message)


if __name__ == '__main__':
    app.run(debug=True)
