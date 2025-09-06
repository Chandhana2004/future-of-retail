from flask import Flask, render_template, request, jsonify, session, Response
from chatbot import chatbot_response
from utils import predict_sales
from utils import chatbot_response
import webbrowser
import threading
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Add this for session management

# Load dataset globally for EDA and chatbot
data = pd.read_csv('bigmart_data.csv')

@app.route("/")
def index():
    # Initialize chat log session on first visit
    session['chat_log'] = []
    return render_template("index.html")

@app.route('/chat', methods=['POST'])
def chat():
    data_json = request.get_json()
    user_msg = data_json.get("message", "")
    
    reply = chatbot_response(user_msg)

    # Save chat in session log
    chat_log = session.get('chat_log', [])
    chat_log.append(('You', user_msg))
    chat_log.append(('Bot', reply))
    session['chat_log'] = chat_log

    return jsonify({"reply": reply})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = request.json.get('features', [])
        features = [float(f) for f in features]
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid input features'}), 400
    prediction = predict_sales(features)
    return jsonify({'prediction': prediction})

@app.route("/eda_bigmart")
def eda_bigmart():
    return render_template("eda_bigmart.html")

@app.route("/eda_rossmann_train")
def eda_rossmann_train():
    return render_template("eda_rossmann_train.html")

@app.route("/eda_rossmann_store")
def eda_rossmann_store():
    return render_template("eda_rossmann_store.html")


### NEW ROUTES FOR SALES CHART & EXPORT ###

@app.route('/sales_chart')
def sales_chart():
    sales_by_type = data.groupby('Item_Type')['Item_Outlet_Sales'].sum().sort_values()
    plt.figure(figsize=(10,6))
    sales_by_type.plot(kind='bar')
    plt.title('Total Sales by Item Type')
    plt.ylabel('Sales')
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return render_template('sales_chart.html', plot_url=plot_url)

@app.route('/export_sales')
def export_sales():
    sales_summary = data.groupby('Item_Type')['Item_Outlet_Sales'].sum()
    csv_data = sales_summary.to_csv()
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=sales_summary.csv"})

@app.route('/export_chat')
def export_chat():
    chat_log = session.get('chat_log', [])
    csv_content = "Speaker,Message\n"
    for speaker, msg in chat_log:
        csv_content += f'"{speaker}","{msg}"\n'
    return Response(
        csv_content,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=chat_log.csv"})



def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")

if __name__ == "__main__":
    threading.Timer(1.5, open_browser).start()
    app.run(debug=True)