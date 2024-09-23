from flask import Flask, request, jsonify
from model_inference import SentimentModel

app = Flask(__name__)

# Initialisiere das Modell mit den Pfaden zu den Gewichten und Konfigurationsdateien
model_weights_path = 'weights_final.h5'
config_path = 'config/config-roberta-base.json'
vocab_path = 'config/vocab-roberta-base.json'
merges_path = 'config/merges-roberta-base.txt'
model = SentimentModel(model_weights_path, config_path, vocab_path, merges_path)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Erwartet wird eine JSON-Nachricht mit 'text' und 'sentiment'
        data = request.json
        text = data.get('text')
        sentiment = data.get('sentiment')
        print(data)

        if not text or not sentiment:
            return jsonify({'Fehler': 'Ungültige Eingabe: Bitte sowohl Text als auch Sentiment eingeben.'}), 400

        # Modellvorhersage
        result = model.predict(text, sentiment)
        return jsonify({'selected_text': result})
    except Exception as e:
        return jsonify({'Fehler': str(e)}), 500


#if __name__ == '__main__':
#    app.run(host='127.0.0.1', port=5000)
