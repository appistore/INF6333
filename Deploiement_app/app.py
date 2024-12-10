from flask import Flask, request, render_template
from transformers import MarianMTModel, MarianTokenizer
import torch

app = Flask(__name__)

# Charger le modèle et le tokenizer
model_name = "model"  # Répertoire où le modèle est sauvegardé
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route("/")
def home():
    """
    Affiche la page principale avec le formulaire de traduction.
    """
    return render_template("index.html")

@app.route("/translate", methods=["POST"])
def translate():
    """
    Traitement de la traduction via le formulaire HTML.
    """
    try:
        # Récupérer le texte du formulaire
        text_to_translate = request.form.get("text")
        if not text_to_translate:
            return render_template("index.html", error="Veuillez entrer une phrase à traduire.")

        # Traduire le texte
        inputs = tokenizer(text_to_translate, return_tensors="pt", truncation=True, padding=True).to(device)
        translated_tokens = model.generate(**inputs)
        translation = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

        # Retourner la traduction à l'interface
        return render_template("index.html", original_text=text_to_translate, translated_text=translation)
    
    except Exception as e:
        return render_template("index.html", error=f"Erreur : {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
