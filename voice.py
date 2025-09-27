import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import torch
import spacy
from spacy.matcher import Matcher
import re

# =====================================================
# 🔧 CONFIGURATION
# =====================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device utilisé : {device}")

fs = 16000
duration = 5
filename = "commande.wav"

# =====================================================
# 🎙️ ENREGISTREMENT + TRANSCRIPTION
# =====================================================
print("🎤 Enregistrement en cours... Parlez maintenant !")
audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
sd.wait()
write(filename, fs, audio)
print(f"✅ Enregistrement terminé ! Fichier audio sauvegardé sous : {filename}")

# Whisper
model = whisper.load_model("small", device=device)
print(f"⏳ Transcription en cours ({device})...")
result = model.transcribe(filename, language="fr", task="transcribe")
text_command = result["text"].strip()
print("📝 Texte détecté :", text_command)

# =====================================================
# 🤖 NLP avec spaCy
# =====================================================
nlp = spacy.load("fr_core_news_sm")
matcher = Matcher(nlp.vocab)

# Nettoyage du texte : minuscule + suppression ponctuation
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text

text_command_clean = clean_text(text_command)

# Lieux connus
locations = ["cuisine", "salon"]

# ----- Patterns -----
patterns = patterns = {
    "GO_TO_LOCATION": [
        {"LEMMA": "aller"},
        {"LOWER": {"IN": ["à", "au"]}},
        {"LOWER": {"IN": ["la", "le"]}, "OP": "?"},  # article optionnel correct
        {"LOWER": {"IN": locations}}
    ],
    "TURN_RIGHT": [
        {"LEMMA": "tourner"}, {"LOWER": "à"}, {"LOWER": "droite"}
    ],
    "TURN_LEFT": [
        {"LEMMA": "tourner"}, {"LOWER": "à"}, {"LOWER": "gauche"}
    ],
    "BACK": [
        {"LEMMA": "reculer"}
    ],
    "STOP": [
        {"LEMMA": "arrêter"}
    ]
}


# Ajouter les patterns au matcher
for intent_name, pattern in patterns.items():
    matcher.add(intent_name, [pattern])

# ----- Fonction parse_command -----
def parse_command(text):
    doc = nlp(text)
    matches = matcher(doc)

    for match_id, start, end in matches:
        intent_name = nlp.vocab.strings[match_id]

        # Gestion du GO_TO_LOCATION avec paramètre
        if intent_name == "GO_TO_LOCATION":
            for token in doc:
                if token.text.lower() in locations:
                    return {"intent": "go_to_location", "params": {"location": token.text.lower()}}
            return {"intent": "go_to_location", "params": {}}

        # Intents simples sans paramètre
        return {"intent": intent_name.lower(), "params": {}}

    return {"intent": "unknown", "params": {}}

# Analyse de la commande
parsed = parse_command(text_command_clean)
print("🎯 Intention détectée :", parsed)
