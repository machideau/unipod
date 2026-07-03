import os
from google import genai
from google.genai import types
from PIL import Image

# Charger la clé API depuis le fichier .env si présent
if os.path.exists('.env'):
    with open('.env') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                parts = line.strip().split('=', 1)
                if len(parts) == 2:
                    os.environ[parts[0]] = parts[1]

# Initialisez le client (nécessite une clé API définie dans vos variables d'environnement : GEMINI_API_KEY)
client = genai.Client()

image_path = "image.png"
image = Image.open(image_path)

# Définition du prompt système pour un diagnostic très synthétique et résumé
system_prompt = (
    "Tu es un expert agronome et phytopathologiste. "
    "Ton rôle est d'analyser des images de plantes de manière scientifique mais en fournissant un rapport **très court, synthétique et résumé (va droit à l'essentiel, pas de longs paragraphes)**. "
    "Structure tes réponses strictement ainsi :\n\n"
    "**Plante** : Nom commun (sans nom scientifique).\n"
    "**État de santé** : Saine ou Malade.\n"
    "**Diagnostic** : Nom de la maladie ou du ravageur, agent responsable et niveau de certitude.\n"
    "**Symptômes clés** : Résumé en 1 ou 2 phrases courtes des principaux signes visibles sur l'image.\n"
    "**Recommandations majeures** : Liste de 2 ou 3 actions concrètes très courtes (maximum 5 mots par recommandation, sans parenthèses ni explications, pas de solutions exagérées qui peuvent nuire aux agriculteurs)."
)

# Poser la question en fournissant l'image et la configuration système
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=[
        image, 
        "Analyse cette image pour identifier la plante et diagnostiquer une éventuelle maladie."
    ],
    config=types.GenerateContentConfig(
        system_instruction=system_prompt
    )
)

print(response.text)
