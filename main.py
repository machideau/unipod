import os

def load_experts_list(filepath: str = "experts.txt") -> str:
    if os.path.exists(filepath):
        with open(filepath, encoding="utf-8") as f:
            return f.read().strip()
    return "Aucun expert disponible pour le moment."

experts_list = load_experts_list()
import io
import json
import uuid
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
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

app = FastAPI(
    title="E-Farm API",
    description="API de diagnostic phytosanitaire et assistant agricole basée sur Gemini 2.5 Flash",
    version="2.0.0"
)

# Configuration CORS pour permettre les connexions depuis des applications mobiles ou web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialisation du client GenAI
client = genai.Client()

# Stockage des sessions en mémoire (session_id -> historique messages)
sessions: dict[str, list] = {}

class DiagnosticReport(BaseModel):
    resume: str = Field(description="Résumé en prose de l'analyse de la plante : nom, état de santé, diagnostic, symptômes et recommandations, rédigé en un ou deux paragraphes naturels et fluides.")

class ChatResponse(BaseModel):
    reply: str = Field(description="Réponse de l'assistant agricole.")
    session_id: str = Field(description="Identifiant de session pour continuer la conversation.")

class DeleteSessionResponse(BaseModel):
    message: str

diagnostic_system_prompt = (
    "Tu es un expert d'agriculture et phytopathologiste. "
    "Analyse l'image de manière scientifique et rédige un résumé en prose naturelle, "
    "comme si tu expliquais ton diagnostic à voix haute à un agriculteur. "
    "Ne liste pas des champs séparés (plante: X, état: Y) — intègre tout dans un texte fluide et cohérent. "
    "Mentionne la plante, son état de santé, le diagnostic précis (maladie/ravageur, agent responsable, niveau de certitude) "
    "et les symptômes visibles, en 3 à 5 phrases. "
    "Sois concis et va droit à l'essentiel.\n\n"
    "Pour les recommandations : "
    "formule des actions professionnelles et prudentes, comme un agronome s'adressant à un agriculteur non-spécialiste. "
    "N'indique jamais de nom de produit chimique précis, de dosage, ni de quantité — recommande plutôt de consulter "
    "un agent agricole local ou un revendeur de produits phytosanitaires certifié pour le traitement adapté. "
    "Privilégie les pratiques culturales sûres et éprouvées (retrait des parties infectées, rotation des cultures, "
    "espacement, drainage, désinfection des outils) avant toute mention de traitement chimique. "
    "Ne suggère jamais d'action pouvant nuire à la santé de l'agriculteur, à l'environnement, ou aggraver la situation "
    "si mal appliquée. "
    "Reste factuel et évite tout ton alarmiste ou exagéré."
)

chat_system_prompt = (
    "Tu es un assistant agricole expert et bienveillant, spécialisé en agronomie, phytopathologie et agriculture durable. "
    "Tu aides les agriculteurs à diagnostiquer les maladies de leurs plantes, à choisir des pratiques adaptées et à améliorer leurs pratiques. "
    "Lorsqu'une image est fournie, analyse-la attentivement avant de répondre. "
    "Tes réponses sont claires, pratiques et adaptées aux agriculteurs de terrain. "
    "Réponds toujours dans la langue de l'utilisateur.\n\n"
    "Si l'utilisateur demande à parler à un expert, à être mis en contact avec un agronome, "
    "ou à obtenir un avis humain professionnel, propose-lui un ou plusieurs experts "
    "de la liste ci-dessous, en choisissant celui dont la spécialité correspond le mieux à son besoin. "
    "Ne propose jamais un expert qui n'est pas dans cette liste. "
    "Si la liste est vide ou ne contient aucun profil pertinent, dis-le clairement et ne propose rien.\n\n"
    f"Liste des experts disponibles :\n{experts_list}"
)

@app.post(
    "/diagnose",
    response_model=DiagnosticReport,
    summary="Diagnostiquer l'état d'une plante à partir d'une image",
    tags=["Diagnostic"]
)
async def diagnose_plant(file: UploadFile = File(...)):
    # 1. Vérification du format du fichier
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier téléversé doit être une image.")

    try:
        # 2. Lecture et conversion en image PIL
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image invalide ou corrompue: {str(e)}")

    try:
        # 3. Appel de l'API Gemini avec résumé en prose
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                image,
                "Analyse cette image pour identifier la plante et diagnostiquer une éventuelle maladie."
            ],
            config=types.GenerateContentConfig(
                system_instruction=diagnostic_system_prompt,
            )
        )

        # 4. Retour du résumé en prose
        return DiagnosticReport(resume=response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'appel à l'API Gemini : {str(e)}")

@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Discuter avec l'assistant agricole (image optionnelle + texte)",
    tags=["Chat"]
)
async def chat(
    message: str = Form(..., description="Message texte de l'utilisateur"),
    session_id: Optional[str] = Form(None, description="ID de session pour continuer une conversation existante"),
    file: Optional[UploadFile] = File(None, description="Image de plante (optionnelle)")
):
    # 1. Récupérer ou créer une session
    if session_id and session_id in sessions:
        history = sessions[session_id]
    else:
        session_id = str(uuid.uuid4())
        history = []

    # 2. Construire le contenu du message utilisateur
    user_parts = []

    if file is not None:
        if file.content_type and not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Le fichier téléversé doit être une image.")
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            user_parts.append(image)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Image invalide ou corrompue: {str(e)}")

    user_parts.append(message)

    # 3. Ajouter le message utilisateur à l'historique
    history.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=p) if isinstance(p, str) else types.Part.from_image(image=p) for p in user_parts]
        )
    )

    try:
        # 4. Appel Gemini avec historique complet
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=history,
            config=types.GenerateContentConfig(
                system_instruction=chat_system_prompt,
            )
        )

        reply_text = response.text

        # 5. Ajouter la réponse de l'IA à l'historique
        history.append(
            types.Content(
                role="model",
                parts=[types.Part.from_text(text=reply_text)]
            )
        )

        # 6. Sauvegarder la session mise à jour
        sessions[session_id] = history

        return ChatResponse(reply=reply_text, session_id=session_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'appel à l'API Gemini : {str(e)}")


@app.delete(
    "/chat/{session_id}",
    response_model=DeleteSessionResponse,
    summary="Réinitialiser une session de conversation",
    tags=["Chat"]
)
async def delete_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
        return DeleteSessionResponse(message=f"Session {session_id} supprimée avec succès.")
    raise HTTPException(status_code=404, detail=f"Session '{session_id}' introuvable.")

@app.get("/health", summary="Vérification de l'état de l'API", tags=["Système"])
async def health_check():
    return {"status": "ok", "api": "E-Farm API", "version": "2.0.0"}
