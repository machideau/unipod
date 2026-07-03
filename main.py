import os
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
    plante: str = Field(description="Nom commun de la plante (sans nom scientifique).")
    etat_de_sante: str = Field(description="Saine ou Malade.")
    diagnostic: str = Field(description="Nom de la maladie ou du ravageur, agent responsable et niveau de certitude.")
    symptomes_cles: str = Field(description="Résumé en 1 ou 2 phrases courtes des principaux signes visibles sur l'image.")
    recommandations_majeures: list[str] = Field(description="Liste de 2 ou 3 actions concrètes très courtes (maximum 5 mots par recommandation, sans parenthèses ni explications, pas de solutions exagérées qui peuvent nuire aux agriculteurs).")

class ChatResponse(BaseModel):
    reply: str = Field(description="Réponse de l'assistant agricole.")
    session_id: str = Field(description="Identifiant de session pour continuer la conversation.")

class DeleteSessionResponse(BaseModel):
    message: str

diagnostic_system_prompt = (
    "Tu es un expert d'agriculture et phytopathologiste. "
    "Ton rôle est d'analyser des images de plantes de manière scientifique et de remplir le schéma JSON requis avec précision. "
    "Sois court et va droit à l'essentiel dans les descriptions."
)

chat_system_prompt = (
    "Tu es un assistant agricole expert et bienveillant, spécialisé en agronomie, phytopathologie et agriculture durable. "
    "Tu aides les agriculteurs à diagnostiquer les maladies de leurs plantes, à choisir des traitements adaptés et à améliorer leurs pratiques. "
    "Lorsqu'une image est fournie, analyse-la attentivement avant de répondre. "
    "Tes réponses sont claires, pratiques et adaptées aux agriculteurs de terrain. "
    "Réponds toujours dans la langue de l'utilisateur."
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
        # 3. Appel de l'API Gemini avec Structured Output
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                image,
                "Analyse cette image pour identifier la plante et diagnostiquer une éventuelle maladie."
            ],
            config=types.GenerateContentConfig(
                system_instruction=diagnostic_system_prompt,
                response_mime_type="application/json",
                response_schema=DiagnosticReport,
            )
        )

        # 4. Parsing et retour du résultat
        result_dict = json.loads(response.text)
        return result_dict
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
