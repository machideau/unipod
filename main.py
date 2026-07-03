import os

EXPERTS = [
    {
        "nom": "Dr. Kokou Mensah",
        "specialite": "Maladies fongiques du cacao et du manioc",
        "zone": "Kara, Togo",
        "contact": "+228 90 11 22 33"
    },
    {
        "nom": "Dr. Abiba Traoré",
        "specialite": "Phytopathologie du maïs et du sorgho, maladies virales des céréales",
        "zone": "Lomé, Togo",
        "contact": "+228 91 33 44 55"
    },
    {
        "nom": "Ing. Séverin Agbéko",
        "specialite": "Fertilité des sols et nutrition des plantes, agronomie des cultures vivrières",
        "zone": "Sokodé, Togo",
        "contact": "+228 92 55 66 77"
    },
    {
        "nom": "Dr. Fatoumata Coulibaly",
        "specialite": "Ravageurs et insectes nuisibles (légumineuses, coton, maraîchage)",
        "zone": "Atakpamé, Togo",
        "contact": "+228 93 77 88 99"
    },
    {
        "nom": "Ing. Yves Koffi Adzodo",
        "specialite": "Agriculture durable, agroécologie et compostage",
        "zone": "Tsévié, Togo",
        "contact": "+228 90 22 33 44"
    },
    {
        "nom": "Dr. Madeleine Amouzou",
        "specialite": "Maladies bactériennes et virales des tomates, poivrons et aubergines",
        "zone": "Lomé, Togo",
        "contact": "+228 91 44 55 66"
    },
    {
        "nom": "Ing. Kossi Djédjé",
        "specialite": "Irrigation, gestion de l'eau agricole et drainage",
        "zone": "Kpalimé, Togo",
        "contact": "+228 92 66 77 88"
    },
    {
        "nom": "Dr. Rébecca Sossah",
        "specialite": "Maladies des cultures de palmier à huile et cocotier",
        "zone": "Aného, Togo",
        "contact": "+228 93 88 99 00"
    },
    {
        "nom": "Ing. Komlan Attivor",
        "specialite": "Protection des cultures maraîchères (choux, laitue, oignon, carotte)",
        "zone": "Dapaong, Togo",
        "contact": "+228 90 44 55 66"
    },
    {
        "nom": "Dr. Saliou Issifou",
        "specialite": "Phytopathologie du coton et des légumineuses (niébé, arachide, soja)",
        "zone": "Bassar, Togo",
        "contact": "+228 91 66 77 88"
    },
    {
        "nom": "Ing. Pauline Gnagna Koudjo",
        "specialite": "Semences améliorées, sélection variétale et stockage post-récolte",
        "zone": "Notsé, Togo",
        "contact": "+228 92 88 99 11"
    },
    {
        "nom": "Dr. Aristide Hounsou",
        "specialite": "Maladies du bananier et du plantain (cercosporiose, fusariose)",
        "zone": "Kpalimé, Togo",
        "contact": "+228 93 99 00 11"
    },
    {
        "nom": "Ing. Akossiwa Dossou",
        "specialite": "Élevage et santé animale, agroélevage mixte",
        "zone": "Lomé, Togo",
        "contact": "+228 90 33 44 55"
    },
    {
        "nom": "Dr. Tchilabalo Pali",
        "specialite": "Gestion intégrée des ravageurs (GIR), entomologie agricole",
        "zone": "Kara, Togo",
        "contact": "+228 91 55 66 77"
    },
    {
        "nom": "Ing. Mawuénam Akpene",
        "specialite": "Cultures sous serre, maraîchage intensif et hydroponique",
        "zone": "Lomé, Togo",
        "contact": "+228 92 77 88 99"
    }
]

# Formater la liste en texte brut pour les prompts système
experts_list = "\n---\n".join([
    f"Nom: {e['nom']}\nSpécialité: {e['specialite']}\nZone: {e['zone']}\nContact: {e['contact']}"
    for e in EXPERTS
])
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

# Charger toutes les clés API Gemini disponibles depuis le fichier .env
def load_api_keys() -> list[str]:
    if os.path.exists('.env'):
        with open('.env', encoding='utf-8') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    parts = line.strip().split('=', 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        val = parts[1].strip().strip('"').strip("'")
                        os.environ[key] = val

    keys = []
    
    # 1. Tenter de lire sous forme de liste (séparée par des virgules ou au format JSON) dans GEMINI_API_KEYS
    env_keys = os.environ.get("GEMINI_API_KEYS", "")
    if env_keys:
        try:
            import json
            parsed = json.loads(env_keys)
            if isinstance(parsed, list):
                keys.extend([str(k).strip() for k in parsed if k])
        except Exception:
            keys.extend([k.strip() for k in env_keys.split(",") if k.strip()])
            
    # 2. Tenter de lire les clés individuelles
    single = os.environ.get("GEMINI_API_KEY", "")
    if single and single not in keys:
        keys.append(single)
        
    i = 1
    while True:
        k = os.environ.get(f"GEMINI_API_KEY_{i}", "")
        if not k:
            break
        if k not in keys:
            keys.append(k)
        i += 1
        
    if not keys:
        raise RuntimeError("Aucune clé API Gemini trouvée dans le fichier .env ou les variables d'environnement.")
    return keys

API_KEYS = load_api_keys()

def gemini_generate_with_fallback(**kwargs):
    """Appelle l'API Gemini en essayant chaque clé disponible en cas d'échec."""
    last_error = None
    for key in API_KEYS:
        try:
            client = genai.Client(api_key=key)
            return client.models.generate_content(**kwargs)
        except Exception as e:
            last_error = e
            continue
    raise last_error

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

# Stockage des sessions en mémoire (session_id -> historique messages)
sessions: dict[str, list] = {}

class Expert(BaseModel):
    nom: str = Field(description="Nom de l'expert")
    specialite: str = Field(description="Spécialité de l'expert")
    zone: str = Field(description="Zone géographique de l'expert")
    contact: str = Field(description="Numéro de contact de l'expert")

class DiagnosticReport(BaseModel):
    resume: str = Field(description="Résumé de l'analyse, strictement limité à 2 phrases courtes et maximum 30 mots au total.")
    experts: list[Expert] = Field(default=[], description="Liste des experts recommandés pour cette pathologie ou cette plante, choisis parmi la liste des experts fournis.")

class ChatResponse(BaseModel):
    reply: str = Field(description="Réponse de l'assistant agricole.")
    session_id: str = Field(description="Identifiant de session pour continuer la conversation.")

class DeleteSessionResponse(BaseModel):
    message: str
 
diagnostic_system_prompt = (
    "Tu es un expert agronome et phytopathologiste expérimenté, habitué à conseiller des agriculteurs de terrain. "
    "Analyse l'image pour le diagnostic.\n\n"
    "1. Rédige un diagnostic ultra-court dans le champ 'resume'. Il doit contenir STRICTEMENT deux phrases courtes et maximum 30 mots au total :\n"
    "   - Phrase 1 : Identifie la plante, son état de santé et la maladie/ravageur.\n"
    "   - Phrase 2 : Donne l'unique recommandation principale réaliste (selon les règles ci-dessous).\n"
    "   Ne fais aucune introduction, aucune transition, aucun bavardage. Va droit au but.\n\n"
    "RÈGLES POUR LA RECOMMANDATION :\n"
    "- Observe d'abord le stade de croissance visible sur l'image (jeune plant, plant en croissance, plant adulte/mature, plant en floraison ou fructification).\n"
    "- Adapte impérativement ton conseil à ce stade : une action faisable sur un jeune plant peut être totalement irréaliste sur un plant adulte.\n"
    "- N'indique JAMAIS d'arracher, détruire, ou replanter si la plante semble être à un stade avancé (adulte, en fleur ou en fruit) — ce serait une perte économique inacceptable pour l'agriculteur.\n"
    "- Pour les plants adultes ou avancés : oriente vers la gestion de la maladie sur place (suppression des parties atteintes, traitement localisé, limitation de la propagation, consultation d'un agent agricole).\n"
    "- Pour les jeunes plants : des actions plus radicales (isolement, remplacement) peuvent être envisagées si justifiées.\n"
    "- Reste toujours réaliste, faisable et économiquement responsable. N'exagère pas la gravité.\n"
    "- Ne mentionne jamais de produit chimique précis, de dosage ou de quantité.\n\n"
    "2. Sélectionne dans la liste ci-dessous le ou les experts dont la spécialité correspond au diagnostic et remplis le champ 'experts'. "
    "Ne propose aucun expert absent de cette liste. Si aucun n'est pertinent, laisse la liste vide.\n\n"
    f"Liste des experts disponibles :\n{experts_list}"
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
        # 3. Appel de l'API Gemini avec Structured Output (+ fallback multi-clés)
        response = gemini_generate_with_fallback(
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
        # 4. Appel Gemini avec historique complet (+ fallback multi-clés)
        response = gemini_generate_with_fallback(
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
