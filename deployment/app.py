from fastapi import FastAPI
from fastapi.responses import JSONResponse
from schema.user_input_output import GrammarRequest, GrammarResponse, GrammarResponseDetailed
from model.predict import MODEL_VERSION, predict_output, predict_multiple, get_all_candidates_with_scores
from fastapi.middleware.cors import CORSMiddleware
from schema.user_input_output import GrammarRequest, GrammarResponseDetailed
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Nepali Grammar Correction API")

# Serve frontend
# app.mount("/", StaticFiles(directory="deployment", html=True), name="static")

# ✅ This MUST be added before any routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
    
@app.get("/")
def home():
    return {'message' : 'Welcome to Nepali Grammar Correction API homepage'}


@app.get("/health")
def health_checkup():
    return {
        'status' : 'OK',
        'version' : MODEL_VERSION,
        # 'model_loaded' : model is not None
    }


@app.post("/correct", response_model=GrammarResponseDetailed)
def correct_grammar(req: GrammarRequest):

    input_text = req.text.strip()
    model_choice = req.model

    if not input_text:
        return {"results": []}

    try:

        sentences = input_text.split("\n")
        results = []

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                results.append({
                    "input": "",
                    "best_output": "",
                    "all_candidates": []
                })
                continue

            all_results = get_all_candidates_with_scores(sentence, model_choice=model_choice)

            print(f"\nInput: {sentence}")
            for result in all_results:
                print(f"Rank {result['rank']}: {result['sentence']} (wins: {result['wins']})")
                
            results.append({
                "input": sentence,
                "best_output": all_results[0]['sentence'],
                "all_candidates": all_results
            })

        return JSONResponse(status_code=200, content={"results": results})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

