from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes import speech

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(speech.router)

@app.get("/")
def root():
    return {"message": "Sambhav backend running"}