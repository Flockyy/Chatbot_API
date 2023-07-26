from typing import Any, List
import json
import torch
import random

from ai.model import NeuralNet
from ai.nltk_utils import bag_of_words, tokenize, correct

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src import crud
from src import schemas
from src.api import deps

import nltk

nltk.download("punkt")

router = APIRouter()

@router.get("/sentry-debug")
async def trigger_error():
    division_by_zero = 1 / 0

@router.get("/", response_model=List[schemas.Chat])
def read_chats(
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100,
) -> Any:
    """
    Retrieve Chats.
    """
    chats = crud.chat.get_multi(db, skip=skip, limit=limit)
    return chats


@router.post("/", response_model=schemas.Answer)
def create_chat(
    *,
    db: Session = Depends(deps.get_db),
    question_in: schemas.QuestionCreate,
    answer_in: schemas.QuestionCreate,
    chat_in: schemas.ChatCreate
) -> Any:
    """
    Create new chat and answer user.
    """

    # GPU if available else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Will be deprecated
    with open(
        "/home/CDG-NORD/florian-a/python-srv/src/dl_model/pytorch_nn/intents.json", "r"
    ) as json_data:
        intents = json.load(json_data)

    # Model import
    FILE = "/home/CDG-NORD/florian-a/python-srv/src/data.pth"
    data = torch.load(FILE)

    # Model parameters
    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data["all_words"]
    tags = data["tags"]
    model_state = data["model_state"]

    # Model recreation
    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()

    # Predict user sentence and answer
    question_in = tokenize(question_in.text)
    question_in = correct(question_in)
    X = bag_of_words(question_in, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # END prediction

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                # Recreate answer with page url or document link
                result = random.choice(intent["responses"])
    else:
        result = "Je ne comprend pas..."
    print(result)
    # answer = {"id": 1, "text": result}

    chat_in = crud.chat.create(db, obj_in=chat_in)
    question = crud.question.create_with_owner(db, obj_in=question_in ,related_chat_id=chat_in.id)
    answer_in.text = result
    answer = crud.question.create_with_owner(db, obj_in=answer_in ,related_chat_id=chat_in.id)

    return answer
    # pred = crud.prediction.create(db=db, obj_in=pred_in)
    # return pred


@router.put("/{id}", response_model=schemas.Chat)
def update_chat(
    *,
    db: Session = Depends(deps.get_db),
    id: int,
    chat_in: schemas.ChatUpdate,
) -> Any:
    """
    Update a chat.
    """
    chat = crud.chat.get(db=db, id=id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    chat = crud.chat.update(db=db, db_obj=chat, obj_in=chat_in)
    return chat


@router.get("/{id}", response_model=schemas.Chat)
def read_chat(
    *,
    db: Session = Depends(deps.get_db),
    id: int,
) -> Any:
    """
    Get chat by ID.
    """
    chat = crud.chat.get(db=db, id=id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    return chat


@router.delete("/{id}", response_model=schemas.Chat)
def delete_chat(
    *,
    db: Session = Depends(deps.get_db),
    id: int,
) -> Any:
    """
    Delete an chat.
    """
    chat = crud.chat.get(db=db, id=id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    chat = crud.chat.remove(db=db, id=id)
    return chat
