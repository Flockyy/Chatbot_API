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

router = APIRouter()

@router.get("/", response_model=List[schemas.Prediction])
def read_predictions(
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100,
) -> Any:
    """
    Retrieve predictions.
    """
    preds = crud.prediction.get_multi(db, skip=skip, limit=limit)
    return preds


@router.post("/", response_model=schemas.Prediction)
def create_prediction(
    *,
    db: Session = Depends(deps.get_db),
    pred_in: schemas.PredictionCreate,
) -> Any:
    """
    Create new prediction and answer user.
    """
    # GPU if available else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Will be deprecated
    with open('/home/CDG-NORD/florian-a/python-srv/src/pytorch_nn/intents.json', 'r') as json_data:
        intents = json.load(json_data)

    # Model import
    FILE = "/home/CDG-NORD/florian-a/fastapi-srv/src/data.pth"
    data = torch.load(FILE)

    # Model parameters
    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    tags = data['tags']
    model_state = data["model_state"]

    # Model recreation
    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()
    print(sentence)

    # Predict user sentence and answer
    sentence = tokenize(sentence.text)
    sentence = correct(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # END prediction

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                # Recreate answer with page url or document link

                result = random.choice(intent['responses'])
    else:
        result = "Je ne comprend pas..."
    print(result)
    return result
    # pred = crud.prediction.create(db=db, obj_in=pred_in)
    # return pred


@router.put("/{id}", response_model=schemas.Prediction)
def update_prediction(
    *,
    db: Session = Depends(deps.get_db),
    id: int,
    pred_in: schemas.PredictionUpdate,
) -> Any:
    """
    Update a prediction.
    """
    pred = crud.prediction.get(db=db, id=id)
    if not pred:
        raise HTTPException(status_code=404, detail="Prediction not found")
    pred = crud.prediction.update(db=db, db_obj=pred, obj_in=pred_in)
    return pred


@router.get("/{id}", response_model=schemas.Prediction)
def read_prediction(
    *,
    db: Session = Depends(deps.get_db),
    id: int,
) -> Any:
    """
    Get prediction by ID.
    """
    pred = crud.prediction.get(db=db, id=id)
    if not pred:
        raise HTTPException(status_code=404, detail="Prediction not found")

    return pred


@router.delete("/{id}", response_model=schemas.Prediction)
def delete_prediction(
    *,
    db: Session = Depends(deps.get_db),
    id: int,
) -> Any:
    """
    Delete an prediction.
    """
    pred = crud.prediction.get(db=db, id=id)
    if not pred:
        raise HTTPException(status_code=404, detail="Prediction not found")
    pred = crud.prediction.remove(db=db, id=id)
    return pred