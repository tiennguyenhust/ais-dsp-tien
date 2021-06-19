import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import List

app = FastAPI()
model = joblib.load('../../models/diabetes_model.joblib')


class DiabetesInfo(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float



@app.post('/predict')
async def predict_diabetes_progress(
        age: float, sex: float, bmi: float, bp: float, s1: float, s2: float, s3: float, s4: float, s5: float,
        s6: float):
    # age, sex, body_mass_index, average_blood_pressure, total_serum_cholesterol, low_density_lipoproteins,
    # high_density_lipoproteins, total_cholesterol, possibly_log_of_serum_triglycerides_level, blood_sugar_level
    print(age, sex, bmi, bp, s1, s2, s3, s4, s5, s6)
    model_input_data = np.array([age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]).reshape(1, -1)
    progression = model.predict(model_input_data)
    print(type(progression))
    return progression[0]


@app.post('/predict_obj')
async def predict_diabetes_progress_1(diabeteses: List[DiabetesInfo]):
    # age, sex, body_mass_index, average_blood_pressure, total_serum_cholesterol, low_density_lipoproteins,
    # high_density_lipoproteins, total_cholesterol, possibly_log_of_serum_triglycerides_level, blood_sugar_level

    model_input_data = pd.DataFrame([d.dict() for d in diabeteses])
    
    progression = model.predict(model_input_data)
    return ' '.join(map(str, progression))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""
To cover:
- possible return types: not nympy array, etc
"""