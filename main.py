from fastapi import FastAPI, UploadFile, File
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
import re
from deepface import DeepFace
from fastapi import FastAPI, UploadFile, File
import os
import uuid
import numpy as np
import asyncio
import json
import whisper
from pydantic import BaseModel


app = FastAPI()
processor = DonutProcessor.from_pretrained(
    "naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained(
    "naver-clova-ix/donut-base-finetuned-cord-v2")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and process the uploaded image
    image = Image.open(file.file)
    if image.mode != "RGB":
        image = image.convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values

    # Generate the sequence
    task_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
        output_scores=True,
    )
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(
        processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()

    return {"result": processor.token2json(sequence)}



models = [
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    "DeepFace",
    "DeepID",
    "ArcFace",
    "Dlib",
    "SFace",
]
metrics = ["cosine", "euclidean", "euclidean_l2"]
backends = [
    'opencv',
    'ssd',
    'dlib',
    'mtcnn',
    'retinaface',
    'mediapipe'
]

async def process_verification(source_filename, target_filename):
    result = DeepFace.verify(img1_path=source_filename,
                             img2_path=target_filename,
                             model_name=models[1],
                             distance_metric=metrics[0],
                             detector_backend=backends[1]
                             )
    return result


@app.post("/verify/images")
async def process_images(source_image: UploadFile = File(...), target_image: UploadFile = File(...)):

    source_filename = str(uuid.uuid4()) + "_" + str(source_image.filename)
    target_filename = str(uuid.uuid4()) + "_" + str(target_image.filename)

    with open(source_filename, 'wb+') as f:
        f.write(source_image.file.read())
        f.close()
    with open(target_filename, 'wb+') as f:
        f.write(target_image.file.read())
        f.close()

    response = await process_verification(source_filename, target_filename)

    # Delete the source and target images
    os.remove(source_filename)
    os.remove(target_filename)
    return {"message": str(response)}




# Define the request model
class AudioRequest(BaseModel):
    audio_file: UploadFile

# Define a temporary directory to store uploaded files
TEMP_DIRECTORY = "./temp"

@app.post("/convert_audio")
async def convert_audio_to_text(request: AudioRequest):
    # Create a temporary directory if it doesn't exist
    os.makedirs(TEMP_DIRECTORY, exist_ok=True)

    # Save the uploaded audio file to the temporary directory
    audio_path = os.path.join(TEMP_DIRECTORY, request.audio_file.filename)
    with open(audio_path, "wb") as audio_file:
        shutil.copyfileobj(request.audio_file.file, audio_file)

    # Convert audio to text using OpenAI

    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    print(result["text"])
    os.remove(audio_path)

    return {"text": result["text"]}
