from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
import numpy as np
import pickle
from io import BytesIO
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Apply CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Load the preprocessed mapping and tokenizer
with open("mapping.pkl", "rb") as f:
    mapping = pickle.load(f)

def all_captions(mapping):
    return [caption for key in mapping for caption in mapping[key]]

all_captions = all_captions(mapping)

def create_token(all_captions):
    from tensorflow.keras.preprocessing.text import Tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    return tokenizer

tokenizer = create_token(all_captions)
max_length = 35

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    repeated_word_count = 0
    previous_word = None
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        
        if word is None or word == 'endseq' or (word == previous_word and repeated_word_count > 2):
            break
        
        in_text += " " + word
        
        if word == previous_word:
            repeated_word_count += 1
        else:
            repeated_word_count = 0
        previous_word = word

    return in_text.strip('startseq ').strip()

# Load the VGG16 model and the captioning model
vgg_model = VGG16()
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
model = load_model("model.keras")

@app.post("/generate-caption/")
async def generate_caption(file: UploadFile = File(...)):
    try:
        # Load and preprocess the image
        image = Image.open(BytesIO(await file.read()))
        image = image.resize((224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = vgg_model.predict(image, verbose=0)

        # Generate caption
        caption = predict_caption(model, feature, tokenizer, max_length)
        
        return JSONResponse(content={"caption": caption})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the app using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
