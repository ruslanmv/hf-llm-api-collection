import argparse
import uvicorn
import sys
import os
import io
from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import time
import json
from typing import List
import torch
import logging
import string
import random
import base64
import re
import requests
from utils.enver import enver
import shutil
import tempfile
import numpy as np


from fastapi import FastAPI, Response, File, UploadFile, Form
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from utils.logger import logger
from networks.message_streamer import MessageStreamer
from messagers.message_composer import MessageComposer
from googletrans import Translator
from io import BytesIO
from gtts import gTTS
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from tempfile import NamedTemporaryFile

class ChatAPIApp:
    def __init__(self):
        self.app = FastAPI(
            docs_url="/",
            title="HuggingFace LLM API",
            swagger_ui_parameters={"defaultModelsExpandDepth": -1},
            version="1.0",
        )
        self.setup_routes()

    def get_available_langs(self):
        f = open('apis/lang_name.json', "r")
        self.available_models = json.loads(f.read())
        return self.available_models

    class TranslateCompletionsPostItem(BaseModel):
        from_language: str = Field(
            default="en",
            description="(str) `Detect`",
        )
        to_language: str = Field(
            default="fa",
            description="(str) `en`",
        )
        input_text: str = Field(
            default="Hello",
            description="(str) `Text for translate`",
        )
   

    def translate_completions(self, item: TranslateCompletionsPostItem):
        translator = Translator()
        f = open('apis/lang_name.json', "r")
        available_langs = json.loads(f.read())
        from_lang = 'en'
        to_lang = 'en'
        for lang_item in available_langs:
          if item.to_language == lang_item['code']:
              to_lang = item.to_language
              break
              
          
        translated = translator.translate(item.input_text, dest=to_lang)
        item_response = {
            "from_language": translated.src,
            "to_language": translated.dest,
            "text": item.input_text,
            "translate": translated.text
        }
        json_compatible_item_data = jsonable_encoder(item_response)
        return JSONResponse(content=json_compatible_item_data)

    def translate_ai_completions(self, item: TranslateCompletionsPostItem):
        translator = Translator()
        #print(os.getcwd())
        f = open('apis/lang_name.json', "r")
        available_langs = json.loads(f.read())
        from_lang = 'en'
        to_lang = 'en'
        for lang_item in available_langs:
          if item.to_language == lang_item['code']:
              to_lang = item.to_language
          if item.from_language == lang_item['code']:
              from_lang = item.from_language

        if to_lang == 'auto':
            to_lang = 'en'

        if from_lang == 'auto':
            from_lang = translator.detect(item.input_text).lang
            
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
            logging.warning("GPU not found, using CPU, translation will be very slow.")

        time_start = time.time()
        #TRANSFORMERS_CACHE
        pretrained_model = "facebook/m2m100_1.2B"
        cache_dir = "models/"
        tokenizer = M2M100Tokenizer.from_pretrained(pretrained_model, cache_dir=cache_dir)
        model = M2M100ForConditionalGeneration.from_pretrained(
            pretrained_model, cache_dir=cache_dir
        ).to(device)
        model.eval()

        tokenizer.src_lang = from_lang
        with torch.no_grad():
            encoded_input = tokenizer(item.input_text, return_tensors="pt").to(device)
            generated_tokens = model.generate(
               **encoded_input, forced_bos_token_id=tokenizer.get_lang_id(to_lang)
            )
            translated_text = tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
            )[0]

        time_end = time.time()
        translated = translated_text
        item_response = {
            "from_language": from_lang,
            "to_language": to_lang,
            "text": item.input_text,
            "translate": translated,
            "start": str(time_start),
            "end": str(time_end)
        }
        json_compatible_item_data = jsonable_encoder(item_response)
        return JSONResponse(content=json_compatible_item_data)

    class TranslateAiPostItem(BaseModel):
        model: str = Field(
            default="t5-base",
            description="(str) `Model Name`",
        )
        from_language: str = Field(
            default="en",
            description="(str) `translate from`",
        )
        to_language: str = Field(
            default="fa",
            description="(str) `translate to`",
        )
        input_text: str = Field(
            default="Hello",
            description="(str) `Text for translate`",
        )    
    def ai_translate(self, item:TranslateAiPostItem):
        MODEL_MAP = {
        "t5-base": "t5-base",
        "t5-small": "t5-small",
        "t5-large": "t5-large",
        "t5-3b": "t5-3b",
        "mbart-large-50-many-to-many-mmt": "facebook/mbart-large-50-many-to-many-mmt",
        "nllb-200-distilled-600M": "facebook/nllb-200-distilled-600M",
        "madlad400-3b-mt": "jbochi/madlad400-3b-mt",    
        "default": "t5-base",
        }
        if item.model in MODEL_MAP.keys():
            target_model = item.model
        else:
            target_model = "default"

        real_name = MODEL_MAP[target_model]
        read_model = AutoModelForSeq2SeqLM.from_pretrained(real_name)
        tokenizer = AutoTokenizer.from_pretrained(real_name)
        #translator = pipeline("translation", model=read_model, tokenizer=tokenizer, src_lang=item.from_language, tgt_lang=item.to_language)
        translate_query = (
            f"translation_{item.from_language}_to_{item.to_language}"
        )
        translator = pipeline(translate_query)
        result = translator(item.input_text)    
           
        item_response = {
            "statue": 200,
            "result": result,
            }
        json_compatible_item_data = jsonable_encoder(item_response)
        return JSONResponse(content=json_compatible_item_data)
    class DetectLanguagePostItem(BaseModel):
        input_text: str = Field(
            default="Hello, how are you?",
            description="(str) `Text for detection`",
        )

    def detect_language(self, item: DetectLanguagePostItem):
        translator = Translator()
        detected = translator.detect(item.input_text)

        item_response = {
            "lang": detected.lang,
            "confidence": detected.confidence,
        }
        json_compatible_item_data = jsonable_encoder(item_response)
        return JSONResponse(content=json_compatible_item_data)
        
    class TTSPostItem(BaseModel):
        input_text: str = Field(
            default="Hello",
            description="(str) `Text for TTS`",
        )
        from_language: str = Field(
            default="en",
            description="(str) `TTS language`",
        )
        
    def text_to_speech(self, item: TTSPostItem):
        try:
            audioobj = gTTS(text = item.input_text, lang = item.from_language, slow = False)
            fileName = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(10));
            fileName = fileName + ".mp3";
            mp3_fp = BytesIO()
            #audioobj.save(fileName)
            #audioobj.write_to_fp(mp3_fp)
            #buffer = bytearray(mp3_fp.read())
            #base64EncodedStr = base64.encodebytes(buffer)
            #mp3_fp.read()
            #return Response(content=mp3_fp.tell(), media_type="audio/mpeg")
            return StreamingResponse(audioobj.stream())
        except:
               item_response = {
                 "status": 400
               }
               json_compatible_item_data = jsonable_encoder(item_response)
               return JSONResponse(content=json_compatible_item_data)
           
        
    def setup_routes(self):
        for prefix in ["", "/v1"]:
            self.app.get(
                prefix + "/langs",
                summary="Get available languages",
            )(self.get_available_langs)

            self.app.post(
                prefix + "/translate",
                summary="translate text",
            )(self.translate_completions)

            self.app.post(
                prefix + "/translate/ai",
                summary="translate text with ai",
            )(self.translate_ai_completions)
            
            self.app.post(
                prefix + "/detect",
                summary="detect language",
            )(self.detect_language)

            self.app.post(
                prefix + "/tts",
                summary="text to speech",
            )(self.text_to_speech)


class ArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super(ArgParser, self).__init__(*args, **kwargs)

        self.add_argument(
            "-s",
            "--server",
            type=str,
            default="0.0.0.0",
            help="Server IP for HF LLM Chat API",
        )
        self.add_argument(
            "-p",
            "--port",
            type=int,
            default=23333,
            help="Server Port for HF LLM Chat API",
        )

        self.add_argument(
            "-d",
            "--dev",
            default=False,
            action="store_true",
            help="Run in dev mode",
        )

        self.args = self.parse_args(sys.argv[1:])


app = ChatAPIApp().app

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/transcribe")
async def whisper_transcribe(
    audio_file: UploadFile = File(description="Audio file for transcribe"),
    language: str = Form(),
    model: str = Form(),
):
    MODEL_MAP = {
        "whisper-small": "openai/whisper-small",
        "whisper-medium": "openai/whisper-medium",
        "whisper-large": "openai/whisper-large",   
        "default": "openai/whisper-small",
    }
    AUDIO_MAP = {
        "audio/wav": "audio/wav",
        "audio/mpeg": "audio/mpeg",
        "audio/x-flac": "audio/x-flac",   
    }
    item_response = {
            "statue": 200,
            "result": "",
            "start": 0,
            "end": 0
    }
    if audio_file.content_type in AUDIO_MAP.keys():
        if model in MODEL_MAP.keys():
            target_model = model
        else:
            target_model = "default"

        real_name = MODEL_MAP[target_model]
        device = 0 if torch.cuda.is_available() else "cpu"
        pipe = pipeline(
           task="automatic-speech-recognition",
           model=real_name,
           chunk_length_s=30,
           device=device,
        )
        time_start = time.time()
        pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(language=language, task="transcribe")
        try:
           suffix = Path(audio_file.filename).suffix
           with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(audio_file.file, tmp)
            tmp_path = Path(tmp.name)
        finally:
           audio_file.file.close()
        #file_data = await audio_file.read()
        # rv = data.encode('utf-8')
        #rv = base64.b64encode(file_data).decode()
        #print(rv, "rvrvrvrvr")
        audio_data = np.fromfile(tmp_path)    
        text = pipe(audio_data)["text"]
        time_end = time.time()
        item_response["status"] = 200
        item_response["result"] = text
        item_response["start"] = time_start
        item_response["end"] = time_end
    else:
        item_response["status"] = 400
        item_response["result"] = 'Acceptable files: audio/wav,audio/mpeg,audio/x-flac'
        
    
    return item_response
    
if __name__ == "__main__":
    args = ArgParser().args
    if args.dev:
        uvicorn.run("__main__:app", host=args.server, port=args.port, reload=True)
    else:
        uvicorn.run("__main__:app", host=args.server, port=args.port, reload=False)

    # python -m apis.chat_api      # [Docker] on product mode
    # python -m apis.chat_api -d   # [Dev]    on develop mode
