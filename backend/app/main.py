from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from spellchecker import SpellChecker
from happytransformer import HappyTextToText, TTSettings
import nltk
import re
from difflib import SequenceMatcher
import itertools
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Download required NLTK data
nltk.download('punkt')

# Initialize models for grammar checking and correction 
happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
spell = SpellChecker()
args = TTSettings(num_beams=5, min_length=1)


# Data models
class TextInput(BaseModel):
    text: str

class Mistake(BaseModel):
    word: str
    type: str
    suggestions: list
    


def find_grammar_errors(text: str):
    input_text = f"grammar: {text}"
    corrected_text = happy_tt.generate_text(input_text, args=args).text
    
    original_words = nltk.word_tokenize(text)
    corrected_words = nltk.word_tokenize(corrected_text)
    
    matcher = SequenceMatcher(None, original_words, corrected_words)
    errors = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != 'equal':
            original_phrase = ' '.join(original_words[i1:i2])
            corrected_phrase = ' '.join(corrected_words[j1:j2])
            start_pos = sum(len(w) + 1 for w in original_words[:i1])  # Calculate start position
            end_pos = start_pos + len(' '.join(original_words[i1:i2]))  # Calculate end position

            errors.append({
                "problem": original_phrase,
                "suggestions": [corrected_phrase],
                "start_pos": start_pos,
                "end_pos": end_pos,
            })
    
    return errors


def find_spelling_errors(text: str):
    errors = []
    word_pattern = re.compile(r"\b\w+['\w]*\b")  # Improved regex
    for match in word_pattern.finditer(text):
        word = match.group()
        start_pos = match.start()
        end_pos = match.end()
        
        if word.lower() != spell.correction(word.lower()):
            errors.append({
                "type": "spelling",
                "word": word,
                "suggestions": list(spell.candidates(word))[:5],
                "start_pos": start_pos,
                "end_pos": end_pos,
            })
    return errors



#API for checking grammar and spelling errors
@app.post("/check/")
async def check_text(input: TextInput):
    text = input.text
    spelling_errors = find_spelling_errors(text)
    grammar_errors = find_grammar_errors(text)
    return {
        "text": text,  # Return original text for synchronization
        "spelling": spelling_errors,
        "grammar": grammar_errors
    }

#Parapharase Part
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load model and tokenizer for paraphrasing
tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")  
model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parapharase API
@app.post("/paraphrase/")
async def paraphrase_text(input: TextInput):
    text = input.text
    try:
        input_text = f"paraphrase: {text} </s>"
        encoding = tokenizer.encode_plus(
            input_text,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        outputs = model.generate(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            max_length=256,
            do_sample=True,
            top_k=200,
            top_p=0.95,
            early_stopping=True,
            num_return_sequences=3
        )
        
        paraphrases = []
        for output in outputs:
            line = tokenizer.decode(
                output, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            paraphrases.append(line)
            
        return {"paraphrases": paraphrases}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



