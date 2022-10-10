from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List
import torch
import re

class SentimentModel():
    def __init__(self, model_name: str = "oliverguhr/german-sentiment-bert"):
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'        
            
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.clean_chars = re.compile(r'[^A-Za-züöäÖÜÄß ]', re.MULTILINE)
        self.clean_http_urls = re.compile(r'https*\S+', re.MULTILINE)
        self.clean_at_mentions = re.compile(r'@\S+', re.MULTILINE)

    def predict_sentiment(self, texts: List[str], output_probabilities = False)-> List[str]:
        texts = [self.clean_text(text) for text in texts]
        # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
        # truncation=True limits number of tokens to model's limitations (512)
        encoded = self.tokenizer.batch_encode_plus(texts,padding=True, add_special_tokens=True,truncation=True, return_tensors="pt")
        encoded = encoded.to(self.device)
        with torch.no_grad():
                logits = self.model(**encoded)
        
        label_ids = torch.argmax(logits[0], axis=1)

        if output_probabilities == False:
            return [self.model.config.id2label[label_id.item()] for label_id in label_ids]
        else:
            predictions = torch.softmax(logits[0], dim=-1).tolist()  
            probabilities = []
            for prediction in predictions:
                probabilities += [[[self.model.config.id2label[index], item] for index, item in enumerate(prediction)]]
                
            return [self.model.config.id2label[label_id.item()] for label_id in label_ids], probabilities

    def replace_numbers(self,text: str) -> str:
            return text.replace("0"," null").replace("1"," eins").replace("2"," zwei")\
                .replace("3"," drei").replace("4"," vier").replace("5"," fünf") \
                .replace("6"," sechs").replace("7"," sieben").replace("8"," acht") \
                .replace("9"," neun")         

    def clean_text(self,text: str)-> str:    
            text = text.replace("\n", " ")        
            text = self.clean_http_urls.sub('',text)
            text = self.clean_at_mentions.sub('',text)        
            text = self.replace_numbers(text)                
            text = self.clean_chars.sub('', text) # use only text chars                          
            text = ' '.join(text.split()) # substitute multiple whitespace with single whitespace   
            text = text.strip().lower()
            return text
