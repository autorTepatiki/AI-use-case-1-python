import boto3
import sagemaker
from sagemaker import get_execution_role
import transformers
from transformers import BertTokenizer, BertForQuestionAnswering
import torch

# Descarga el modelo pre-entrenado BERT
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

# Procesa el texto del PDF y genera respuestas
def generate_response(text):
    inputs = tokenizer.encode_plus(text, return_tensors='pt', add_special_tokens=True)
    start_scores, end_scores = model(**inputs)
    start_idx = torch.argmax(start_scores)
    end_idx = torch.argmax(end_scores)
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][start_idx:end_idx+1]))
    return answer

# Configura el cliente de S3
s3 = boto3.client('s3')
bucket_name = 'demo-exercises'

# Descarga el archivo PDF
file_key = '/caso-de-uso-1-IA/Favaron_Pedro_2012_these.pdf'
response = s3.get_object(Bucket=bucket_name, Key=file_key)
pdf_text = response['Body'].read().decode('utf-8')

# Genera una respuesta
response_text = generate_response(pdf_text)
print(f'Respuesta generada: {response_text}')
