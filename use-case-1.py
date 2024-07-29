import boto3
import PyPDF2
import sagemaker
from sagemaker import get_execution_role
import transformers
from transformers import BertTokenizer, BertForQuestionAnswering
import torch
from io import BytesIO

# Descarga el modelo pre-entrenado BERT
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

# Procesa el texto del PDF y genera respuestas
def generate_response(text):
    # https://huggingface.co/transformers/v4.8.2/internal/tokenization_utils.html#transformers.tokenization_utils_base.PreTrainedTokenizerBase.encode_plus
    inputs = tokenizer.encode_plus(text, return_tensors='pt', add_special_tokens=True, max_length=512, padding=True, truncation=True)
 
    # optionally:
    # https://huggingface.co/transformers/v4.8.2/main_classes/tokenizer.html#transformers.PreTrainedTokenizer.encode
    # inputs = tokenizer.encode(text, padding=True, truncation=True, max_length=512, add_special_tokens=True,  return_tensors='pt')

    start_scores, end_scores = model(**inputs)
    start_idx = torch.argmax(start_scores)
    end_idx = torch.argmax(end_scores)
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][start_idx:end_idx+1]))
    return answer

# Configura la sesion de libreria boto (para AWS por entorno)
# session = boto3.Session(
#    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
#    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
# )
# Configura el cliente de S3
s3 = boto3.client('s3')
bucket_name = 'demo-exercises'

# Descarga el archivo PDF
file_key = 'caso-de-uso-1-IA/Favaron_Pedro_2012_these.pdf'
response = s3.get_object(Bucket=bucket_name, Key=file_key)
pdf_bytes = response['Body'].read()

# Crea un objeto lector de PDF
pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))

# Itera a través de las páginas y extrae el texto
pdf_text = ""
for page in range(len(pdf_reader.pages)):
    page_obj = pdf_reader.pages[page]
    pdf_text += "\n" + page_obj.extract_text()
    
# Cierra el archivo PDF
pdf_reader.stream.close()

# Genera una respuesta
response_text = generate_response(pdf_text)
print(f'Respuesta generada: {response_text}')
