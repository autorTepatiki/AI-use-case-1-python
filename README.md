README

PRE-REQUISITES 
+ Install the PYTHON language from scratch, in the given operating system you have. Refer to https://www.python.org/downloads/ 
+ Don´t frget to install the Python Package Manager, "pip". E.g. if under Ubuntu: "sudo apt-get install pip"

PRE-REQUISITES FOR THE PYTHON LANGUAGE
+ run install.sh, "pip install -r requirements.txt" 
+ Alternativelly, per library:
    * pip install boto3
    * pip install sagemaker
    * pip install transformers

![installing](./images/installng.png)

PRE-REQUISITES FOR RUNNING LOCALLY
+ PyTorch > 2.0 installed , visit : https://pytorch.org/get-started/locally/
    * pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu121

PRE-TRAINED MODEL 'BERT'
+ https://huggingface.co/transformers/v4.8.2/model_doc/bert.html?highlight=berttokenizer

WARNING MESSAGES

+ A parameter name that contains `beta` will be renamed internally to `bias`. Please use a different name to suppress this warning.
+ Some weights of BertForQuestionAnswering were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['qa_outputs.bias', 'qa_outputs.weight']
+ You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.

EXCEPTION MESSAGES

+ The size of tensor a (XXX) must match the size of tensor b (YYY) at non-singleton dimension 1
    * This is because, BERT uses word-piece tokenization. If any of the word is not in the vocabulary, it splits the word to it's word pieces. Example: if "playing" is not in the vocabulary, it can split down to "play", "##ing". This increases the amount of tokens in a given sentence after tokenization. You can specify certain parameters to get fixed length tokenization:
    * tokenized_sentence = tokenizer.encode(test_sentence, padding=True, truncation=True,max_length=50, add_special_tokens = True)
    * https://huggingface.co/transformers/v4.8.2/model_doc/bert.html?highlight=berttokenizer
    * https://huggingface.co/transformers/v4.8.2/main_classes/tokenizer.html#transformers.PreTrainedTokenizer.encode
