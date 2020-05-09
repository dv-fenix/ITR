import torch
from model import itr_Net
from transformers import BertTokenizer


#load pre-trained model
model = itr_Net.load_from_metrics(weights_path='/content/data/model1.ckpt', tags_csv='/content/itr/tb_logs/translation_model/version_0/meta_tags.csv').to('cpu')
#set to evaluation mode
model.eval()

#readtext from file
File_object = open("/content/data/hin-eng/test.txt", mode = 'r')
text = File_object.read()
File_object.close()
print(text)

src_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
tgt_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

tokenized_text = src_tokenizer.tokenize(text) #tokenize
indexed_tokens = src_tokenizer.convert_tokens_to_ids(tokenized_text)  #get indices
it_tensor = torch.tensor([indexed_tokens]) #convert to tensor

decoder_input = ""  #initialize string
tgt_tokenizer.eos_token = '</s>'
tgt_tokenizer.bos_token = '<s>'

tokenized_decoder_input = tgt_tokenizer.tokenize(decoder_input) #tokenize
indexed_decoder_tokens = tgt_tokenizer.convert_tokens_to_ids(tokenized_decoder_input)
indexed_decoder_tokens = [tgt_tokenizer.bos_token_id] + indexed_decoder_tokens + [tgt_tokenizer.mask_token] #bos and mask added

id_tensor = torch.tensor([indexed_decoder_tokens]) #convert to tensor

encoder = model.encoder
decoder = model.decoder

with torch.no_grad():
	h_s = encoder(it_tensor)[0] #last_hidden_state

mask_ids = [[1, 0]] #attention mask
mask_tensor = torch.tensor(mask_ids)

i = 1 #word_counter
ref = tgt_tokenizer.eos_token_id
while(indexed_decoder_tokens[-1] != ref):
	with torch.no_grad():
		predictions = decoder(input_ids = id_tensor, attention_mask = mask_tensor, encoder_hidden_state = h_s)
		predicted_index = torch.argmax(predictions[0][0][i])
		predicted_token = tgt_tokenizer.convert_ids_to_tokens([predicted_index])[0]
		print(predicted_token)
	if(predicted_index != ref):
		index_decoder_tokens = indexed_decoder_tokens[:-1] + [predicted_index] + [tgt_tokenizer.mask_token]
		id_tensor = torch.tensor([indexed_decoder_tokens]) #updated_tensor
		mask_ids[0] = mask_ids[0][:-1] + [1] + [0]
		mask_tensor = torch.tensor(mask_ids) #update attention tensor
	i = i+1 #update word count





