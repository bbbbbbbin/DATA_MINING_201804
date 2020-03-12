import json

input=open('./data/train_data_complete.json','r',encoding='utf-8')
output = open('./data/train_data_complete_done.txt','w')

frame = json.load(input)
# print(frame)
for i in range(len(frame)):
	ques = frame[i]['question']
	i_id = frame[i]['item_id']
	for j in range(len(frame[i]['passages'])):
		cont = frame[i]['passages'][j]['content']
		p_id = frame[i]['passages'][j]['passage_id']
		label = frame[i]['passages'][j]['label']
		line = ques + '\t' + cont + '\t' + str(label) + '\t' + str(p_id) + '\n'
		output.write(line)

output.close()
