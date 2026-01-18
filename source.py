import torch
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer,BertForQuestionAnswering
import faiss
import numpy as np
import os 
def chunk_text(text, chunk_size=2):
    sentences = text.split(". ")
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk = ". ".join(sentences[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

tokenizer=BertTokenizer.from_pretrained("deepset/bert-base-cased-squad2")
model=BertForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2")
fiass_model=SentenceTransformer("all-MiniLM-L6-v2")
if os.path.exists("context.index"):
    index=faiss.read_index("context.index")
    chucks=np.load("chucks.npy",allow_pickle=True).tolist()
else:
    index=None
    chucks=[]
choice = input("Type 'add' to add new context or 'ask' to ask questions: ").lower()
if choice=='add':
    context=input("Enter the context ?")
    new_chucks=chunk_text(context)
    new_embedding=fiass_model.encode(new_chucks).astype("float32")
    if index is None:
        index = faiss.IndexFlatL2(new_embedding.shape[1])
    index.add(np.array(new_embedding))
    chucks.extend(new_chucks)
    faiss.write_index(index,"context.index")
    np.save("chucks.npy",chucks)
elif choice=='ask':
    if index is None or len(chucks) == 0:
        print("No context found. Please add context first.")
        exit()
    while True:
        question=input("Answer any question regarding context ? to stop querying type 'exit'")
        if question.lower()=='exit':
            break
        question_embedding=fiass_model.encode(question)
        k=1
        distance,indices=index.search(question_embedding.astype('float32').reshape(1, -1),k)
        retrived_data=chucks[indices[0][0]]
        inputs=tokenizer(
        question,
        retrived_data,
        return_tensors="pt"
        )
        with torch.no_grad():
            output=model(**inputs)
        start_logits=output.start_logits
        end_logits=output.end_logits
        start_index=torch.argmax(start_logits)
        end_index=torch.argmax(end_logits)
        if end_index < start_index:
            print("Answer: I could not find an answer.")
            continue

        answer = tokenizer.decode(
            inputs["input_ids"][0][start_index:end_index + 1],
            skip_special_tokens=True
        )

        print("Answer:", answer)