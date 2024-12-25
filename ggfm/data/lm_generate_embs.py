import torch
import numpy as np


def generate_lm_embs(all_name, tokenizer, model, indxs, device):

    def get_word_embeddings(word, device):
        encoded_word = tokenizer.encode(word, add_special_tokens=False)
        tokens_tensor = torch.tensor([encoded_word]).to(device)
        with torch.no_grad():
            output = model(tokens_tensor)
            embeddings = output[0][0].mean(dim=0)
        return embeddings.cpu().numpy()

    emb = get_word_embeddings("hello", device)
    emb = np.zeros((len(indxs), len(emb)))

    for i in range(len(all_name)):
        word = all_name[i]
        emb[i] = get_word_embeddings(word, device)

    return emb