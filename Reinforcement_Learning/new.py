import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from collections import Counter
import nltk
from nltk.translate.bleu_score import corpus_bleu
from transformers import ViTFeatureExtractor, ViTModel
from tqdm import tqdm

nltk.download('punkt')

class Flickr8kDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_captions = {}
        self.imgs = []
        self.all_captions = []
        
        with open(captions_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                image_name = parts[0].split('#')[0]
                caption = parts[1]
                if image_name not in self.img_captions:
                    self.img_captions[image_name] = []
                self.img_captions[image_name].append(caption)
                
        for img, caps in self.img_captions.items():
            self.imgs.append(img)
            self.all_captions.append(caps)
        
        self.vocab = self.build_vocab()
        self.vocab_size = len(self.vocab)
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.imgs[idx])
        image = Image.open(img_name).convert('RGB')
        captions = self.all_captions[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, captions
    
    def build_vocab(self, threshold=4):
        counter = Counter()
        for captions in self.all_captions:
            for caption in captions:
                counter.update(nltk.word_tokenize(caption.lower()))
        
        words = [word for word, count in counter.items() if count >= threshold]
        vocab = {'<PAD>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3}
        vocab.update({word: idx + 4 for idx, word in enumerate(words)})
        return vocab

class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        
    def forward(self, images):
        images = ((images * 0.5 + 0.5) * 255).type(torch.uint8)
        
        images = [Image.fromarray(img.permute(1, 2, 0).cpu().numpy()) for img in images]
        
        features = self.feature_extractor(images, return_tensors="pt")
        features = self.vit(**features.to(next(self.vit.parameters()).device)).last_hidden_state
        return features.mean(dim=1)  # Global average pooling

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs

class CaptionRL(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CaptionRL, self).__init__()
        self.encoder = EncoderCNN()
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)
        
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def sample(self, images, max_len=20):
        features = self.encoder(images)
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(max_len):
            hiddens, _ = self.decoder.lstm(inputs)
            outputs = self.decoder.linear(hiddens.squeeze(1))
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted)
            inputs = self.decoder.embed(predicted)
            inputs = inputs.unsqueeze(1)
            if predicted.item() == self.decoder.embed.num_embeddings - 1:  # <END> token
                break
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids

def compute_reward(generated_captions, reference_captions, vocab):
    inv_vocab = {v: k for k, v in vocab.items()}
    generated_words = [[inv_vocab[idx.item()] for idx in caption if idx.item() not in [vocab['<START>'], vocab['<END>'], vocab['<PAD>']]] 
                       for caption in generated_captions]
    reference_words = [[nltk.word_tokenize(ref.lower()) for ref in refs] for refs in reference_captions]
    return corpus_bleu(reference_words, generated_words)

def train_rl(model, train_loader, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        total_reward = 0
        for images, captions in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            sampled_captions = model.sample(images)
            reward = compute_reward(sampled_captions, captions, train_loader.dataset.vocab)
            total_reward += reward
            
            loss = -torch.tensor(reward, device=device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        avg_reward = total_reward / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Reward: {avg_reward:.4f}')
        
        evaluate(model, train_loader, device)

def evaluate(model, data_loader, device):
    model.eval()
    total_bleu = 0
    with torch.no_grad():
        for images, captions in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device)
            generated_captions = model.sample(images)
            bleu = compute_reward(generated_captions, captions, data_loader.dataset.vocab)
            total_bleu += bleu
    
    avg_bleu = total_bleu / len(data_loader)
    print(f'Evaluation BLEU Score: {avg_bleu:.4f}')

def main():
    embed_size = 256
    hidden_size = 512
    num_layers = 1
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    
    dataset = Flickr8kDataset(root_dir='flickr8k/Flicker8k_Dataset', 
                              captions_file='flickr8k/Flickr8k.token.txt', 
                              transform=transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    vocab_size = len(dataset.vocab)
    model = CaptionRL(embed_size, hidden_size, vocab_size, num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print("Starting reinforcement learning training...")
    train_rl(model, train_loader, optimizer, num_epochs, device)

if __name__ == '__main__':
    main()