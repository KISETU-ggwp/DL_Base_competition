import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import torchtext
import pandas as pd
from PIL import Image
import numpy as np
import time
import re
import random
import csv
from sklearn.model_selection import KFold

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def process_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

class VQADataset(Dataset):
    def __init__(self, df_path, image_dir, transform=None, answer=True, max_length=12, class_mapping=None):
        self.df = pd.read_json(df_path)
        self.image_dir = image_dir
        self.transform = transform
        self.answer = answer
        self.max_length = max_length
        self.class_mapping = class_mapping

        self.glove = torchtext.vocab.GloVe(name='6B', dim=300)
        self.question_vocab = self._build_vocab(self.df['question'])

        if self.answer and self.class_mapping:
            self.idx2answer = {v: k for k, v in self.class_mapping.items()}
        elif self.answer:
            self.answer_vocab = self._build_vocab([ans['answer'] for answers in self.df['answers'] for ans in answers])
            self.idx2answer = {v: k for k, v in self.answer_vocab.items()}

    def _build_vocab(self, sentences):
        vocab = {'<pad>': 0, '<unk>': 1}
        for sentence in sentences:
            for word in process_text(sentence).split():
                if word not in vocab:
                    vocab[word] = len(vocab)
        return vocab

    def __getitem__(self, idx):
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        image = self.transform(image) if self.transform else image

        question = process_text(self.df['question'][idx])
        question_tokens = question.split()[:self.max_length]
        question_tokens += ['<pad>'] * (self.max_length - len(question_tokens))
        question_ids = [self.question_vocab.get(token, self.question_vocab['<unk>']) for token in question_tokens]
        question_tensor = torch.tensor(question_ids)

        if self.answer and 'answers' in self.df.columns:
            answers = [process_text(answer['answer']) for answer in self.df['answers'][idx]]
            if self.class_mapping:
                answer_ids = [self.class_mapping.get(ans, self.class_mapping['unanswerable']) for ans in answers]
            else:
                answer_ids = [self.answer_vocab.get(ans, self.answer_vocab['<unk>']) for ans in answers]
            mode_answer_id = max(set(answer_ids), key=answer_ids.count)
            return image, question_tensor, torch.tensor(answer_ids), torch.tensor(mode_answer_id)
        else:
            return image, question_tensor

    def __len__(self):
        return len(self.df)

class ImageFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg19 = models.vgg19(pretrained=True)
        self.features = nn.Sequential(*list(vgg19.features.children())[:-1])

    def forward(self, x):
        features = self.features(x)
        return features.view(x.size(0), 512, -1).permute(0, 2, 1)

class GRUEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.gru(embedded)
        return output

class CoAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear_v = nn.Linear(dim, dim)
        self.linear_q = nn.Linear(dim, dim)
        self.linear_hv = nn.Linear(dim, 1)
        self.linear_hq = nn.Linear(dim, 1)

    def forward(self, v, q):
        v = self.linear_v(v)
        q = self.linear_q(q)

        hv = torch.tanh(v.unsqueeze(1) + q.unsqueeze(2))
        hq = torch.tanh(v.unsqueeze(1) + q.unsqueeze(2))

        av = torch.softmax(self.linear_hv(hv).squeeze(-1), dim=2)
        aq = torch.softmax(self.linear_hq(hq).squeeze(-1), dim=1)

        v_att = torch.bmm(av, v)
        q_att = torch.bmm(aq.transpose(1, 2), q)

        return v_att, q_att

class ImageAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim * 2, dim)
        self.linear_att = nn.Linear(dim, 1)

    def forward(self, v, q):
        combined = torch.cat([v, q.unsqueeze(1).repeat(1, v.size(1), 1)], dim=2)
        features = torch.tanh(self.linear(combined))
        attention = torch.softmax(self.linear_att(features).squeeze(-1), dim=1)
        v_att = torch.bmm(attention.unsqueeze(1), v).squeeze(1)
        return v_att

class VQAModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes, num_gru_layers=1):
        super().__init__()
        self.image_extractor = ImageFeatureExtractor()
        self.question_encoder = GRUEncoder(vocab_size, embed_size, hidden_size, num_gru_layers)
        self.co_attention = CoAttention(hidden_size)
        self.image_attention = ImageAttention(hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, image, question):
        img_features = self.image_extractor(image)
        que_features = self.question_encoder(question)

        v_att, q_att = self.co_attention(img_features, que_features)
        v_att = self.image_attention(v_att, q_att[:, -1, :])

        combined = torch.cat([v_att, q_att[:, -1, :]], dim=1)
        output = self.classifier(combined)

        return output

def vqa_score(batch_pred: torch.Tensor, batch_answers: torch.Tensor) -> float:
    batch_size = batch_pred.size(0)
    pred = batch_pred.argmax(dim=1)

    scores = []
    for i in range(batch_size):
        answer_count = torch.bincount(batch_answers[i], minlength=batch_pred.size(1))
        num_match = answer_count[pred[i]].item()
        score = min(num_match / 3, 1)
        scores.append(score)

    return sum(scores) / batch_size

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_vqa_score = 0
    total_simple_acc = 0
    start = time.time()
    for batch in dataloader:
        image, question, answers, mode_answer = [item.to(device) for item in batch]

        optimizer.zero_grad()
        pred = model(image, question)
        loss = criterion(pred, mode_answer)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_vqa_score += vqa_score(pred, answers)
        total_simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()

    num_batches = len(dataloader)
    return (total_loss / num_batches,
            total_vqa_score / num_batches,
            total_simple_acc / num_batches,
            time.time() - start)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_vqa_score = 0
    total_simple_acc = 0
    start = time.time()
    with torch.no_grad():
        for batch in dataloader:
            image, question, answers, mode_answer = [item.to(device) for item in batch]

            pred = model(image, question)
            loss = criterion(pred, mode_answer)

            total_loss += loss.item()
            total_vqa_score += vqa_score(pred, answers)
            total_simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()

    num_batches = len(dataloader)
    return (total_loss / num_batches,
            total_vqa_score / num_batches,
            total_simple_acc / num_batches,
            time.time() - start)

def load_class_mapping(file_path):
    class_mapping = {}
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_mapping[row['answer']] = int(row['class_id'])
    return class_mapping

def main():
    config = {
        "seed": 42,
        "batch_size": 128,
        "num_epochs": 5,
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "embed_size": 300,
        "hidden_size": 512,
        "image_size": (224, 224),
        "max_question_length": 12,
        "num_gru_layers": 2,
        "data_path": {
            "train_json": "/content/data/train.json",
            "train_image": "/content/data/train",
            "valid_json": "/content/data/valid.json",
            "valid_image": "/content/data/valid"
        },
        "class_mapping_path": "/content/drive/MyDrive/DL_base/DL_base_Last/data/class_mapping.csv"
    }

    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(config["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class_mapping = load_class_mapping(config["class_mapping_path"])

    full_dataset = VQADataset(df_path=config["data_path"]["train_json"],
                              image_dir=config["data_path"]["train_image"],
                              transform=transform,
                              max_length=config["max_question_length"],
                              class_mapping=class_mapping)

    test_dataset = VQADataset(df_path=config["data_path"]["valid_json"],
                              image_dir=config["data_path"]["valid_image"],
                              transform=transform,
                              max_length=config["max_question_length"],
                              answer=False,
                              class_mapping=class_mapping)

    kfold = KFold(n_splits=5, shuffle=True, random_state=config["seed"])

    fold_results = []

    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset)):
        print(f"FOLD {fold}")
        print("--------------------------------")

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

        train_loader = DataLoader(full_dataset, batch_size=config["batch_size"], sampler=train_subsampler)
        val_loader = DataLoader(full_dataset, batch_size=config["batch_size"], sampler=val_subsampler)

        model = VQAModel(vocab_size=len(full_dataset.question_vocab),
                         embed_size=config["embed_size"],
                         hidden_size=config["hidden_size"],
                         num_classes=len(class_mapping),
                         num_gru_layers=config["num_gru_layers"]).to(device)

        glove_embeddings = torch.zeros(len(full_dataset.question_vocab), config["embed_size"])
        for word, idx in full_dataset.question_vocab.items():
            if word in full_dataset.glove.stoi:
                glove_embeddings[idx] = full_dataset.glove.vectors[full_dataset.glove.stoi[word]]

        model.question_encoder.embedding.weight.data.copy_(glove_embeddings)
        model.question_encoder.embedding.weight.requires_grad = False

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

        best_val_score = 0
        for epoch in range(config["num_epochs"]):
            train_loss, train_vqa_score, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device)
            val_loss, val_vqa_score, val_simple_acc, val_time = evaluate(model, val_loader, criterion, device)

            print(f"Epoch {epoch+1}/{config['num_epochs']}")
            print(f"Train - Loss: {train_loss:.4f}, VQA Score: {train_vqa_score:.4f}, Acc: {train_simple_acc:.4f}, Time: {train_time:.2f}s")
            print(f"Val   - Loss: {val_loss:.4f}, VQA Score: {val_vqa_score:.4f}, Acc: {val_simple_acc:.4f}, Time: {val_time:.2f}s")

            if val_vqa_score > best_val_score:
                best_val_score = val_vqa_score
                torch.save(model.state_dict(), f"/content/drive/MyDrive/DL_base/DL_base_Last/data/best_model_fold{fold}.pth")
                print("Best model saved!")
