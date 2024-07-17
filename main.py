# 必要なライブラリのインポート
import re  
import random 
import time 
from statistics import mode  
from PIL import Image  
import numpy as np  
import pandas 
import torch  
import torch.nn as nn  
import torchvision  
from torchvision import transforms
import os
import shutil
from concurrent.futures import ThreadPoolExecutor  
from google.colab import drive
drive.mount('/content/drive')


# # 並列ファイルコピー機能(少し不安)
# # コピー元とコピー先のディレクトリを指定
# src_dir = '/content/drive/MyDrive/DL_base/DL_base_Last/data'
# dst_dir = '/content/data'

# # コピー先ディレクトリを作成
# os.makedirs(dst_dir, exist_ok=True)

# # コピーするファイルのリストを作成
# files_to_copy = []
# for root, _, files in os.walk(src_dir):
#     for file in files:
#         src_file = os.path.join(root, file)
#         dst_file = os.path.join(dst_dir, os.path.relpath(src_file, src_dir))
#         files_to_copy.append((src_file, dst_file))

# # 並列でファイルをコピーする関数
# def copy_file(src_dst):
#     src_file, dst_file = src_dst
#     dst_file_dir = os.path.dirname(dst_file)
#     os.makedirs(dst_file_dir, exist_ok=True)
#     try:
#         shutil.copy2(src_file, dst_file)
#     except Exception as e:
#         print(f"Error copying {src_file} to {dst_file}: {e}")

# # ThreadPoolExecutorを使って並列でコピー
# with ThreadPoolExecutor(max_workers=24) as executor:
#     executor.map(copy_file, files_to_copy)

# # ファイルがすべてコピーされたかを確認
# copied_files = [os.path.join(root, file) for root, _, files in os.walk(dst_dir) for file in files]
# missing_files = [src for src, dst in files_to_copy if dst not in copied_files]

# if missing_files:
#     print(f"Missing files: {missing_files}")
# else:
#     print("All files copied successfully.")


def set_seed(seed):

    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  

# テキストの前処理
def process_text(text):
    """
    入力テキストを前処理する

    Args:
        text (str): 前処理する入力テキスト

    Returns:
        str: 前処理されたテキスト
    """
    # すべての文字を小文字に変換
    text = text.lower()

    # 数詞を数字に変換するための辞書
    num_word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10'
    }
    # 数詞を数字に置換
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)

    # 単独のピリオド（小数点でないもの）を削除
    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)

    # 冠詞（a, an, the）を削除
    text = re.sub(r'\b(a|an|the)\b', '', text)

    # 一般的な短縮形にアポストロフィを追加
    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)

    # 句読点をスペースに変換（アポストロフィとコロンは除く）
    text = re.sub(r"[^\w\s':]", ' ', text)

    # カンマの前のスペースを削除
    text = re.sub(r'\s+,', ',', text)

    # 連続するスペースを1つに変換し、前後の空白を削除
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# 1. データローダーの作成
class VQADataset(torch.utils.data.Dataset):
    def __init__(self, json_path, img_directory, img_transform=None, include_answers=True):
        self.img_transform = img_transform 
        self.img_directory = img_directory  
        self.data_frame = pandas.read_json(json_path) 
        self.include_answers = include_answers  

        # 質問と回答のインデックス辞書を初期化
        self.word_to_index = {}
        self.answer_to_index = {}
        self.index_to_word = {}
        self.index_to_answer = {}

        for query in self.data_frame["question"]:
            cleaned_query = process_text(query) 
            query_words = cleaned_query.split(" ")
            for word in query_words:
                if word not in self.word_to_index:
                    # 新しい単語にインデックスを割り当て
                    self.word_to_index[word] = len(self.word_to_index)

        
        self.data_frame["cleaned_question"] = self.data_frame["question"].apply(process_text)

        self.index_to_word = {idx: word for word, idx in self.word_to_index.items()}

        if self.include_answers:
            # 回答に含まれる単語を辞書に追加（回答がある場合のみ）
            for answer_list in self.data_frame["answers"]:
                for answer_dict in answer_list:
                    answer_text = answer_dict["answer"]
                    cleaned_answer = process_text(answer_text)  # テキストの前処理
                    if cleaned_answer not in self.answer_to_index:
                        self.answer_to_index[cleaned_answer] = len(self.answer_to_index)

            
            self.index_to_answer = {idx: answer for answer, idx in self.answer_to_index.items()}

    def update_dict(self, train_dataset):
        """
        検証用データ，テストデータの辞書を訓練データの辞書に更新する．
        Parameters
        ----------
        train_dataset : Dataset
            訓練データのDataset
        """
        # 訓練データセットの辞書で、現在のデータセットの辞書を更新
        self.word_to_index = train_dataset.word_to_index
        self.answer_to_index = train_dataset.answer_to_index
        self.index_to_word = train_dataset.index_to_word
        self.index_to_answer = train_dataset.index_to_answer

    def __getitem__(self, idx):

        # 画像の読み込みと前処理
        img_path = f"{self.img_directory}/{self.data_frame['image'][idx]}"
        img = Image.open(img_path)
        if self.img_transform:
            img = self.img_transform(img)

        # 質問文のインデックス化
        query_words = self.data_frame["cleaned_question"][idx].split()
        query_indices = [self.word_to_index.get(word, len(self.word_to_index)) for word in query_words]

        max_query_length = 100
        if len(query_indices) > max_query_length:
            query_indices = query_indices[:max_query_length]
        else:
            query_indices += [0] * (max_query_length - len(query_indices))

        query_tensor = torch.LongTensor(query_indices)

        if self.include_answers:
            # 回答の処理（回答がある場合のみ）
            answer_indices = [self.answer_to_index[process_text(ans["answer"])] for ans in self.data_frame["answers"][idx]]
            most_common_answer_idx = mode(answer_indices)  
            return img, query_tensor, torch.Tensor(answer_indices), int(most_common_answer_idx)
        else:
            return img, query_tensor

    def __len__(self):
        # データセットの長さ（サンプル数）を返す
        return len(self.data_frame)
    

# 2. 評価指標の実装
# 簡単にするならBCEを利用する
def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):

    total_acc = 0.  # バッチ全体の累積精度

    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.  # 現在の質問に対する精度

        # 各回答に対してループ
        for i in range(len(answers)):
            num_match = 0

            # 他の全ての回答と比較
            for j in range(len(answers)):
                if i == j:
                    continue
                if pred == answers[j]:
                    num_match += 1

            acc += min(num_match / 3, 1)

        # 質問ごとの平均精度を累積精度に追加
        total_acc += acc / 10  #回答者数に従う(10人って書いてた気がする)

    return total_acc / len(batch_pred)

# 3. モデルの実装
# ResNetを利用できるようにしておく

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):

        super().__init__()

        # 1つ目の畳み込み層とバッチ正規化
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 2つ目の畳み込み層とバッチ正規化
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # スキップ接続（ショートカット）の定義
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(residual)  # スキップ接続の追加したお
        out = self.relu(out)

        return out


class EnhancedBottleneckBlock(nn.Module):
    expansion_factor = 4  # チャンネル拡大係数

    def __init__(self, input_channels: int, bottleneck_channels: int, stride: int = 1):
        super().__init__()

        # 次元圧縮層: 1x1畳み込みで特徴マップを圧縮
        self.dimension_reducer = nn.Conv2d(input_channels, bottleneck_channels, kernel_size=1, stride=1)
        self.reducer_normalizer = nn.BatchNorm2d(bottleneck_channels)

        # 特徴抽出層: 3x3畳み込みで空間的特徴を抽出
        self.feature_extractor = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, padding=1)
        self.extractor_normalizer = nn.BatchNorm2d(bottleneck_channels)

        # 次元復元層: 1x1畳み込みで特徴マップを元の次元に戻す
        self.dimension_restorer = nn.Conv2d(bottleneck_channels, bottleneck_channels * self.expansion_factor, kernel_size=1, stride=1)
        self.restorer_normalizer = nn.BatchNorm2d(bottleneck_channels * self.expansion_factor)

        # 活性化関数
        self.activation = nn.ReLU(inplace=True)

        # 恒等写像パス（スキップ接続）
        self.identity_path = nn.Sequential()
        if stride != 1 or input_channels != bottleneck_channels * self.expansion_factor:
            self.identity_path = nn.Sequential(
                nn.Conv2d(input_channels, bottleneck_channels * self.expansion_factor, kernel_size=1, stride=stride),
                nn.BatchNorm2d(bottleneck_channels * self.expansion_factor)
            )

    def forward(self, input_tensor):
        identity = input_tensor

        # 次元圧縮
        compressed = self.activation(self.reducer_normalizer(self.dimension_reducer(input_tensor)))
        
        # 特徴抽出
        features = self.activation(self.extractor_normalizer(self.feature_extractor(compressed)))
        
        # 次元復元
        restored = self.restorer_normalizer(self.dimension_restorer(features))

        # 恒等写像との結合
        output = restored + self.identity_path(identity)
        
        # 最終活性化
        return self.activation(output)

class ResNet(nn.Module):
    def __init__(self, block, layers):

        super().__init__()
        self.in_channels = 64

        # 入力層
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNetの主要な層
        self.layer1 = self._make_layer(block, layers[0], 64)
        self.layer2 = self._make_layer(block, layers[1], 128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], 256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], 512, stride=2)

        # 出力層
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 512)

    def _make_layer(self, block, blocks, out_channels, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet50():
    return ResNet(EnhancedBottleneckBlock, [3, 4, 6, 3])


class VQAModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_answers: int):
        super().__init__()
        
        self.image_encoder = self._create_image_encoder()
        self.text_encoder = self._create_text_encoder(vocab_size, embed_dim)
        self.fusion_network = self._create_fusion_network(num_answers)

    def _create_image_encoder(self):
        return ResNet18()

    def _create_text_encoder(self, vocab_size: int, embed_dim: int):
        return nn.Sequential(
            nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0),  # +1 for unknown words
            nn.Linear(embed_dim, 512),
            nn.ReLU(inplace=True)
        )

    def _create_fusion_network(self, num_answers: int):
        return nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_answers)
        )

    def forward(self, image, question):
        image_features = self.encode_image(image)
        question_features = self.encode_question(question)
        
        combined_features = torch.cat([image_features, question_features], dim=1)
        return self.fusion_network(combined_features)

    def encode_image(self, image):
        return self.image_encoder(image)

    def encode_question(self, question):
        embedded = self.text_encoder[0](question)
        
        # Create a mask to ignore padding tokens
        mask = (question != 0).float().unsqueeze(-1)
        
        # Compute mean of non-padding embeddings
        question_feature = (embedded * mask).sum(dim=1) / mask.sum(dim=1)
        
        return self.text_encoder[1:](question_feature)


# 4. 学習の実装

def train(model, dataloader, optimizer, criterion, device):
    total_loss = 0
    total_acc = 0
    simple_acc = 0
    start = time.time()
    for image, question, answers, mode_answer in dataloader:
    
        image, question, answer, mode_answer = \
            image.to(device), question.to(device), answers.to(device), mode_answer.to(device)

        pred = model(image, question)  
        loss = criterion(pred, mode_answer.squeeze())

        optimizer.zero_grad()  
        loss.backward() 
        optimizer.step()  # パラメータの更新

        total_loss += loss.item()  
        total_acc += VQA_criterion(pred.argmax(1), answers) 
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item() 

    # 平均損失、精度、訓練時間を計算して返す
    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start

def eval(model, dataloader, optimizer, criterion, device):
    total_loss = 0
    total_acc = 0
    simple_acc = 0
    start = time.time()
    with torch.no_grad():  # 勾配計算を無効化
        for image, question, answers, mode_answer in dataloader:
            # データをデバイスに移動
            image, question, answer, mode_answer = \
                image.to(device), question.to(device), answers.to(device), mode_answer.to(device)

            pred = model(image, question)  # モデルによる予測
            loss = criterion(pred, mode_answer.squeeze())  # 損失の計算

            total_loss += loss.item()  # 累積損失の更新
            total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
            simple_acc += (pred.argmax(1) == mode_answer).mean().item()  # simple accuracy

    # 平均損失、精度、評価時間を計算して返す
    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start

def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 画像の前処理
    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # 訓練データセットとテストデータセットの作成
    train_dataset = VQADataset(
        json_path="./data/train.json",
        img_directory="./data/train",
        img_transform=img_transform
    )
    
    test_dataset = VQADataset(
        json_path="./data/valid.json",
        img_directory="./data/valid",
        img_transform=img_transform,
        include_answers=False
    )
    
    test_dataset.update_dict(train_dataset)

    # DataLoaderの作成
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # モデルの初期化
    model = VQAModel(
        vocab_size=len(train_dataset.word_to_index),
        embed_dim=300,
        num_answers=len(train_dataset.answer_to_index)
    ).to(device)

    num_epochs = 20
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # 訓練ループ
    for epoch in range(num_epochs):
        model.train()  # モデルを訓練モードに設定
        # モデルの訓練
        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device)
        
        print(f"【{epoch + 1}/{num_epochs}】")
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Simple Acc: {train_simple_acc:.4f}, Time: {train_time:.2f}s")

    # テストデータに対する予測
    model.eval()
    submission = []
    with torch.no_grad():
        for image, question in test_loader:
            image, question = image.to(device), question.to(device)
            pred = model(image, question)
            pred = pred.argmax(1).cpu().item()
            submission.append(pred)

    # 予測結果をインデックスから実際の回答に変換
    submission = [train_dataset.index_to_answer[id] for id in submission]
    submission = np.array(submission)

    # モデルと予測結果の保存
    torch.save(model.state_dict(), "/content/drive/MyDrive/DL_base/model_0717_1.pth")
    np.save("/content/drive/MyDrive/DL_base/submission_0717_1.npy", submission)

if __name__ == "__main__":
    main()
