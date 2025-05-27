import pandas as pd, numpy as np
import gzip, json
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import torch
from transformers import AutoTokenizer, AutoModel




####### Data
def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

def Load_data(df, user, item, rating, text): # insert variable's name of Data coresponding each variable of function
    df = df[[user, item, rating, text]] # extract user ID, item ID, rating, reviewtext
    df.rename(columns = {user: "user",
                        item: "item",
                        rating: "rating",
                        text: "text"},
             inplace = True)

    df = df.dropna()
    le = LabelEncoder()
    df["user"] = le.fit_transform(df["user"].values)
    df["item"] = le.fit_transform(df["item"].values)

    USER_LEN = df["user"].max() + 1 # number of users
    ITEM_LEN = df["item"].max() + 1 # number of items
    return df, USER_LEN, ITEM_LEN


def train_validation_test(dataset):
    np.random.seed(0)

    # 학습 데이터셋과 테스트 데이터셋 초기화
    train_dataset = pd.DataFrame(columns=['user', 'item', 'rating', 'text'])
    test_dataset = pd.DataFrame(columns=['user', 'item', 'rating', 'text'])

    # 각 유저에 대해 아이템을 8:2로 분할
    for user in tqdm(dataset['user'].unique()):
        user_data = dataset[dataset['user'] == user]
        if len(user_data) > 1:
            train, test = train_test_split(user_data, test_size=0.2, random_state=42)
            train_dataset = pd.concat([train_dataset, train])
            test_dataset = pd.concat([test_dataset, test])
        else:
            train_dataset = pd.concat([train_dataset, user_data])
    test_dataset = test_dataset[test_dataset['item'].isin(train_dataset['item'])]
    test_dataset = test_dataset[test_dataset['user'].isin(train_dataset['user'])]
    # 결과 출력
    print(f"Train dataset shape: {train_dataset.shape}")
    print(f"Test dataset shape: {test_dataset.shape}")

    print(set(test_dataset.user).issubset(set(train_dataset.user)))
    print(set(test_dataset.item).issubset(set(train_dataset.item)))

    return train_dataset, test_dataset


####### Transformer
def Tokenize(data, model_ckpt, batch_size): # function of extracting [CLS] Token embedding from BERT-based model

    """
    model_ckpt: verion of BERT or RoBERTa model
    col_name: append cls token embedding data column into dataframe
    batch_size: recommend that the value of this variable be 2 or 4
    """

    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = AutoModel.from_pretrained(model_ckpt).to(device)

    embeddings = []
    text_list = data['text'].tolist()

    for i in tqdm(range(0, len(text_list), batch_size)):
        batch_texts = text_list[i:i+batch_size]

        inputs = tokenizer(batch_texts, return_tensors='pt', truncation=True, padding=True) # default of max_length is 512
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings.append(embedding.last_hidden_state[:, 0, :])  # append CLS token embedding data

    # Stack embeddings into a tensor
    stacked_embeddings = torch.cat(embeddings, dim=0)

    stacked_embeddings = stacked_embeddings.cpu().numpy()

    result = stacked_embeddings.tolist()

    return result


def bert_roberta(train_df, test_df, batch_size = 1):
    global user_grouped, item_grouped
    user_grouped = train_df[["user", "text"]].groupby('user')["text"].apply(" ".join).reset_index()
    item_grouped = train_df[["item", "text"]].groupby('item')["text"].apply(" ".join).reset_index()

    user_grouped["user_bert"] = Tokenize(user_grouped, 'bert-base-uncased', batch_size=batch_size)
    item_grouped["item_bert"] = Tokenize(item_grouped, 'bert-base-uncased', batch_size=batch_size)

    user_grouped["user_roberta"] = Tokenize(user_grouped, 'roberta-base', batch_size=batch_size)
    item_grouped["item_roberta"] = Tokenize(item_grouped, 'roberta-base', batch_size=batch_size)

    def group_merge(user_df, item_df, df):
        bert_user = pd.merge(df, user_df[["user", "user_bert"]], how = "left", on  = "user")
        bert_user_item = pd.merge(bert_user, item_df[["item", "item_bert"]], how = "left", on  = "item")

        bert_roberta_user = pd.merge(bert_user_item, user_df[["user", "user_roberta"]], how = "left", on  = "user")
        final_df = pd.merge(bert_roberta_user, item_df[["item", "item_roberta"]], how = "left", on  = "item")

        return final_df

    train_dataset = group_merge(user_grouped, item_grouped, train_df)
    test_validation_dataset = group_merge(user_grouped, item_grouped, test_df)

    validation_dataset, test_dataset = train_test_split(test_validation_dataset, test_size=0.5, random_state=42)

    return train_dataset, validation_dataset, test_dataset
