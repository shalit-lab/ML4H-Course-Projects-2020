import torch
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset


class NotesDataReader:
    def __init__(self, df, embeddings_version, max_len, text_feature):
        self.texts = list(df[text_feature].values)
        self.df = df
        if 'label' in df.columns:
            self.labels = torch.tensor(df['label'].values)
        else:
            self.labels = torch.tensor(df['Label'].values)
        self.hadm_id = torch.tensor(df['HADM_ID'].values)
        self.tokenizer = BertTokenizer.from_pretrained(embeddings_version)
        self.inputs_ids = self.create_inputs_ids(max_len)
        self.masks = self.create_masks()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return self.inputs_ids[index], self.masks[index], \
               self.labels[index], self.hadm_id[index]

    def create_inputs_ids(self, max_len):
        # do_lower_case = False
        input_ids = []
        for note in self.texts:
            encoded_note = self.tokenizer.encode(note, add_special_tokens=True,
                                                 max_length=max_len, truncation=True)
            input_ids.append(encoded_note)
        input_ids = pad_sequences(input_ids, maxlen=max_len, dtype="long",
                                  value=0, truncating="post", padding="post")
        return torch.tensor(input_ids)

    def create_masks(self):
        masks = []
        for encoded_note in self.inputs_ids:
            mask = [int(token_idx > 0) for token_idx in encoded_note]
            masks.append(mask)
        return torch.tensor(masks)

    # def create_tensor_dataset(self):
    #     return TensorDataset(self.inputs_ids, self.masks, self.labels, self.hadm_id)





