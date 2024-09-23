import tensorflow as tf
from transformers import RobertaConfig, TFRobertaModel
import tokenizers


class SentimentModel:
    def __init__(self, model_weights_path, config_path, vocab_path, merges_path, MAX_LEN=96):
        # Initialisiere Tokenizer
        self.tokenizer = tokenizers.ByteLevelBPETokenizer(
            vocab=vocab_path,
            merges=merges_path,
            lowercase=True,
            add_prefix_space=True
        )
        self.MAX_LEN = MAX_LEN

        # Laden des RoBERTa-Modells
        config = RobertaConfig.from_pretrained(config_path)
        self.model = self.build_model(model_weights_path, config)

    def build_model(self, model_weights_path, config):
        # Modellarchitektur aufbauen
        ids = tf.keras.layers.Input((self.MAX_LEN,), dtype=tf.int32)
        att = tf.keras.layers.Input((self.MAX_LEN,), dtype=tf.int32)
        tok = tf.keras.layers.Input((self.MAX_LEN,), dtype=tf.int32)

        roberta_model = TFRobertaModel.from_pretrained(model_weights_path, config=config)
        x = roberta_model(ids, attention_mask=att, token_type_ids=tok)

        x1 = tf.keras.layers.Conv1D(1, 1)(x[0])
        x1 = tf.keras.layers.Flatten()(x1)
        x1 = tf.keras.layers.Activation('softmax')(x1)

        x2 = tf.keras.layers.Conv1D(1, 1)(x[0])
        x2 = tf.keras.layers.Flatten()(x2)
        x2 = tf.keras.layers.Activation('softmax')(x2)

        model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1, x2])
        model.load_weights(model_weights_path)
        return model

    def predict(self, text, sentiment):
        # Text vorbereiten
        ids, att, tok = self.tokenize_text(text, sentiment)

        # Modellvorhersage
        preds = self.model.predict([ids, att, tok])
        start_idx = tf.argmax(preds[0], axis=1).numpy()[0]
        end_idx = tf.argmax(preds[1], axis=1).numpy()[0]

        # Vorhersage in Text umwandeln
        return self.decode_prediction(text, start_idx, end_idx)

    def tokenize_text(self, text, sentiment):
        sentiment_ids = {'positive': 1313, 'negative': 2430, 'neutral': 7974}
        text = " " + " ".join(text.split())
        enc = self.tokenizer.encode(text)
        s_tok = sentiment_ids[sentiment]

        ids = [0] + enc.ids + [2, 2] + [s_tok] + [2]
        att = [1] * len(ids)

        # Padding auf MAX_LEN
        padding_length = self.MAX_LEN - len(ids)
        ids += [1] * padding_length
        att += [0] * padding_length
        tok = [0] * self.MAX_LEN

        return tf.constant([ids]), tf.constant([att]), tf.constant([tok])

    def decode_prediction(self, text, start_idx, end_idx):
        # Dekodiere die vorhergesagten Token IDs in Text
        text = " " + " ".join(text.split())
        enc = self.tokenizer.encode(text)
        return self.tokenizer.decode(enc.ids[start_idx - 1:end_idx])