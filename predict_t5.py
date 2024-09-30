from transformers import AutoTokenizer, TFT5ForConditionalGeneration
import numpy as np
import tensorflow as tf
class SnapthatT5(TFT5ForConditionalGeneration):
    def __init__(self, *args, log_dir=None, cache_dir= None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_tracker= tf.keras.metrics.Mean(name='loss')

    @tf.function
    def train_step(self, data):
        x = data
        y = x["labels"]
        y = tf.reshape(y, [-1, 1])
        with tf.GradientTape() as tape:
            outputs = self(x, training=True)
            loss = outputs[0]
            logits = outputs[1]
            loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, self.trainable_variables)

        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        lr = 0.001# self.optimizer._decayed_lr(tf.float32)

        self.loss_tracker.update_state(loss)
        self.compiled_metrics.update_state(y, logits)
        metrics = {m.name: m.result() for m in self.metrics}
        metrics.update({'lr': lr})

        return metrics

    def test_step(self, data):
        x = data
        y = x["labels"]
        y = tf.reshape(y, [-1, 1])
        output = self(x, training=False)
        loss = output[0]
        loss = tf.reduce_mean(loss)
        logits = output[1]

        self.loss_tracker.update_state(loss)
        self.compiled_metrics.update_state(y, logits)
        return {m.name: m.result() for m in self.metrics}
tokenizer = AutoTokenizer.from_pretrained("t5-base")
#encoder_max_len = 200
#decoder_max_len = 54
#decoder_max_len = 250
data_dir = "./data2"
save_path = f"{data_dir}/experiments/t5/models"
model=SnapthatT5.from_pretrained(save_path)
context = """We went on a trip to Europe. We had our breakfast at 7 am in the morning at \
the nearby coffee shop. Wore a dark blue over coat for our first visit to Louvre Museum \
to experience history and art."""
def predict_default(context):
	input_text =  f"context: {context} </s>"
	encoded_query = tokenizer(input_text,
				 return_tensors='tf', pad_to_max_length=True, truncation=True, max_length=200)
	input_ids = encoded_query["input_ids"]
	attention_mask = encoded_query["attention_mask"]
	generated_answer = model.generate(input_ids, attention_mask=attention_mask, do_sample=True,
					     max_length=200,top_p=0.7)
	decoded_answer = tokenizer.decode(generated_answer.numpy()[0])
	print(decoded_answer)
def predict(context,encoder_max_len=200,decoder_max_len=200, top_p=0.7, top_k=40, repetition_penalty=1.0, temperature=0.6):
	input_text =  f"context: {context} </s>"
	encoded_query = tokenizer(input_text,
				 return_tensors='tf', pad_to_max_length=True, truncation=True, max_length=encoder_max_len)
	input_ids = encoded_query["input_ids"]
	attention_mask = encoded_query["attention_mask"]
	generated_answer = model.generate(input_ids, attention_mask=attention_mask, do_sample=True,
					     max_length=decoder_max_len, top_p=top_p , top_k=top_k, repetition_penalty=repetition_penalty, temperature=temperature)
	decoded_answer = tokenizer.decode(generated_answer.numpy()[0])
	print(decoded_answer)
