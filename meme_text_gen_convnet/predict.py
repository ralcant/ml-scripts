from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import json
import random
import util
import os

def predict_meme_text(model_path, template_id, num_boxes, init_text = '', model_filename="model.h5", params_filename="params.json", beam_width=1, max_output_length=140):
    """
    Required: 
        - pretrained model
        - params.json: contains information like seq_length, mappings char_to_int. It should be automatically generated after running train.py
    """
    model = load_model(os.path.join(model_path, model_filename))
    params = json.load(open(os.path.join(model_path, params_filename)))
    SEQUENCE_LENGTH = params['sequence_length']
    char_to_int = params['char_to_int']
    labels = {v: k for k, v in params['labels_index'].items()}


    template_id = str(template_id).zfill(12)
    min_score = 0.1

    final_texts = [{'text': init_text, 'score': 1}]
    finished_texts = []
    for char_count in range(len(init_text), max_output_length):
        texts = []
        for i in range(0, len(final_texts)):
            box_index = str(final_texts[i]['text'].count('|'))
            texts.append(template_id + '  ' + box_index + '  ' + final_texts[i]['text'])
        sequences = util.texts_to_sequences(texts, char_to_int)
        data = pad_sequences(sequences, maxlen=SEQUENCE_LENGTH)
        predictions_list = model.predict(data)
        sorted_predictions = []
        for i in range(0, len(predictions_list)):
            for j in range(0, len(predictions_list[i])):
                sorted_predictions.append({
                    'text': final_texts[i]['text'],
                    'next_char': labels[j],
                    'score': predictions_list[i][j] * final_texts[i]['score']
                })

        sorted_predictions = sorted(sorted_predictions, key=lambda p: p['score'], reverse=True)
        top_predictions = []
        top_score = sorted_predictions[0]['score']
        rand_int = random.randint(int(min_score * 1000), 1000)
        for prediction in sorted_predictions:
            # give each prediction a chance of being chosen corresponding to its score
            if prediction['score'] >= rand_int / 1000 * top_score:
            # or swap above line with this one to enable equal probabilities instead
            # if prediction['score'] >= top_score * min_score:
                top_predictions.append(prediction)
        random.shuffle(top_predictions)
        final_texts = []
        for i in range(0, min(beam_width, len(top_predictions)) - len(finished_texts)):
            prediction = top_predictions[i]
            final_texts.append({
                'text': prediction['text'] + prediction['next_char'],
                # normalize all scores around top_score=1 so tiny floats don't disappear due to rounding
                'score': prediction['score'] / top_score
            })
            if prediction['next_char'] == '|' and prediction['text'].count('|') == num_boxes - 1:
                finished_texts.append(final_texts[len(final_texts) - 1])
                final_texts.pop()

        if char_count >= max_output_length - 1 or len(final_texts) == 0:
            final_texts = final_texts + finished_texts
            final_texts = sorted(final_texts, key=lambda p: p['score'], reverse=True)
            return final_texts[0]['text']


if __name__ == '__main__':
    print('predicting meme text for ID 61533...')
    print(predict_meme_text(".", 93895088, 4, '', "model_epoch_32.h5", "params_epoch_32.json"))
