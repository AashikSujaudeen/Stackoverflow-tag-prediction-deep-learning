import os
import pickle
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from normalize import Normalize
import logging

logging.basicConfig(level=logging.DEBUG, filename='./logs/logfile_prediction.txt')

logging.debug('')
logging.debug('Model Prediction - Phase')
logging.debug('------------------------')
logging.debug('')

class CustomModelPrediction(object):
    def __init__(self, model, tokenizer, tag_encoder,  y_test ):
        self._model = model
        self._tokenizer = tokenizer
        self._tag_encoder = tag_encoder
        #self._X_test = X_test
        self._y_test = y_test
        self._accuracy = 0

    def predict(self, test_input, custom_input, test_case_count):
        normalize = Normalize()
        if custom_input:
            input_count = 0
            #prediction_df = pd.DataFrame(columns = ["Que No","Questions", "Predicted_Tags"])
            prediction_list = []
            for question in test_input:
                input_count += 1
                question_ = normalize.normalize(question)
                logging.debug('-' * (len(question) + 16))
                logging.debug('Test Case No.{}: {}'.format(input_count, str(question)))
                predicted_tag = self.tag_predictor(question_)
                logging.debug('Predicted Tags: {}'.format(predicted_tag))
                prediction_list.append({'que_no':input_count,'questions':str(question),'predicted_tags':predicted_tag})

                #logging.debug('')
            logging.debug('')
            return prediction_list

        else:
            test_idx = np.random.randint(len(test_input), size=test_case_count)
            logging.debug("Predicted Vs Ground Truth for {} sample(s)".format(test_case_count))
            logging.debug('-' * 50)
            logging.debug('')
            input_count = 0
            input_predicted_list = []
            prediction_score = 0
            predicted_tag_list = []
            prediction_list = []
            #pd.DataFrame(columns = ["Que No", "Questions", "Ground_Truth","Predicted_Tags"])
            for idx in test_idx:
                input_count += 1
                test_case = idx
                question = str(test_input[test_case])
                logging.debug('')
                logging.debug('-' * 100)
                logging.debug('Test Case No.{}:'.format(input_count))
                logging.debug("Question ID: {}".format(test_case))
                logging.debug('Question: {}'.format(question))
                predicted_tag = self.tag_predictor(normalize.normalize_(question))
                predicted_tag_list.append(predicted_tag)
                ground_truth = self._tag_encoder.inverse_transform(np.array([self._y_test[test_case]]))
                score = 0
                ground_truth_ = [*ground_truth[0]]
                #predicted_tag_ = [*predicted_tag]

                for tag in predicted_tag:
                    tags =  [*tag]
                    for tag in tags:
                        if tag in ground_truth_ :
                            if(len(tag) >0):
                                score = 1
                                prediction_score += 1
                            break
                        else:
                            for gt_tag in ground_truth_:
                                if (gt_tag.startswith(tag) or tag.startswith(gt_tag)) and len(gt_tag) > 0 :
                                    score = 1
                                    prediction_score += 1
                                    break

                prediction_current = {'que_no':input_count, 'questions':question , 'ground_truth': str(ground_truth), 'predicted_tags': str(predicted_tag)}
                prediction_list.append(prediction_current)

                # append row to the dataframe
                input_predicted_list.append([input_count, ground_truth, predicted_tag,score])

                # log the ground truth & prediction
                logging.debug('Predicted: ' + str(predicted_tag))
                logging.debug('Ground Truth: ' + str(ground_truth))
                logging.debug('\n')



            accuracy = prediction_score / input_count
            self._accuracy = accuracy
            return prediction_list


    def tag_predictor(self, text):
        # Tokenize text
        X_test = pad_sequences(self._tokenizer.texts_to_sequences([text]), maxlen=500)
        # Predict
        prediction = self._model.predict([X_test])[0]
        for i, value in enumerate(prediction):
            if value > 0.5:
                prediction[i] = 1
            else:
                prediction[i] = 0
        tags = self._tag_encoder.inverse_transform(np.array([prediction]))
        return tags

    def evaluate(self, X_test, y_test, batch_size, MAX_SEQUENCE_LENGTH=400):
        score =  self._model.evaluate(X_test, y_test, batch_size=batch_size)
        evaluation_loss = score[0]
        evaluation_accuracy = score[1]
        return evaluation_loss, evaluation_accuracy

    @classmethod
    def from_path(cls):
        import keras
        model = load_model('models/tag_predictor_keras_model.h5')

        with open('models/y_test.pickle', 'rb') as handle:
            y_test = pickle.load(handle)
            y_test = y_test.astype(float)

        with open('models/tag_encoder.pickle', 'rb') as handle:
            tag_encoder = pickle.load(handle)

        with open('models/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

        logging.debug(' ')
        return cls(model, tokenizer, tag_encoder,  y_test)




def findPredictions(custom_input, question_count, question):
    model = CustomModelPrediction.from_path()
    evaluate_flag = False
    if not custom_input:
        with open('models/X_test_raw.pickle', 'rb') as handle:
            X_test = pickle.load(handle)

        predictions = model.predict(X_test, custom_input, question_count)
        accuracy = model._accuracy
        logging.debug("Accuracy score: {:.2f}%".format(accuracy*100))
        if evaluate_flag:
            with open('models/X_test_padded.pickle', 'rb') as handle:
                X_test_padded = pickle.load(handle)

            with open('models/y_test.pickle', 'rb') as handle:
                y_test = pickle.load(handle)
                y_test = y_test.astype(float)

            batch_size = 128
            evaluation_loss, evaluation_accuracy = model.evaluate(X_test_padded, y_test, batch_size)
            logging.debug("")
            logging.debug("Evaluation Loss: {:.4f}%, Evaluation Accuracy {:.2f}%".format(evaluation_loss * 100,
                                                                                         evaluation_accuracy * 100))

        return predictions, accuracy

    else:
        X_test = [question]
        predictions = model.predict(X_test, custom_input, len(X_test))
        return predictions




origins = [
    "http://localhost:3000",
    "localhost:3000"
]

#For FastAPI
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from UserInputs import UserInput
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"Stack Overflow - Tag Predictions"}

@app.post("/predict")
async def get_prediction(data:UserInput):
    data = data.dict()
    customInput = data['customInputFlag']
    if(customInput == 'true'):
        customInputFlag_ = True
    else:
        customInputFlag_ = False
    #customInputFlag = data['customInputFlag']
    customInputFlag = customInputFlag_
    questions = data['questions']
    questionCount = data['questionCount']
    print("React Inputs: ",customInput,questions,questionCount)
    if customInputFlag:
        predictions = findPredictions(customInputFlag, len(questions), questions)
        return {'predictions': predictions}
    else:
        predictions, accuracy = findPredictions(customInputFlag, questionCount,'')
        accuracy = float(round((accuracy*100),2))
        return {'predictions': predictions,'accuracy': accuracy}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
"""

# For Flask API

from flask import Flask, request, json, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/")
@cross_origin()
def homepage():
    return jsonify({'test': "Working!"})

@app.route('/predict')
@cross_origin()
def get_prediction():
    params = request.json
    if (params == None):
        params = request.args
    customInput = params['customInputFlag']
    if(customInput == 'true'):
        customInputFlag_ = True
    else:
        customInputFlag_ = False
    customInputFlag = customInputFlag_
    questions = params['questions']
    questionCount = int(params['questionCount'])
    logging.debug("React Inputs: ",customInput,questions,questionCount,type(customInput),type(questions),type(questionCount))
    if customInputFlag:
        predictions = findPredictions(customInputFlag, len(questions), questions)
        return jsonify({'predictions': predictions})
    else:
        predictions, accuracy = findPredictions(customInputFlag, questionCount,'')
        accuracy = float(round((accuracy),4))
        return jsonify({'predictions': predictions, 'accuracy': accuracy})


if __name__ == '__main__':
    # For running in local
    #app.run(debug=True)
    
    # For running on cloud
    app.run(host='0.0.0.0', port=5000)

