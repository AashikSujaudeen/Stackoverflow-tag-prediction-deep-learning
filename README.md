# StackOverflow-tag-prediction-deep-learning

## Problem Statement
Stack Overflow is a largest, most trusted online community for program developers to post the issues, get answers for the issues, learn, share their
programming knowledge, and build their careers. It is something which every programmer uses one way,
or another.

It is very important and critical for Stack overflow to predict the accurate tags for the question posted
based on the title and description of the question posted as the posted question will be routed to
appropriate experts or set of people based on the tags to get the appropriate answer with quick
turnaround time. Any wrong tagging would drastically reduce the time taken to get the answer to the
questions. The wrongly tagged question can also be left unanswered at times. It can also suggest other
questions like the posted questions as helpful Q&A threads to bring good customer experience which is
again based on the predicted tags. Also, many times, users add wrong tags to their posted question
which also need to be taken care when predicting the accurate tag. Any inefficient tagging would cause a
mess and bring down the user traffic to the website. Therefore, efficient tag prediction is business
critical for Stack Overflow website.

## Machine Learning Problem Interpretation
This is a problem that involves training a machine learning model using the labelled data and predicting the tags. An input record can have multiple tags as the title and question posted by the user can be related to multiple technologies. For an instance, a question is asked on python based tensorflow related error, it can be classified as 'tensorflow', 'deep learning', 'python'. So it is essential to train the model to learn and predict/classify all possible tags for each input record. Therefore, it is a supervised multilabel classification problem. For data wrangling steps, sklearn preprocessing, keras text preprocessing, Natural Language Tool Kit (NLTK), beautiful re soup some of the key libraries used.  

## Proposed Architecture
![Architecture](https://github.com/AashikSujaudeen/Stackoverflow-tag-prediction-deep-learning/blob/master/Architecture.png)

## Algorithm Evaluation & Selection
I tried several supervised classification machine learning algorithms on oneVsRest classifier but got less accuracy as the dataset is super imbalanced and there is no hard truth or hard false in the predicteded labels on many instances; also, the learning rate was very less. At the end, I opted for deep learning approach with LSTM (Long SHort Term Memory) so that my model can learn and remember efficiently for better prediction. Technically, I used tensorflow keras sequential LSTM in the deep learning model.

## Installation Steps
1. Clone the repository using git clone https://github.com/AashikSujaudeen/Stackoverflow-tag-prediction-deep-learning.git
2. Virtual Environment Set up: Create a virtual environment and activate the same using below commands:
   conda create --name stackoverflow python=3.8
3. Check local system pre-requisites CUDNN and CUDA for tensorflow 2.4 installation. Also please ensure the compatible CUDNN & CUDA are installed based on your machine type and tensorflow 2.2 above compatibility.
4. PYTHON Package & API Server - Installations: Install the required python packages using command: "pip install -r requirements.txt". It will also install the flask server required for the API set up.
5. Frontend Server & Packages - Installation & Set up: Install the node js server and set up by running the below commands
    a) Server:
      sudo yum install -y gcc-c++ make
      curl -sL https://rpm.nodesource.com/setup_14.x | sudo -E bash -
      sudo yum install -y nodejs
    b) Set up React js:
      npx create-react-app my-app
      cd my-app
    c) Install required react js libraries by running the below command:
      npm install axios react-bootstrap-validation react-range-step-input semantic-ui-css react-loader-spinner react-promise-tracker react-progress-button --save
6. Copy the content from my-app/src of the cloned directory and replace the content of 'my-app/src' of the recently set up React js 'my-app/src' on your machine

## Steps to Run (on local machine)
1. Download the Stack-overflow dataset 'Train.csv' from the link https://www.kaggle.com/c/facebook-recruiting-iii-keyword-extraction/data?select=Train.zip and place the extracted Train.csv file under 'datasets' directory location.
2. In one terminal on virtualenv location, run the command "python preprocess.py"
3. In the same terminal after completion of preprocessing, train the model using the command: "python train.py"
4. Post the model training, it is time for running the model prediction by using command "python app.py"
5. In another terminal in parallel, you can start the frontend server by navigating to my-app folder (cd my-app)  and running the command "npm start"
6. Now a web browser session will be opened on the address http://localhost:3000 where the Stack-overflow tag predictor application running.
6 .Happy prediction!

## Steps to Run (on already deployed cloud version):
1. Create a session in Putty to connect with AWS instance.
2. Navigate to stackoverflowapp folder using command: cd stackoverflowapp
3. Activate the virtual environment using command: conda activate production-deployment-so-dl
4. Run the app.py python file using command: python3 app.py
5. Running app.py will internally start the flask server and the running api will be ready to get the request and post the response.
6. Create another session in Putty with the AWS instance and repeat step 2 and 3.
7. Navigate to frontend folder 'my-app' by using command: cd my-app
8. Start the node js server using command: npm start
9. After completing untiil step 8, both api and frontend servers will be up and running. Therefore, the application can be tested by connecting to the prediction page using the link http://instance-name:3000/ in a new browser session. The instance-name can be different each time the AWS instance is restarted (the instance name or aws address can be made static by opting Elastic IP Address on AWS console which is not the scope of this project).

## Metrics
### Offline Metrics:
Accuracy

### Online Metrics:
Average Accuracy, Response Time
Note: As the data is very imbalanced and there is no hard truth or hard false on prediction w.r.t. multilabel classification, the prediction is considered as correct if at least one tag is predicted correctly when compared with respective ground truth.

## Demo
https://user-images.githubusercontent.com/49852589/118376495-2f680480-b596-11eb-8cf2-8d4951fd77e4.mp4

## Monitoring, Model Evaluation & Model Training
The above mentioned online metrics Average accuracy and the Response time can be used to monitor the production deployed model for model eavaluation. Upon the model drift breaches the threshold %, the engineer can plan for the model retraining using the updated dataset and further improvements in the model.

## Future Enhancements & Conclusions:
The model can be further enhanced to make it as a stack-overflow search engine along with the already created tag predictor solution. So when the custom question is inputted, the model will predict the applicable tags and also list downs the related stack-overflow disucussion links related to the search question. Also different State of the Art (SOTA) models can be tried using transfer learning like GENSIM word2vec model to get high accuracy and efficient solution. 

