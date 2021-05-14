# Stackoverflow-tag-prediction-deep-learning

## Problem Statement
Stack Overflow is a largest, most trusted online community for program developers to learn, share their
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
This is a problem that involves training a machine learning model using the labelled data and predicting the tags. An input record can have multiple tags as the title and question posted by the user can be related to multiple technologies. For an instance, a question is asked on python based tensorflow related error, it can be classified as 'tensorflow', 'deep learning', 'python'. So it is essential to train the model to learn and predict/classify all possible tags for each input record. Therefore, it is a supervised multilabel classification problem. For data wrangling steps, sklearnpreprocessing, keras text preprocessing, Natural Language Tool Kit (NLTK), beautiful re soup some of the key libraries used.  

## Proposed Architecture
![Architecture](https://github.com/AashikSujaudeen/Stackoverflow-tag-prediction-deep-learning/blob/master/Architecture.png)

## Algorithm Evaluation & Selection
I tried several supervised classification machine learning algorithms on oneVsRest classifier but got less accuracy as the dataset is super imbalanced and there is no hard truth or hard false in the predicteded labels on many instances; also, the learning rate was very less. At the end, I opted for deep learning approach with LSTM (Long SHort Term Memory) so that my model can learn and remember efficiently for better prediction. Technically, I used tensorflow keras sequential LSTM in the deep learning model.

## Steps to Run:
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
