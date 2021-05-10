import React, {useState, useEffect} from 'react';
/*import logo from './logo.svg'; */
import './App.css';
import axios from 'axios';
import logo_white_bg from './images/logo_white_bg.png';
import {Card, Badge} from 'react-bootstrap';
import {RangeStepInput} from 'react-range-step-input';
import moment from 'moment'
import {forceNumber} from './utils';
/*import Slider from 'react-rangeslider';

import 'bootstrap/dist/css/bootstrap.min.css' */

function App() {
    const [submitted, setSubmitted] = useState(false);
    const [question, setQuestion] = useState('');
    const [customInputFlag, setCustomInputFlag] = useState('false');
    const [custom, setCustom] = useState(2);
    const [customFlag, setCustomFlag] = useState(true);
    const [questionCount, setQuestionCount] = useState(0);
    const [count, setCount] = useState(0);
    const [avgAccuracy, setAvgAccuracy] = useState(0.00);
    const [findPredictions, setFindPredictions] = useState([]);
    const [questionCountTemp, setQuestionCountTemp] = useState([]);
    const [questionTemp, setQuestionTemp] = useState([]);
    const [customInputFlagTemp, setCustomInputFlagTemp] = useState([]);
      /*const [groundTruths, setGroundTruths] = useState([]);*/
    const [accuracy, setAccuracy] = useState(0.00);
    const [totalResponseTime, setTotalResponseTime] = useState("");



   const fetchAll = React.useCallback(async() => {
   if(customInputFlag === 'false'){
   setCustomFlag(false)
   }
   if(customInputFlag === 'true'){
   setCustomFlag(true)
   }

    if((customInputFlag === 'true' && question.includes('?')) || (customInputFlag === 'false' && questionCount > 0)){
      setSubmitted(true)
      try{
         var requestTime = moment(new Date().toLocaleString(), 'DD-MM-YYYY hh:mm:ss');
        /*axios.get('http://ec2-54-162-14-176.compute-1.amazonaws.com:5000/predict', { */
         axios.get('http://127.0.0.1:5000/predict', {
          params: {
            "customInputFlag": customInputFlag,
            "questions": question,
            "questionCount": questionCount
          }

        })
        .then((response) => {
            var responseTime = moment(new Date().toLocaleString(), 'DD-MM-YYYY hh:mm:ss');
            if(customInputFlag === 'true' && question.includes('?')){
            setFindPredictions(response.data.predictions)
            setCustom(1)
            setCustomFlag(true)
            setSubmitted(false)
            }
            else if (customInputFlag === 'false' && questionCount > 0) {
            setFindPredictions(response.data.predictions)
            setAccuracy(response.data.accuracy*100)
            setCount(count + 1)
            if(count===0)
                {
                    setAvgAccuracy(response.data.accuracy*100)
                }
            else
                {
                    setAvgAccuracy((avgAccuracy+(response.data.accuracy*100))/(2))
                }
            setCustom(0)
            setCustomFlag(false)
            setSubmitted(false)
            var totalRespTime = responseTime.diff(requestTime, 'seconds');
            setTotalResponseTime(totalRespTime)

            }
        });
      }
      catch(e){
        console.log('Error: ' + e)
      }

    }

    else{
      setTimeout(() => {
        setSubmitted(false)
        setFindPredictions([])
        setAccuracy(0)
      }, 700);

    }
    }, [customInputFlag, question, questionCount])

    useEffect(async() => {
    await fetchAll()
    }, [fetchAll])

    const handleSubmit = (e) => {

        e.preventDefault();
        fetchAll()
        console.log(`Form submitted, ${questionCount+''+customInputFlag+''+question}`);

    }


  var Tags = findPredictions.map((predictions,i) => {
    return(
       <Badge className="tags">{custom? predictions.predicted_tags[0]:''}</Badge>
     );
  });
  const renderTags = () => {
     if(findPredictions.length > 0 && custom === 1){
        return(
          <div className="inputfieldcontainer">
            <h1 className="predicted_tags_h1">Predicted Tags</h1>
             {Tags}
            </div>
        );
     }
  }


  var renderAccuracy = <p></p>
  var renderResults =  <p></p>
  if(custom === 0){
  if(count===1){
  var renderAccuracy = <h3>
  <span style={{color: '#4d4d4d', fontFamily: 'Axiforma Bold'}}>Prediction Accuracy: </span>
  <span style={{color: '#FF8319', fontFamily: 'Axiforma Bold'}}>{Number(accuracy).toFixed(2)}%&nbsp;&nbsp;</span>
  <span style={{color: '#4d4d4d', fontFamily: 'Axiforma Bold'}}>|&nbsp;&nbsp;&nbsp;Average Accuracy: </span>
  <span style={{color: '#FF8319', fontFamily: 'Axiforma Bold'}}>{Number(avgAccuracy).toFixed(2)}%&nbsp;&nbsp;</span>
  <span style={{color: '#4d4d4d', fontFamily: 'Axiforma Bold'}}>for&nbsp;last&nbsp;</span>
  <span style={{color: '#FF8319', fontFamily: 'Axiforma Bold'}}>{count}&nbsp;</span>
  <span style={{color: '#4d4d4d', fontFamily: 'Axiforma Bold'}}>prediction execution </span>
  <span style={{color: '#4d4d4d', fontFamily: 'Axiforma Bold'}}>|&nbsp;&nbsp;Response Time: </span>
  <span style={{color: '#FF8319', fontFamily: 'Axiforma Bold'}}>{totalResponseTime}&nbsp;</span>
  <span style={{color: '#4d4d4d', fontFamily: 'Axiforma Bold'}}>seconds </span></h3>}
  else {
  var renderAccuracy = <h3>
  <span style={{color: '#4d4d4d', fontFamily: 'Axiforma Bold'}}>Prediction Accuracy: </span>
  <span style={{color: '#FF8319', fontFamily: 'Axiforma Bold'}}>{Number(accuracy).toFixed(2)}%&nbsp;&nbsp;</span>
  <span style={{color: '#4d4d4d', fontFamily: 'Axiforma Bold'}}>|&nbsp;&nbsp;&nbsp;Average Accuracy: </span>
  <span style={{color: '#FF8319', fontFamily: 'Axiforma Bold'}}>{Number(avgAccuracy).toFixed(2)}%&nbsp;&nbsp;</span>
  <span style={{color: '#4d4d4d', fontFamily: 'Axiforma Bold'}}>for&nbsp;last&nbsp;</span>
  <span style={{color: '#FF8319', fontFamily: 'Axiforma Bold'}}>{count}&nbsp;</span>
  <span style={{color: '#4d4d4d', fontFamily: 'Axiforma Bold'}}>prediction executions </span>
  <span style={{color: '#4d4d4d', fontFamily: 'Axiforma Bold'}}>|&nbsp;&nbsp;Response Time: </span>
  <span style={{color: '#FF8319', fontFamily: 'Axiforma Bold'}}>{totalResponseTime}&nbsp;</span>
  <span style={{color: '#4d4d4d', fontFamily: 'Axiforma Bold'}}>seconds </span></h3>
  }
  var renderResults = findPredictions.map((predictions,i) => {
     return(

        <a target="_blank" className="predictlink">

         <div className="findprediction">
            <div className="row">

                <div className="col-12 col-md-10">

                    <br />
                    <p><span style={{color: '#4d4d4d', fontFamily: 'Axiforma Bold'}}>Question {predictions.que_no}: </span></p>
                    <p><span style={{color: '#707070', fontFamily: 'Axiforma Bold'}}>{predictions.questions}</span></p>

                    <p><span style={{color: '#4d4d4d', fontFamily: 'Axiforma Bold'}}>Ground Truth: </span><span style={{color: '#FF8319', fontFamily: 'Axiforma Bold'}}>{predictions.ground_truth}</span></p>
                    <p><span style={{color: '#4d4d4d', fontFamily: 'Axiforma Bold'}}>Predicted Tags: </span><span style={{color: '#FF8319', fontFamily: 'Axiforma Bold'}}>{predictions.predicted_tags}</span></p>
                </div>
            </div>
         </div>
        </a>
     );
  });
   }
   else if(custom === 1 && customInputFlag=='true'){
  var renderAccuracy = <h3>
  <span style={{color: '#4d4d4d', fontFamily: 'Axiforma Bold'}}>Response Time: </span>
  <span style={{color: '#FF8319', fontFamily: 'Axiforma Bold'}}>{totalResponseTime}&nbsp;</span>
  <span style={{color: '#4d4d4d', fontFamily: 'Axiforma Bold'}}>seconds </span></h3>

  var renderResults = findPredictions.map((predictions,i) => {
     return(

        <a target="_blank" className="predictlink">

         <div className="findprediction">
            <div className="row">

                <div className="col-12 col-md-10">

                    <br />
                    <p><span style={{color: '#4d4d4d', fontFamily: 'Axiforma Bold'}}>Input Question: </span></p>
                    <p><span style={{color: '#707070', fontFamily: 'Axiforma Bold'}}>{predictions.questions}</span></p>
                    <p><span style={{color: '#4d4d4d', fontFamily: 'Axiforma Bold'}}>Predicted Tags: </span><span style={{color: '#FF8319', fontFamily: 'Axiforma Bold'}}>{predictions.predicted_tags[0].map((tag,index) =>
            <li key={index}>{tag}</li>
        )}</span></p>
                </div>
            </div>
         </div>
        </a>
     );
  });
   }

  return (


    <div>

      <div className="container" >
      <Card className='maincard' style={{'margin-top':submitted?'5%':'5%'}}>
          <Card.Body>
            <img className="logo_white_bg" src={logo_white_bg}/>
        <br />

      <div class="form-group has-feedback" className="inputfieldcontainer" align="center">


          <div class="form-control inputfield1" >
            <h1 className="num_results_h2" style={{color: '#464646'}}>Choose the question input type: </h1>
            <h3  style={{color:'#464646'}}><input type="radio" value="false" id="stack"
              onChange={e => setCustomInputFlag(e.target.value)} name="customInputFlag" defaultChecked />
            <label for="stack">&nbsp;Stack Overflow Dataset- Questions &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</label>
            <input type="radio" value="true" id="custom"
              onChange={e => setCustomInputFlag(e.target.value, setQuestion(''))} name="customInputFlag" />
            <label for="custom">&nbsp;Custom Questions</label>

            </h3>

            <br />
            {customFlag
            ?<div><input type="text" class="form-control inputfield2" style = {{width: 380, height:30, borderRadius: 7, borderColor:'orange'}} onChange={(e) => setQuestion(e.target.value)} value={question} placeholder="Type your custom query ending with '?' to predict its tags!" /></div>
            : <div className="col-9">
              <h2 className="num_results_h1" style={{color:'#464646'}}>No. of Stack-Overflow questions to retrieve:&nbsp;&nbsp;&nbsp;

                <RangeStepInput min={0} max={2500} onChange={(e) => setQuestionCount(e.target.value)} value={questionCount} step={100} />&nbsp;&nbsp;
                        &nbsp;     <input type="integer" class="form-control inputfield3" disabled = {true} style = {{borderRadius: 7, borderColor:'orange'}} onInput={(e) => setQuestionCount(e.target.value)} value={questionCount}/> </h2>



             </div>
            }

            <br/>


           </div>

          <i class="fas fa-search search_icon"></i>

        <div className="row" style={{marginTop: '10px'}}>
        <div className="col-md-6">
          </div>
          <div className="col-12 col-md-6">
            <div className="row">

              <div>


                </div>

            </div>
          </div>
        </div>
        </div>

        <br /><br />
        <div align="center">
        {renderTags}

        {renderAccuracy}
        </div>
        <br /><br />
        {renderResults}
          </Card.Body>
      </Card>
      <br />
      <br />
      </div>
    </div>

  );
}

export default App;
