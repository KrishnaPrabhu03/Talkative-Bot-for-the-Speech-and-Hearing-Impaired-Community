import {
  KNNImageClassifier
} from 'deeplearn-knn-image-classifier';
import * as dl from 'deeplearn';


const IMAGE_SIZE = 227;

const TOPK = 10;

const confidenceThreshold = 0.98


var words = ["start", "stop"];


class Main {
  constructor() {
    
    this.exampleCountDisplay = [];
    this.checkMarks = [];
    this.gestureCards = [];
    this.training = -1; 
    this.videoPlaying = false;
    this.previousPrediction = -1;
    this.currentPredictedWords = [];

    
    this.now;
    this.then = Date.now();
    this.startTime = this.then;
    this.fps = 5; 
    this.fpsInterval = 1000 / this.fps;
    this.elapsed = 0;

    
    this.knn = null;
    
    this.previousKnn = this.knn;

    
    this.welcomeContainer = document.getElementById("welcomeContainer");
    this.proceedBtn = document.getElementById("proceedButton");
    this.proceedBtn.style.display = "block";
    this.proceedBtn.classList.add("animated");
    this.proceedBtn.classList.add("flash");
    this.proceedBtn.addEventListener('click', () => {
      this.welcomeContainer.classList.add("slideOutUp");
    })

    this.stageTitle = document.getElementById("stage");
    this.stageInstruction = document.getElementById("steps");
    this.predButton = document.getElementById("predictButton");
    this.backToTrainButton = document.getElementById("backButton");
    this.nextButton = document.getElementById('nextButton');

    this.statusContainer = document.getElementById("status");
    this.statusText = document.getElementById("status-text");

    this.translationHolder = document.getElementById("translationHolder");
    this.translationText = document.getElementById("translationText");
    this.translatedCard = document.getElementById("translatedCard");

    this.initialTrainingHolder = document.getElementById('initialTrainingHolder');

    this.videoContainer = document.getElementById("videoHolder");
    this.video = document.getElementById("video");

    this.trainingContainer = document.getElementById("trainingHolder");
    this.addGestureTitle = document.getElementById("add-gesture");
    this.plusImage = document.getElementById("plus_sign");
    this.addWordForm = document.getElementById("add-word");
    this.newWordInput = document.getElementById("new-word");
    this.doneRetrain = document.getElementById("doneRetrain");
    this.trainingCommands = document.getElementById("trainingCommands");

    this.videoCallBtn = document.getElementById("videoCallBtn");
    this.videoCall = document.getElementById("videoCall");

    this.trainedCardsHolder = document.getElementById("trainedCardsHolder");

    
    this.initializeTranslator();

   
    this.predictionOutput = new PredictionOutput();
  }

  
  initializeTranslator() {
    this.startWebcam();
    this.initialTraining();
    this.loadKNN();
  }

  
  startWebcam() {
    navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'user'
        },
        audio: false
      })
      .then((stream) => {
        this.video.srcObject = stream;
        this.video.width = IMAGE_SIZE;
        this.video.height = IMAGE_SIZE;
        this.video.addEventListener('playing', () => this.videoPlaying = true);
        this.video.addEventListener('paused', () => this.videoPlaying = false);
      })
  }

  
  initialTraining() {
    
    this.nextButton.addEventListener('click', () => {
      const exampleCount = this.knn.getClassExampleCount();
      if (Math.max(...exampleCount) > 0) {
        
        if (exampleCount[0] == 0) {
          alert('You haven\'t added examples for the Start Gesture');
          return;
        }

        
        if (exampleCount[1] == 0) {
          alert('You haven\'t added examples for the Stop Gesture.\n\nCapture yourself in idle states e.g hands by your side, empty background etc.');
          return;
        }

        this.nextButton.style.display = "none";
        this.stageTitle.innerText = "Continue Training";
        this.stageInstruction.innerText = "Add Gesture Name and Train.";

        
        this.setupTrainingUI();
      }
    });

   
    this.initialGestures(0, "startButton");
    this.initialGestures(1, "stopButton");
  }

 
  loadKNN() {
    this.knn = new KNNImageClassifier(words.length, TOPK);

    
    this.knn.load().then(() => this.initializeTraining());
  }

 
  initialGestures(i, btnType) {

    var trainBtn = document.getElementById(btnType);

    
    trainBtn.addEventListener('click', () => {
      this.train(i);
    });

    var clearBtn = document.getElementById('clear_' + btnType);
    clearBtn.addEventListener('click', () => {
      this.knn.clearClass(i);
      this.exampleCountDisplay[i].innerText = " 0 examples";
      this.gestureCards[i].removeChild(this.gestureCards[i].childNodes[1]);
      this.checkMarks[i].src = "Images\\loader.gif";
    });

   
    var exampleCountDisplay = document.getElementById('counter_' + btnType);
    var checkMark = document.getElementById('checkmark_' + btnType);

    
    var gestureCard = document.createElement("div");
    gestureCard.className = "trained-gestures";

    var gestName = "";
    if (i == 0) {
      gestName = "Start";
    } else {
      gestName = "Stop";
    }
    var gestureName = document.createElement("h5");
    gestureName.innerText = gestName;
    gestureCard.appendChild(gestureName);
    this.trainedCardsHolder.appendChild(gestureCard);

    exampleCountDisplay.innerText = " 0 examples";
    checkMark.src = 'Images\\loader.gif';
    this.exampleCountDisplay.push(exampleCountDisplay);
    this.checkMarks.push(checkMark);
    this.gestureCards.push(gestureCard);
  }

 
  setupTrainingUI() {
    const exampleCount = this.knn.getClassExampleCount();
    
    if (Math.max(...exampleCount) > 0) {
      
      if (exampleCount[0] == 0) {
        alert('You haven\'t added examples for the wake word');
        return;
      }

      
      if (exampleCount[1] == 0) {
        alert('You haven\'t added examples for the Stop Gesture.\n\nCapture yourself in idle states e.g hands by your side, empty background etc.');
        return;
      }

      
      this.initialTrainingHolder.style.display = "none";

      
      this.trainingContainer.style.display = "block";
      this.trainedCardsHolder.style.display = "block";

      
      this.addWordForm.addEventListener('submit', (e) => {
        this.trainingCommands.innerHTML = "";

        e.preventDefault(); 
        var word = this.newWordInput.value.trim(); 

        
        if (word && !words.includes(word)) {
         
          words.push(word);

          
          this.createTrainingBtns(words.indexOf(word));
          this.newWordInput.value = '';

          
          this.knn.numClasses += 1;
          this.knn.classLogitsMatrices.push(null);
          this.knn.classExampleCount.push(0);

          
          this.initializeTraining();
          this.createTranslateBtn();
        } else {
          alert("Duplicate word or no word entered");
        }
        return;
      });
    } else {
      alert('You haven\'t added any examples yet.\n\nAdd a Gesture, then perform the sign in front of the webcam.');
    }
  }

  
  createTrainingBtns(i) { 
    
    var trainBtn = document.createElement('button');
    trainBtn.className = "trainBtn";
    trainBtn.innerText = "Train";
    this.trainingCommands.appendChild(trainBtn);

    var clearBtn = document.createElement('button');
    clearBtn.className = "clearButton";
    clearBtn.innerText = "Clear";
    this.trainingCommands.appendChild(clearBtn);

    
    trainBtn.addEventListener('mousedown', () => {
      this.train(i);
    });

    
    clearBtn.addEventListener('click', () => {
      this.knn.clearClass(i);
      this.exampleCountDisplay[i].innerText = " 0 examples";
      this.gestureCards[i].removeChild(this.gestureCards[i].childNodes[1]);
      this.checkMarks[i].src = 'Images\\loader.gif';
    });

    
    var exampleCountDisplay = document.createElement('h3');
    exampleCountDisplay.style.color = "black";
    this.trainingCommands.appendChild(exampleCountDisplay);

    var checkMark = document.createElement('img');
    checkMark.className = "checkMark";
    this.trainingCommands.appendChild(checkMark);

    
    var gestureCard = document.createElement("div");
    gestureCard.className = "trained-gestures";

    var gestName = words[i];
    var gestureName = document.createElement("h5");
    gestureName.innerText = gestName;
    gestureCard.appendChild(gestureName);
    this.trainedCardsHolder.appendChild(gestureCard);

    exampleCountDisplay.innerText = " 0 examples";
    checkMark.src = 'Images\\loader.gif';
    this.exampleCountDisplay.push(exampleCountDisplay);
    this.checkMarks.push(checkMark);
    this.gestureCards.push(gestureCard);

    
    gestureCard.addEventListener('click', () => { 
      
      if (gestureCard.style.marginTop == "17px" || gestureCard.style.marginTop == "") {
        this.addWordForm.style.display = "none";
        this.addGestureTitle.innerText = gestName;
        this.plusImage.src = "Images/retrain.svg";
        this.plusImage.classList.add("rotateIn");

        
        this.doneRetrain.style.display = "block";
        this.trainingCommands.innerHTML = "";
        this.trainingCommands.appendChild(trainBtn);
        this.trainingCommands.appendChild(clearBtn);
        this.trainingCommands.appendChild(exampleCountDisplay);
        this.trainingCommands.appendChild(checkMark);
        gestureCard.style.marginTop = "-10px";
      }
      
      else {
        this.addGestureTitle.innerText = "Add Gesture";
        this.addWordForm.style.display = "block";
        gestureCard.style.marginTop = "17px";

        this.trainingCommands.innerHTML = "";
        this.addWordForm.style.display = "block";
        this.doneRetrain.style.display = "none";
        this.plusImage.src = "Images/plus_sign.svg";
        this.plusImage.classList.add("rotateInLeft");
      }
    });

    
    this.doneRetrain.addEventListener('click', () => {
      this.addGestureTitle.innerText = "Add Gesture";
      this.addWordForm.style.display = "block";
      gestureCard.style.marginTop = "17px";

      this.trainingCommands.innerHTML = "";
      this.addWordForm.style.display = "block";
      this.plusImage.src = "Images/plus_sign.svg";
      this.plusImage.classList.add("rotateInLeft");
      this.doneRetrain.style.display = "none";
    });
  }

  
  initializeTraining() {
    if (this.timer) {
      this.stopTraining();
    }
    var promise = this.video.play();

    if (promise !== undefined) {
      promise.then(_ => {
        console.log("Autoplay started")
      }).catch(error => {
        console.log("Autoplay prevented")
      })
    }
  }

  
  train(gestureIndex) {
    console.log(this.videoPlaying);
    if (this.videoPlaying) {
      console.log("entered training");
      
      const image = dl.fromPixels(this.video);

      
      this.knn.addImage(image, gestureIndex);

      
      const exampleCount = this.knn.getClassExampleCount()[gestureIndex];

      if (exampleCount > 0) {
        
        this.exampleCountDisplay[gestureIndex].innerText = ' ' + exampleCount + ' examples';

        
        if (exampleCount == 1 && this.gestureCards[gestureIndex].childNodes[1] == null) {
          var gestureImg = document.createElement("canvas");
          gestureImg.className = "trained_image";
          gestureImg.getContext('2d').drawImage(video, 0, 0, 400, 180);
          this.gestureCards[gestureIndex].appendChild(gestureImg);
        }

         
        if (exampleCount == 30) {
          this.checkMarks[gestureIndex].src = "Images"
          this.checkMarks[gestureIndex].classList.add("animated");
          this.checkMarks[gestureIndex].classList.add("rotateIn");
        }
      }
    }
  }

  
  createTranslateBtn() {
    this.predButton.style.display = "block";
    this.createVideoCallBtn(); 
    this.createBackToTrainBtn(); 
    this.predButton.addEventListener('click', () => {
     
      console.log("go to translate");
      const exampleCount = this.knn.getClassExampleCount();
      if (Math.max(...exampleCount) > 0) {
        this.video.style.display = "inline-block"; 

        this.videoCall.style.display = "none"; 
        this.videoCallBtn.style.display = "block";

        this.backToTrainButton.style.display = "block";
        this.video.className = "videoPredict";
        this.videoContainer.style.display = "inline-block";
        this.videoContainer.style.width = "";
        this.videoContainer.style.height = "";
        this.videoContainer.className = "videoContainerPredict";
        this.videoContainer.style.border = "8px solid black";
        this.stageTitle.innerText = "Translate";
        this.stageInstruction.innerText = "Start Translating with your Start Gesture.";
        this.trainingContainer.style.display = "none";
        this.trainedCardsHolder.style.marginTop = "130px";
        this.translationHolder.style.display = "block";

        this.predButton.style.display = "none";
        this.setUpTranslation();
      } else {
        alert('You haven\'t added any examples yet.\n\nPress and hold on the "Add Example" button next to each word while performing the sign in front of the webcam.');
      }
    })
  }
  setUpTranslation() {
    if (this.timer) {
      this.stopTraining();
    }
    this.setStatusText("Status: Ready to Predict!", "predict");
    this.video.play();
    this.pred = requestAnimationFrame(this.predict.bind(this));
  }
  predict() {
    this.now = Date.now();
    this.elapsed = this.now - this.then;

    if (this.elapsed > this.fpsInterval) {
      this.then = this.now - this.elapsed % this.fpsInterval;
      if (this.videoPlaying) {
        const exampleCount = this.knn.getClassExampleCount();
        const image = dl.fromPixels(this.video);

        if (Math.max(...exampleCount) > 0) {
          this.knn.predictClass(image)
            .then((res) => {
              for (let i = 0; i < words.length; i++) {
                if (res.classIndex == i && res.confidences[i] > confidenceThreshold && res.classIndex != this.previousPrediction) { 
                  this.setStatusText("Status: Predicting!", "predict");
                  this.predictionOutput.textOutput(words[i], this.gestureCards[i], res.confidences[i] * 100);
                  this.previousPrediction = res.classIndex;
                }
              }
            }).then(() => image.dispose())
        } else {
          image.dispose();
        }
      }
    }
    this.pred = requestAnimationFrame(this.predict.bind(this));
  }
  pausePredicting() {
    console.log("pause predicting");
    this.setStatusText("Status: Paused Predicting", "predict");
    cancelAnimationFrame(this.pred);
    this.previousKnn = this.knn;
  }
  createBackToTrainBtn() {
    this.backToTrainButton.addEventListener('click', () => {
      main.pausePredicting();

      this.stageTitle.innerText = "Continue Training";
      this.stageInstruction.innerText = "Add Gesture Name and Train.";

      this.predButton.innerText = "Translate";
      this.predButton.style.display = "block";
      this.backToTrainButton.style.display = "none";
      this.statusContainer.style.display = "none";
      this.video.className = "videoTrain";
      this.videoContainer.className = "videoContainerTrain";
      this.videoCallBtn.style.display = "none";

      this.translationHolder.style.display = "none";
      this.statusContainer.style.display = "none";
      this.trainingContainer.style.display = "block";
      this.trainedCardsHolder.style.marginTop = "0px";
      this.trainedCardsHolder.style.display = "block";
    });
  }
  stopTraining() {
    this.video.pause();
    cancelAnimationFrame(this.timer);
    console.log("Knn for start: " + this.knn.getClassExampleCount()[0]);
    this.previousKnn = this.knn; 
  }
  createVideoCallBtn() {
    videoCallBtn.addEventListener('click', () => {
      this.stageTitle.innerText = "Video Call";
      this.stageInstruction.innerText = "Translate Gestures to talk to people on Video Call";

      this.video.style.display = "none";
      this.videoContainer.style.borderStyle = "none";
      this.videoContainer.style.overflow = "hidden";
      this.videoContainer.style.width = "630px";
      this.videoContainer.style.height = "355px";

      this.videoCall.style.display = "block";
      this.videoCallBtn.style.display = "none";
      this.backToTrainButton.style.display = "none";
      this.predButton.innerText = "Local Translation";
      this.predButton.style.display = "block";

      this.setStatusText("Status: Video Call Activated");
    })
  }
  setStatusText(status, type) { 
    this.statusContainer.style.display = "block";
    this.statusText.innerText = status;
    if (type == "copy") {
      console.log("copy");
      this.statusContainer.style.backgroundColor = "blue";
    } else {
      this.statusContainer.style.backgroundColor = "black";
    }
  }
}
class PredictionOutput {
  constructor() {
    this.synth = window.speechSynthesis;
    this.voices = [];
    this.pitch = 1.0;
    this.rate = 0.9;

    this.statusContainer = document.getElementById("status");
    this.statusText = document.getElementById("status-text");

    this.translationHolder = document.getElementById("translationHolder");
    this.translationText = document.getElementById("translationText");
    this.translatedCard = document.getElementById("translatedCard");
    this.trainedCardsHolder = document.getElementById("trainedCardsHolder");

    this.selectedVoice = 48; 

    this.currentPredictedWords = [];
    this.waitTimeForQuery = 10000;

    this.synth.onvoiceschanged = () => {
      this.populateVoiceList()
    };
    this.copyTranslation();
  }
  populateVoiceList() {
    if (typeof speechSynthesis === 'undefined') {
      console.log("no synth");
      return;
    }
    this.voices = this.synth.getVoices();

    if (this.voices.indexOf(this.selectedVoice) > 0) {
      console.log(this.voices[this.selectedVoice].name + ':' + this.voices[this.selectedVoice].lang);
    }
  }
  textOutput(word, gestureCard, gestureAccuracy) {
    if (word == 'start') {
      this.clearPara();

      setTimeout(() => {
        if (this.currentPredictedWords.length == 1) {
          this.clearPara();
        }
      }, this.waitTimeForQuery);
    }
    if (word != 'start' && this.currentPredictedWords.length == 0) {
      return;
    }
    if (this.currentPredictedWords.includes(word)) {
      return;
    }
    this.currentPredictedWords.push(word);
    if (word == "start") {
      this.translationText.innerText += ' ';
    } else if (word == "stop") {
      this.translationText.innerText += '.';
    } else {
      this.translationText.innerText += ' ' + word;
    }
    this.translatedCard.innerHTML = " ";
    var clonedCard = document.createElement("div");
    clonedCard.className = "trained-gestures";

    var gestName = gestureCard.childNodes[0].innerText;
    var gestureName = document.createElement("h5");
    gestureName.innerText = gestName;
    clonedCard.appendChild(gestureName);

    var gestureImg = document.createElement("canvas");
    gestureImg.className = "trained_image";
    gestureImg.getContext('2d').drawImage(gestureCard.childNodes[1], 0, 0, 400, 180);
    clonedCard.appendChild(gestureImg);

    var gestAccuracy = document.createElement("h7");
    gestAccuracy.innerText = "Confidence: " + gestureAccuracy + "%";
    clonedCard.appendChild(gestAccuracy);

    this.translatedCard.appendChild(clonedCard);
    if (word != "start" && word != "stop") {
      this.speak(word);
    }
  }
  clearPara() {
    this.translationText.innerText = '';
    main.previousPrediction = -1;
    this.currentPredictedWords = []; 
    this.translatedCard.innerHTML = " ";
  }

  copyTranslation() {
    this.translationHolder.addEventListener('mousedown', () => {
      main.setStatusText("Text Copied!", "copy");
      const el = document.createElement('textarea'); 
      el.value = this.translationText.innerText; 
      el.setAttribute('readonly', ''); 
      el.style.position = 'absolute';
      el.style.left = '-9999px'; 
      document.body.appendChild(el); 
      const selected =
        document.getSelection().rangeCount > 0 
        ?
        document.getSelection().getRangeAt(0) 
        :
        false; 
      el.select(); 
      document.execCommand('copy'); 
      document.body.removeChild(el); 
      if (selected) { 
        document.getSelection().removeAllRanges(); 
        document.getSelection().addRange(selected); 
      }
    });
  }
  speak(word) {
    var utterThis = new SpeechSynthesisUtterance(word);

    utterThis.onerror = function (evt) {
      console.log("Error speaking");
    };

    utterThis.voice = this.voices[this.selectedVoice];
    utterThis.pitch = this.pitch;
    utterThis.rate = this.rate;

    this.synth.speak(utterThis);
  }
}

var main = null;
window.addEventListener('load', () => {
  main = new Main()
});