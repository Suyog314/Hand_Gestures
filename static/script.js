
const video = document.getElementById('video');
const output = document.getElementById('output');

navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => video.srcObject = stream);

const canvas = document.createElement('canvas');

function captureAndSend() {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  const dataUrl = canvas.toDataURL('image/jpeg');

  fetch('/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ image: dataUrl })
  })
    .then(res => res.json())
    .then(data => {
      output.innerText = data.prediction;

      // Speak
      const utterance = new SpeechSynthesisUtterance(data.prediction);
      speechSynthesis.speak(utterance);
    });

  setTimeout(captureAndSend, 2000); // every 2 seconds
}

setTimeout(captureAndSend, 2000);
