require('@tensorflow/tfjs-node')
const faceapi = require("@vladmandic/face-api");
const { Canvas, Image, ImageData } = require("canvas");
const canvas = require("canvas");
const yaml = require('js-yaml');
const prompt = require('prompt-sync')();
const fs = require("fs")
faceapi.env.monkeyPatch({ Canvas, Image, ImageData })

async function main(image) {
    await faceapi.nets.faceRecognitionNet.loadFromDisk(__dirname + "/models");
    console.log("[Info] Loaded faceRecognitionNet")
    await faceapi.nets.faceLandmark68Net.loadFromDisk(__dirname + "/models");
    console.log("[Info] Loaded faceLandmark68Net")
    await faceapi.nets.ssdMobilenetv1.loadFromDisk(__dirname + "/models");
    console.log("[Info] Loaded ssdMobilenetv1")
    let faces = require("./data.json")
    for (i = 0; i < faces.length; i++) {
      for (j = 0; j < faces[i].descriptions.length; j++) {
        faces[i].descriptions[j] = new Float32Array(Object.values(faces[i].descriptions[j]));
      }
      faces[i] = new faceapi.LabeledFaceDescriptors(faces[i].label, faces[i].descriptions);
    }
  
    const faceMatcher = new faceapi.FaceMatcher(faces, 0.6);
  
    const img = await canvas.loadImage(image);
    let temp = faceapi.createCanvasFromMedia(img);
    const displaySize = { width: img.width, height: img.height };
    faceapi.matchDimensions(temp, displaySize);
  
    const detections = await faceapi.detectAllFaces(img).withFaceLandmarks().withFaceDescriptors();
    const resizedDetections = faceapi.resizeResults(detections, displaySize);
    const results = resizedDetections.map((d) => faceMatcher.findBestMatch(d.descriptor));
    
    if (results.length == 0) {
        console.log("Kathryn not detected!")
    } else {
       console.log("Kathryn detected!")
    }
  }
var d = prompt("Please provide a URL or local file path: ")
if (!d) process.exit()
main(d)
