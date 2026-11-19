import { useEffect, useRef, useState } from 'react'
import './App.css'
import {
  FilesetResolver,
  FaceLandmarker,
  type FaceLandmarkerResult,
} from '@mediapipe/tasks-vision'

// Keep the wasm version in sync with the installed @mediapipe/tasks-vision package.
const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.3/wasm'
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'

function App() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const landmarkerRef = useRef<FaceLandmarker | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const rafRef = useRef<number | null>(null)

  const [modelReady, setModelReady] = useState(false)
  const [running, setRunning] = useState(false)
  const [faceDetected, setFaceDetected] = useState(false)

  // Load the face landmarker once, when the component mounts.
  useEffect(() => {
    let cancelled = false

    async function load() {
      const vision = await FilesetResolver.forVisionTasks(WASM_URL)
      const landmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
        runningMode: 'VIDEO',
        numFaces: 1,
        outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: true,
      })
      if (cancelled) {
        landmarker.close()
        return
      }
      landmarkerRef.current = landmarker
      setModelReady(true)
    }

    load().catch((error) => console.error('Could not load face model:', error))

    // Stop the camera, the loop and the model when the component unmounts.
    return () => {
      cancelled = true
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current)
      streamRef.current?.getTracks().forEach((track) => track.stop())
      landmarkerRef.current?.close()
      landmarkerRef.current = null
    }
  }, [])

  function handleResult(result: FaceLandmarkerResult) {
    setFaceDetected(result.faceLandmarks.length > 0)
  }

  function startLoop() {
    let lastTime = -1

    function loop() {
      const video = videoRef.current
      const landmarker = landmarkerRef.current
      if (video && landmarker && video.readyState >= 2 && video.currentTime !== lastTime) {
        lastTime = video.currentTime
        handleResult(landmarker.detectForVideo(video, performance.now()))
      }
      rafRef.current = requestAnimationFrame(loop)
    }

    loop()
  }

  async function start() {
    if (running) return
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      })
      streamRef.current = stream
      const video = videoRef.current
      if (!video) return
      video.srcObject = stream
      await video.play()
      setRunning(true)
      startLoop()
    } catch (error) {
      console.error('Could not access camera:', error)
    }
  }

  return (
    <>
      <section id="center">
        <div className="hero">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            style={{ transform: 'scaleX(-1)' }}
          />
        </div>
        {running && <p>{faceDetected ? 'Face detected' : 'No face detected'}</p>}
      </section>
      <button onClick={start} className="button" disabled={!modelReady || running}>
        {modelReady ? 'Start' : 'Loading model…'}
      </button>
    </>
  )
}

export default App
