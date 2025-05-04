import React, { useRef, useEffect, useState } from "react";
import axios from "axios";

const mpHandsScript = "https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js";
const mpCameraScript =
  "https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js";
const mpDrawingScript =
  "https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js";

// Configure axios defaults
const api = axios.create({
  baseURL: "http://localhost:5000", // Backend server URL
  timeout: 10000, // 10 seconds timeout
  headers: {
    "Content-Type": "application/json",
    Accept: "application/json",
  },
  withCredentials: false, // Disable credentials
});

// Add request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log("Making request to:", config.url);
    return config;
  },
  (error) => {
    console.error("Request error:", error);
    return Promise.reject(error);
  }
);

// Add response interceptor for logging
api.interceptors.response.use(
  (response) => {
    console.log(
      "Received response:",
      response.status,
      "from",
      response.config.url
    );
    return response;
  },
  (error) => {
    console.error("Response error from", error.config?.url, ":", error);
    return Promise.reject(error);
  }
);

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = src;
    script.onload = resolve;
    script.onerror = reject;
    document.body.appendChild(script);
  });
}

const WebcamFeed = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);
  const [handDetected, setHandDetected] = useState(false);
  const [canSendRequest, setCanSendRequest] = useState(true);
  const [isBackendConnected, setIsBackendConnected] = useState(false);
  const [currentWord, setCurrentWord] = useState([]);
  const [isRecording, setIsRecording] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const lastAppendedSign = useRef(null);
  const lastRequestTime = useRef(0);
  const lastSignTime = useRef(0);
  const REQUEST_COOLDOWN = 1000; // 1 second cooldown between requests
  const SIGN_TIMEOUT = 2000; // 2 seconds timeout between signs
  const connectionCheckInterval = useRef(null);
  const handDetectionTimeout = useRef(null);
  const isRecordingRef = useRef(isRecording);

  // Define checkBackendConnection function
  const checkBackendConnection = async () => {
    try {
      console.log("Checking backend connection...");
      const response = await api.get("/health");
      console.log("Backend health check response:", response.data);

      if (response.data.status === "healthy") {
        console.log("Backend is connected and healthy");
        setIsBackendConnected(true);
        setError(null);
        return true;
      } else {
        console.error("Backend is unhealthy:", response.data);
        setIsBackendConnected(false);
        setError("Backend server is unhealthy");
        return false;
      }
    } catch (error) {
      console.error("Backend connection failed:", error);
      setIsBackendConnected(false);
      setError(
        "Cannot connect to backend server. Please make sure it is running."
      );
      return false;
    }
  };

  useEffect(() => {
    Promise.all([
      loadScript(mpHandsScript),
      loadScript(mpCameraScript),
      loadScript(mpDrawingScript),
    ]).then(() => {
      // eslint-disable-next-line no-undef
      const hands = new window.Hands({
        locateFile: (file) =>
          `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
      });
      hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.7,
        minTrackingConfidence: 0.7,
      });
      hands.onResults(onResults);

      // eslint-disable-next-line no-undef
      const camera = new window.Camera(videoRef.current, {
        onFrame: async () => {
          await hands.send({ image: videoRef.current });
        },
        width: 640,
        height: 480,
      });
      camera.start();
    });

    // Cleanup function
    return () => {
      if (handDetectionTimeout.current) {
        clearTimeout(handDetectionTimeout.current);
      }
      if (connectionCheckInterval.current) {
        clearInterval(connectionCheckInterval.current);
      }
    };
  }, []);

  useEffect(() => {
    isRecordingRef.current = isRecording;
  }, [isRecording]);

  // Check backend connection on component mount
  useEffect(() => {
    // Initial check
    checkBackendConnection();

    // Check connection every 5 seconds
    connectionCheckInterval.current = setInterval(checkBackendConnection, 5000);

    // Cleanup on unmount
    return () => {
      if (connectionCheckInterval.current) {
        clearInterval(connectionCheckInterval.current);
      }
    };
  }, []);

  const addToWord = (sign) => {
    if (isRecording && !isPaused && sign) {
      // Simply append the new sign to the current word
      setCurrentWord((prev) => [...prev, sign]);
    }
  };

  const clearWord = () => {
    setCurrentWord([]);
    lastAppendedSign.current = null;
  };

  const toggleRecording = () => {
    setIsRecording(!isRecording);
    setIsPaused(false);
    lastAppendedSign.current = null;
  };

  const togglePause = () => {
    if (isRecording) {
      setIsPaused(!isPaused);
      lastAppendedSign.current = null;
    }
  };

  const onResults = (results) => {
    const canvasElement = canvasRef.current;
    const canvasCtx = canvasElement.getContext("2d");
    canvasElement.width = 640;
    canvasElement.height = 480;
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(
      results.image,
      0,
      0,
      canvasElement.width,
      canvasElement.height
    );

    const hasHand =
      results.multiHandLandmarks && results.multiHandLandmarks.length > 0;
    const currentTime = Date.now();

    // Update hand detection state with debounce
    if (hasHand !== handDetected) {
      if (handDetectionTimeout.current) {
        clearTimeout(handDetectionTimeout.current);
      }

      handDetectionTimeout.current = setTimeout(() => {
        setHandDetected(hasHand);
        if (!hasHand) {
          setPrediction(null);
          setCanSendRequest(true);
        }
      }, 300); // 300ms debounce for hand detection
    }

    if (hasHand) {
      // eslint-disable-next-line no-undef
      window.drawConnectors(
        canvasCtx,
        results.multiHandLandmarks[0],
        window.HAND_CONNECTIONS,
        { color: "#00FF00", lineWidth: 2 }
      );
      // eslint-disable-next-line no-undef
      window.drawLandmarks(canvasCtx, results.multiHandLandmarks[0], {
        color: "#FF0000",
        lineWidth: 1,
      });

      // Only send request if hand is newly detected and cooldown period has passed
      if (
        canSendRequest &&
        currentTime - lastRequestTime.current >= REQUEST_COOLDOWN
      ) {
        const landmarks = results.multiHandLandmarks[0].flatMap((lm) => [
          lm.x,
          lm.y,
          lm.z,
        ]);

        // Send to backend
        sendLandmarksToAPI(landmarks);
        setCanSendRequest(false);
        lastRequestTime.current = currentTime;

        // Reset canSendRequest after cooldown
        setTimeout(() => {
          setCanSendRequest(true);
        }, REQUEST_COOLDOWN);
      }
    }
    canvasCtx.restore();
  };

  const sendLandmarksToAPI = async (landmarks) => {
    try {
      // First check if backend is connected
      const isConnected = await checkBackendConnection();
      if (!isConnected) {
        console.error("Backend is not connected");
        setError(
          "Backend server is not connected. Please check if it is running."
        );
        return;
      }

      // Validate landmarks
      if (!landmarks || !Array.isArray(landmarks) || landmarks.length === 0) {
        console.error("Invalid landmarks data:", landmarks);
        return;
      }

      // Check for NaN or invalid values
      const hasInvalidValues = landmarks.some(
        (value) =>
          value === null ||
          value === undefined ||
          isNaN(value) ||
          !isFinite(value)
      );

      if (hasInvalidValues) {
        console.error("Landmarks contain invalid values:", landmarks);
        return;
      }

      console.log(
        "Sending landmarks to API. Number of landmarks:",
        landmarks.length
      );
      console.log("First few landmarks:", landmarks.slice(0, 5));

      const response = await api.post("/predict", {
        landmarks: landmarks,
      });

      console.log("API Response:", response.data);

      if (response.data.error) {
        console.error("Backend error:", response.data.error);
        if (response.data.traceback) {
          console.error("Traceback:", response.data.traceback);
        }
        setPrediction(null);
        setError(response.data.error);

        // If it's a server error, try to reconnect
        if (response.status >= 500) {
          console.log("Attempting to reconnect to backend...");
          await checkBackendConnection();
        }
      } else if (!response.data.prediction) {
        console.error("No prediction received from backend");
        setPrediction(null);
        setError("No prediction received from backend");
      } else {
        const prediction = response.data.prediction;
        const confidence = parseFloat(response.data.confidence);

        if (isNaN(confidence) || confidence < 0 || confidence > 1) {
          console.error("Invalid confidence value received:", confidence);
          setPrediction(null);
          setError("Invalid confidence value received from backend");
          return;
        }

        setPrediction({
          prediction: prediction,
          confidence: confidence,
        });
        setError(null);

        // Debug logging
        console.log(
          "isRecording:",
          isRecordingRef.current,
          "isPaused:",
          isPaused,
          "confidence:",
          confidence,
          "prediction:",
          prediction,
          "lastAppendedSign:",
          lastAppendedSign.current
        );

        // Only append if recording, confidence is high enough, and sign is new
        if (
          isRecordingRef.current &&
          !isPaused &&
          confidence > 0.5 &&
          prediction !== lastAppendedSign.current
        ) {
          console.log("Appending to word:", prediction, confidence);
          setCurrentWord((prev) => [...prev, prediction]);
          lastAppendedSign.current = prediction;
        }

        // Reset lastAppendedSign if not recording or paused
        if (!isRecordingRef.current || isPaused) {
          lastAppendedSign.current = null;
        }
      }
    } catch (error) {
      console.error("API request failed:", error);
      if (error.response) {
        console.error("Error response data:", error.response.data);
        setError(error.response.data.error || "API request failed");

        // If it's a server error, try to reconnect
        if (error.response.status >= 500) {
          console.log("Attempting to reconnect to backend...");
          await checkBackendConnection();
        }
      } else if (error.request) {
        console.error("No response received:", error.request);
        setError(
          "No response from server. Please check if the backend is running."
        );
        setIsBackendConnected(false);

        // Try to reconnect after a delay
        setTimeout(async () => {
          console.log("Attempting to reconnect to backend...");
          await checkBackendConnection();
        }, 5000);
      } else {
        console.error("Error setting up request:", error.message);
        setError(error.message);
      }
      setPrediction(null);
    }
  };

  return (
    <div className="webcam-container">
      <div className="webcam-section">
        <video ref={videoRef} style={{ display: "none" }} />
        <canvas ref={canvasRef} width={640} height={480} />
      </div>

      <div className="right-panel">
        <div className="controls-container">
          <button
            className={`record-button ${isRecording ? "recording" : ""}`}
            onClick={toggleRecording}
          >
            <span className="status-indicator"></span>
            {isRecording ? "Stop Recording" : "Start Recording"}
          </button>
          {isRecording && (
            <button
              className={`pause-button ${isPaused ? "paused" : ""}`}
              onClick={togglePause}
            >
              {isPaused ? "Resume" : "Pause"}
            </button>
          )}
          <button className="clear-button" onClick={clearWord}>
            Clear Word
          </button>
        </div>

        <div className="prediction-container">
          {!isBackendConnected && (
            <div className="warning-message">
              <span className="status-indicator status-disconnected"></span>
              Warning: Backend server is not connected
            </div>
          )}
          {error && (
            <div className="error-message">
              <span className="status-indicator status-disconnected"></span>
              Error: {error}
            </div>
          )}
          {prediction && (
            <div className="prediction-results">
              <h3>Current Sign:</h3>
              <p>
                <strong>Sign:</strong> {prediction.prediction}
              </p>
              <p>
                <strong>Confidence:</strong>{" "}
                {(prediction.confidence * 100).toFixed(2)}%
              </p>
            </div>
          )}
        </div>

        <div className="current-word-container">
          <h3>Current Word:</h3>
          <div className="word-display">
            {currentWord.length > 0 ? (
              currentWord.join("")
            ) : (
              <p className="empty-word">No signs recorded yet</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default WebcamFeed;
