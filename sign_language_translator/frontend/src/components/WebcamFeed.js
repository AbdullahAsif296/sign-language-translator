import React, { useRef, useEffect, useState } from "react";
import axios from "axios";
import {
  FaPlus,
  FaEraser,
  FaBackspace,
  FaTrash,
  FaPlay,
  FaPause,
  FaStop,
  FaRecordVinyl,
  FaExclamationTriangle,
  FaExclamationCircle,
  FaHandPointRight,
  FaSave,
} from "react-icons/fa";

// Configure paths for loading MediaPipe scripts using CDN links
const mpHandsScript = "https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js";
const mpDrawingScript =
  "https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js";
const mpCameraScript =
  "https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js";

// Performance configuration for camera and hand detection
const CAMERA_CONFIG = {
  width: 480, // Lower resolution for better performance
  height: 360,
  frameRate: { ideal: 25, max: 30 }, // Limit frame rate for better performance
};

const HANDS_CONFIG = {
  selfieMode: true, // Mirror mode for more natural interaction
  maxNumHands: 1, // Only track one hand for better performance
  modelComplexity: 0, // Use a lighter model (0=light, 1=full)
  minDetectionConfidence: 0.6,
  minTrackingConfidence: 0.5,
};

// Optimized request cooldown time for better response while maintaining performance
const REQUEST_COOLDOWN = 700;

const BANNED_WORDS = ["nigger", "...", "offensiveword2", "offensiveword3"];
function isCleanWord(word) {
  return !BANNED_WORDS.some((bad) => word.toLowerCase().includes(bad));
}

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
    // Check if script is already loaded
    const existingScript = document.querySelector(`script[src="${src}"]`);
    if (existingScript) {
      resolve();
      return;
    }

    const script = document.createElement("script");
    script.src = src;
    script.crossOrigin = "anonymous"; // Add CORS support
    script.async = true;

    script.onload = () => {
      console.log(`Successfully loaded: ${src}`);
      resolve();
    };

    script.onerror = (error) => {
      console.error(`Failed to load script: ${src}`, error);
      // Remove the failed script to allow retry
      script.remove();
      reject(error);
    };

    document.head.appendChild(script); // Append to head instead of body
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
  const [sentence, setSentence] = useState([]);
  const [sentenceError, setSentenceError] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const lastAppendedSign = useRef(null);
  const lastRequestTime = useRef(0);
  const lastSignTime = useRef(0);
  const SIGN_TIMEOUT = 2000; // 2 seconds timeout between signs
  const connectionCheckInterval = useRef(null);
  const handDetectionTimeout = useRef(null);
  const isRecordingRef = useRef(isRecording);
  const [suggestions, setSuggestions] = useState([]);
  const [autocorrect, setAutocorrect] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [retryCount, setRetryCount] = useState(0);

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
    // Use a flag to prevent race conditions when initializing camera
    let isMounted = true;
    let cameraStream = null;
    let handsInstance = null;
    let cameraInstance = null;
    let frameCount = 0;

    const initializeCamera = async () => {
      setIsLoading(true);
      try {
        // Request camera permission with optimized settings
        console.log("Requesting camera access with optimized settings...");
        const stream = await navigator.mediaDevices.getUserMedia({
          video: CAMERA_CONFIG,
        });

        // Store stream for cleanup
        cameraStream = stream;

        if (!isMounted) {
          stopStream(stream);
          return;
        }

        if (videoRef.current) {
          videoRef.current.srcObject = stream;

          videoRef.current.onloadedmetadata = async () => {
            try {
              if (isMounted && videoRef.current) {
                await videoRef.current.play();
                console.log("Video playback started with optimized settings");
                await loadMediaPipeLibraries();
              }
            } catch (err) {
              console.error("Error starting video playback:", err);
              setError(`Could not play video: ${err.message}`);
              setIsLoading(false);
            }
          };
        }
      } catch (err) {
        console.error("Camera access error:", err);
        setError(
          `Camera access denied: ${err.message}. Please enable camera permissions and refresh.`
        );
        setIsLoading(false);
      }
    };

    const loadMediaPipeLibraries = async () => {
      try {
        console.log("Loading MediaPipe libraries...");

        // Load all MediaPipe scripts one by one to ensure proper loading sequence
        try {
          // First try to load drawing utils
          await loadScript(mpDrawingScript);
          console.log("Successfully loaded MediaPipe drawing utils");

          // Then load camera utils
          await loadScript(mpCameraScript);
          console.log("Successfully loaded MediaPipe camera utils");

          // Finally load hands
          await loadScript(mpHandsScript);
          console.log("Successfully loaded MediaPipe hands");
        } catch (error) {
          console.warn("Error loading MediaPipe libraries:", error);
          console.warn("Trying alternative CDN URLs...");

          // Fallback to specific versions if needed
          try {
            await loadScript(
              "https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils@0.3.1632713810/drawing_utils.js"
            );
            await loadScript(
              "https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils@0.3.1632713810/camera_utils.js"
            );
            await loadScript(
              "https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1646424915/hands.js"
            );
            console.log(
              "Successfully loaded MediaPipe from alternative CDN URLs"
            );
          } catch (fallbackError) {
            throw new Error(
              `All loading attempts failed: ${fallbackError.message}`
            );
          }
        }

        if (!isMounted) return;

        console.log("MediaPipe libraries loaded successfully");
        initializeHandTracking();
      } catch (err) {
        console.error("All MediaPipe loading attempts failed:", err);
        setError(
          `Could not load hand tracking libraries. Please refresh and try again.`
        );
        setIsLoading(false);
      }
    };

    const initializeHandTracking = () => {
      try {
        if (!window.Hands) {
          console.error("MediaPipe Hands not available");
          setError(
            "Hand tracking library not available. Please refresh the page."
          );
          return;
        }

        console.log("Initializing MediaPipe Hands with optimized settings...");
        handsInstance = new window.Hands({
          locateFile: (file) => {
            // Always use CDN path to ensure reliable loading
            return `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1646424915/${file}`;
          },
        });

        // Use optimized settings for better performance
        handsInstance.setOptions(HANDS_CONFIG);

        // Register callback
        handsInstance.onResults((results) => {
          // Turn off loading indicator on first successful results
          setIsLoading(false);
          // Call the actual results handler
          onResults(results);
        });

        // Start camera processing
        startCameraProcessing(handsInstance);
      } catch (err) {
        console.error("Error initializing hand tracking:", err);
        setError(`Hand tracking initialization failed: ${err.message}`);
        setIsLoading(false);
      }
    };

    const startCameraProcessing = (hands) => {
      if (!videoRef.current || videoRef.current.readyState < 2) {
        setTimeout(() => {
          if (isMounted) startCameraProcessing(hands);
        }, 200); // Reduced retry time
        return;
      }

      try {
        if (window.Camera) {
          console.log("Starting MediaPipe Camera with frame rate control...");
          cameraInstance = new window.Camera(videoRef.current, {
            onFrame: async () => {
              if (!isMounted) return;

              try {
                // Process every other frame to reduce CPU usage
                frameCount++;
                if (frameCount % 2 === 0 && videoRef.current?.readyState >= 2) {
                  await hands.send({ image: videoRef.current });
                }
              } catch (err) {
                // Silent error handling
              }
            },
            width: CAMERA_CONFIG.width,
            height: CAMERA_CONFIG.height,
          });

          cameraInstance.start();
        } else {
          console.warn("Using optimized manual frame processing");

          let lastFrameTime = 0;
          const FRAME_INTERVAL = 1000 / 20; // Target 20fps for better performance

          const processFrame = async (timestamp) => {
            if (!isMounted) return;

            // Skip frames to achieve desired frame rate
            if (timestamp - lastFrameTime >= FRAME_INTERVAL) {
              lastFrameTime = timestamp;

              try {
                if (videoRef.current?.readyState >= 2) {
                  await hands.send({ image: videoRef.current });
                }
              } catch (err) {
                // Silent error handling
              }
            }

            if (isMounted) {
              requestAnimationFrame(processFrame);
            }
          };

          requestAnimationFrame(processFrame);
        }
      } catch (err) {
        console.error("Error starting camera processing:", err);
        setError(`Failed to start hand tracking: ${err.message}`);
      }
    };

    const stopStream = (stream) => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };

    // Start the initialization process
    initializeCamera();

    // Cleanup function
    return () => {
      isMounted = false;

      if (cameraInstance) {
        try {
          cameraInstance.stop();
        } catch (err) {
          console.error("Error stopping camera:", err);
        }
      }

      stopStream(cameraStream);

      if (handDetectionTimeout.current) {
        clearTimeout(handDetectionTimeout.current);
      }
    };
  }, [retryCount]);

  useEffect(() => {
    isRecordingRef.current = isRecording;
  }, [isRecording]);

  // Check backend connection on component mount
  useEffect(() => {
    // Initial check
    checkBackendConnection();

    // Check connection less frequently (10 seconds) to reduce overhead
    connectionCheckInterval.current = setInterval(
      checkBackendConnection,
      10000
    );

    // Cleanup on unmount
    return () => {
      if (connectionCheckInterval.current) {
        clearInterval(connectionCheckInterval.current);
      }
    };
  }, []);

  // Fetch suggestions when currentWord changes
  useEffect(() => {
    const fetchSuggestions = async () => {
      const word = currentWord.join("");
      if (word.length > 0) {
        try {
          const response = await api.post("/suggest", { current_word: word });
          setSuggestions(response.data.autocomplete || []);
          setAutocorrect(response.data.autocorrect || null);
        } catch (e) {
          setSuggestions([]);
          setAutocorrect(null);
        }
      } else {
        setSuggestions([]);
        setAutocorrect(null);
      }
    };
    fetchSuggestions();
  }, [currentWord]);

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
    // Skip rendering if canvas isn't available
    if (!canvasRef.current) return;

    const canvasElement = canvasRef.current;
    const canvasCtx = canvasElement.getContext("2d", { alpha: false }); // Optimize rendering with alpha: false

    // Set canvas dimensions only once to avoid performance hit
    if (
      canvasElement.width !== CAMERA_CONFIG.width ||
      canvasElement.height !== CAMERA_CONFIG.height
    ) {
      canvasElement.width = CAMERA_CONFIG.width;
      canvasElement.height = CAMERA_CONFIG.height;
    }

    // Optimize rendering
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    // Use faster image drawing method
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

    // Use shorter debounce time for more responsive UI
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
      }, 200); // Reduced from 300ms for more responsive UI
    }

    if (hasHand) {
      try {
        // Draw hand landmarks with optimized rendering
        window.drawConnectors(
          canvasCtx,
          results.multiHandLandmarks[0],
          window.HAND_CONNECTIONS,
          { color: "#00FF00", lineWidth: 2 }
        );

        window.drawLandmarks(
          canvasCtx,
          results.multiHandLandmarks[0],
          { color: "#FF0000", lineWidth: 1, radius: 3 } // Reduced radius for better performance
        );

        // Only send requests at controlled intervals
        if (
          canSendRequest &&
          currentTime - lastRequestTime.current >= REQUEST_COOLDOWN
        ) {
          // Optimize landmark extraction
          const landmarks = results.multiHandLandmarks[0].flatMap((lm) => [
            lm.x,
            lm.y,
            lm.z,
          ]);

          // Use non-blocking request pattern
          setTimeout(() => sendLandmarksToAPI(landmarks), 0);

          setCanSendRequest(false);
          lastRequestTime.current = currentTime;

          // Reset canSendRequest after cooldown
          setTimeout(() => {
            setCanSendRequest(true);
          }, REQUEST_COOLDOWN);
        }
      } catch (err) {
        // Silent error handling for render errors
      }
    }

    canvasCtx.restore();
  };

  const sendLandmarksToAPI = async (landmarks) => {
    try {
      // Skip backend connection check on every request to reduce lag
      // Only check if we've previously detected a connection issue
      if (!isBackendConnected) {
        const isConnected = await checkBackendConnection();
        if (!isConnected) {
          console.error("Backend is not connected");
          return;
        }
      }

      // Quick validation without expensive operations
      if (!landmarks?.length) return;

      // Use a faster validation approach - just check first few landmarks
      const sampleSize = Math.min(landmarks.length, 10);
      for (let i = 0; i < sampleSize; i++) {
        if (
          landmarks[i] === null ||
          landmarks[i] === undefined ||
          isNaN(landmarks[i]) ||
          !isFinite(landmarks[i])
        ) {
          return;
        }
      }

      // Reduce logging for better performance
      if (process.env.NODE_ENV !== "production") {
        console.log("Sending landmarks to API, length:", landmarks.length);
      }

      const response = await api.post("/predict", {
        landmarks: landmarks,
      });

      if (response.data.error) {
        console.error("Backend error:", response.data.error);
        setPrediction(null);
        setError(response.data.error);
        return;
      }

      if (!response.data.prediction) {
        setPrediction(null);
        return;
      }

      const prediction = response.data.prediction;
      const confidence = parseFloat(response.data.confidence);

      if (isNaN(confidence) || confidence < 0 || confidence > 1) {
        setPrediction(null);
        return;
      }

      setPrediction({
        prediction: prediction,
        confidence: confidence,
      });
      setError(null);

      // Only append if recording, confidence is high enough, and sign is new
      if (
        isRecordingRef.current &&
        !isPaused &&
        confidence > 0.55 && // Slightly increased threshold for better accuracy
        prediction !== lastAppendedSign.current
      ) {
        setCurrentWord((prev) => [...prev, prediction]);
        lastAppendedSign.current = prediction;
      }

      if (!isRecordingRef.current || isPaused) {
        lastAppendedSign.current = null;
      }
    } catch (error) {
      // Simplified error handling to reduce UI updates
      console.error("API request failed:", error.message);

      // Only update UI for persistent errors
      if (
        error.message.includes("NetworkError") ||
        error.message.includes("Failed to fetch")
      ) {
        setIsBackendConnected(false);
        setError("Connection to backend lost. Trying to reconnect...");

        // Try to reconnect after a delay
        setTimeout(checkBackendConnection, 3000);
      }

      setPrediction(null);
    }
  };

  // Remove last character from current word
  const removeLastCharacter = () => {
    if (currentWord.length > 0) {
      setCurrentWord((prev) => prev.slice(0, -1));
    }
  };

  // Accept a suggestion
  const acceptSuggestion = (word) => {
    setCurrentWord(word.split(""));
    setSuggestions([]);
    setAutocorrect(null);
  };

  // Add word to sentence with moderation
  const addWordToSentence = () => {
    const word = currentWord.join("");
    if (word.length > 0 && isCleanWord(word)) {
      setSentence((prev) => [...prev, word]);
      setCurrentWord([]);
      setSentenceError("");
    } else if (!isCleanWord(word)) {
      setSentenceError("Inappropriate word detected. Please try again.");
      setCurrentWord([]);
    }
  };

  // Remove last word from sentence
  const removeLastWord = () => {
    setSentence((prev) => prev.slice(0, -1));
  };

  // Clear the entire sentence
  const clearSentence = () => {
    setSentence([]);
    setSentenceError("");
  };

  // Reset and restart the hand tracking in case of failures
  const resetHandTracking = () => {
    setIsLoading(true);
    setError(null);
    setRetryCount((prev) => prev + 1);
  };

  return (
    <div className="webcam-container">
      <div className="webcam-section">
        {isLoading && (
          <div
            className="loading-container"
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              backgroundColor: "rgba(0,0,0,0.7)",
              color: "white",
              zIndex: 10,
              borderRadius: "16px",
            }}
          >
            <div
              className="loading-spinner"
              style={{
                width: "50px",
                height: "50px",
                border: "5px solid rgba(255,255,255,0.3)",
                borderRadius: "50%",
                borderTop: "5px solid #ffffff",
                animation: "spin 1s linear infinite",
                marginBottom: "15px",
              }}
            ></div>
            <p>Loading hand tracking...</p>
            <style jsx="true">{`
              @keyframes spin {
                0% {
                  transform: rotate(0deg);
                }
                100% {
                  transform: rotate(360deg);
                }
              }
            `}</style>
          </div>
        )}
        <video
          ref={videoRef}
          style={{
            position: "absolute",
            width: "1px",
            height: "1px",
            opacity: 0,
            pointerEvents: "none",
          }}
          playsInline
          autoPlay
          muted
        />
        <canvas
          ref={canvasRef}
          width={CAMERA_CONFIG.width}
          height={CAMERA_CONFIG.height}
          style={{
            maxWidth: "100%",
            height: "auto",
            display: "block",
          }}
        />
        <div className="prediction-container">
          {!isBackendConnected && (
            <div className="warning-message">
              <FaExclamationTriangle style={{ fontSize: "1.2rem" }} />
              <span>Backend server is not connected</span>
            </div>
          )}
          {error && (
            <div className="error-message">
              <FaExclamationCircle style={{ fontSize: "1.2rem" }} />
              <span>Error: {error}</span>
              <button
                onClick={resetHandTracking}
                style={{
                  marginLeft: "10px",
                  padding: "5px 10px",
                  backgroundColor: "#555",
                  color: "white",
                  border: "none",
                  borderRadius: "4px",
                  cursor: "pointer",
                  fontSize: "14px",
                }}
              >
                Retry
              </button>
            </div>
          )}
          {prediction && (
            <div className="prediction-results">
              <h3>
                <FaHandPointRight /> Current Sign:
              </h3>
              <p>
                {prediction.prediction}{" "}
                <span style={{ fontSize: "0.8rem", color: "#666" }}>
                  ({Math.round(prediction.confidence * 100)}% confidence)
                </span>
              </p>
            </div>
          )}
        </div>
      </div>

      <div className="right-panel">
        <div className="controls-container">
          <button
            className={`record-button ${isRecording ? "recording" : ""}`}
            onClick={toggleRecording}
            title={isRecording ? "Stop Recording" : "Start Recording"}
          >
            {isRecording ? (
              <>
                <FaStop /> Stop
              </>
            ) : (
              <>
                <FaRecordVinyl /> Record
              </>
            )}
          </button>
          {isRecording && (
            <button
              className={`pause-button ${isPaused ? "paused" : ""}`}
              onClick={togglePause}
              title={isPaused ? "Resume" : "Pause"}
            >
              {isPaused ? (
                <>
                  <FaPlay /> Resume
                </>
              ) : (
                <>
                  <FaPause /> Pause
                </>
              )}
            </button>
          )}
          <button
            className="clear-button"
            onClick={clearWord}
            title="Clear Current Word"
          >
            <FaEraser /> Clear
          </button>
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

          <div className="word-actions">
            {currentWord.length > 0 && (
              <>
                <button
                  className="backspace-button"
                  onClick={removeLastCharacter}
                  title="Remove Last Character"
                >
                  <FaBackspace /> Backspace
                </button>
                <button className="add-button" onClick={addWordToSentence}>
                  <FaPlus /> Add to Sentence
                </button>
              </>
            )}
            {currentWord.length === 0 && (
              <p className="word-instruction">Start signing to create a word</p>
            )}
          </div>

          {suggestions.length > 0 && (
            <div className="suggestions-box">
              <div>
                <strong>Common Word Suggestions:</strong>
              </div>
              <ul className="suggestion-list">
                {suggestions.map((word, index) => (
                  <li key={index} onClick={() => acceptSuggestion(word)}>
                    {word}
                  </li>
                ))}
              </ul>
              {autocorrect && (
                <div style={{ marginTop: "8px" }}>
                  <strong>Did you mean:</strong>
                  <span
                    className="autocorrect-suggestion"
                    onClick={() => acceptSuggestion(autocorrect)}
                  >
                    {autocorrect}
                  </span>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="sentence-container">
          <h3>
            <FaSave /> Sentence:
          </h3>
          <div className="sentence-display">
            {sentence.length > 0 ? (
              sentence.join(" ")
            ) : (
              <p className="empty-sentence">Your sentence will appear here…</p>
            )}
          </div>

          {sentence.length > 0 && (
            <div className="sentence-actions">
              <button
                onClick={removeLastWord}
                className="remove-word-button"
                title="Remove Last Word"
              >
                <FaBackspace /> Undo Last
              </button>
              <button
                onClick={clearSentence}
                className="clear-sentence-button"
                title="Clear Sentence"
              >
                <FaTrash /> Clear All
              </button>
            </div>
          )}

          {sentenceError && (
            <div className="sentence-error">
              <FaExclamationCircle style={{ marginRight: "8px" }} />
              {sentenceError}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default WebcamFeed;
