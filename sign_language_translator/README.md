# Sign Language Translator

This project is a real-time sign language recognition web app. It uses a webcam to detect hand signs, predicts the corresponding letter, and helps users build words interactively. The app features **auto-correct**, **autocomplete** suggestions with improved common word prioritization, and convenient text editing tools.

## Features

- Real-time hand sign recognition using MediaPipe and a custom ML model
- Build words by showing signs one by one
- **Auto-correct**: Suggests the closest valid English word for the current input
- **Autocomplete**: Suggests common words first, prioritizing everyday vocabulary
- **Convenient text editing**:
  - Backspace button under the current word for easy character deletion
  - Clear button to reset the current word
  - Intuitive word and sentence management
- Modern, responsive UI with user-friendly controls

## How It Works

- The frontend captures hand landmarks and sends them to the backend for prediction.
- As you build a word, the frontend requests suggestions from the backend.
- The backend uses Python's `nltk` with both the standard words corpus and the Brown corpus to provide:
  - **Smart Autocomplete**: Prioritizes common English words over obscure ones
  - **Auto-correct**: Finds the closest valid English word using edit distance
- Suggestions are shown below the current word. Click to accept.

## Backend: Suggestion API

- **Endpoint:** `/suggest` (POST)
- **Request:** `{ "current_word": "helo" }`
- **Response:**
  ```json
  {
    "autocomplete": ["hello", "help", ...],
    "autocorrect": "hello"
  }
  ```
- Uses multiple word sources prioritized by commonality:
  1. Most frequent English words list
  2. NLTK Brown corpus (everyday language)
  3. Standard NLTK words corpus (full dictionary)

## Frontend: Usage

- As you sign, the current word is updated.
- Use the **backspace button** directly under the word to remove the last character.
- Suggestions appear as "Common Word Suggestions" with options you can click.
- Use the "Record" button to begin capturing signs, and "Pause" to temporarily stop.
- Click "Add to Sentence" to build complete sentences from your words.

## Setup

1. **Backend**
   - Install dependencies:
     ```bash
     pip install flask flask-cors nltk
     ```
   - Download NLTK words corpus (first run will auto-download, or run in Python):
     ```python
     import nltk
     nltk.download('words')
     ```
   - Start the backend:
     ```bash
     python src/api.py
     ```
2. **Frontend**
   - Install dependencies:
     ```bash
     npm install
     ```
   - Start the frontend:
     ```bash
     npm start
     ```

## Technologies Used

- **Frontend:** React, MediaPipe Hands, Axios
- **Backend:** Flask, NLTK, difflib
- **ML Model:** (your custom model for sign prediction)

## Extending Further

- Add user profiles, voice output, or practice/quiz modes
- Use a trie or more advanced ML for smarter suggestions
- Support multiple sign languages

---

**Enjoy building words with sign language and smart suggestions!**

## System Requirements

- **Operating System**: Windows 10/11, macOS 10.15+, or Linux
- **CPU**: Intel Core i5 or equivalent (4+ cores recommended)
- **RAM**: Minimum 8GB, 16GB recommended
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended for better performance)
- **Storage**: Minimum 2GB free space
- **Python**: 3.8 or higher
- **Node.js**: 14 or higher
- **Webcam**: HD webcam (720p or higher recommended)
- **Browser**: Chrome, Firefox, or Edge with WebGL support

## Features

- Real-time sign language to text conversion
- Multiple model ensemble (LSTM, CNN, Random Forest)
- Webcam integration with MediaPipe
- Context-aware sequence validation
- Advanced data augmentation techniques
- User-friendly React-based interface
- RESTful API architecture
- Offline-first design with local MediaPipe scripts

## Prerequisites

- Python 3.8 or higher
- Node.js 14 or higher
- Webcam access
- Modern web browser with WebGL support

## Installation

### Backend Setup

1. Navigate to the backend directory:

   ```bash
   cd backend
   ```

2. Create and activate a virtual environment:

   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

### Frontend Setup

1. Navigate to the frontend directory:

   ```bash
   cd frontend
   ```

2. Install Node.js dependencies:

   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

## Project Structure

```
sign_language_translator/
├── backend/                           # Flask backend server
│   ├── src/                          # Source code
│   │   ├── api.py                    # REST API endpoints
│   │   ├── enhanced_model.py         # Enhanced ML model architecture
│   │   ├── train_nn_model.py         # Neural network training script
│   │   ├── train_model.py            # Random forest training script
│   │   └── utils/                    # Utility modules
│   ├── data/                         # Data directories
│   ├── models/                       # Trained models
│   └── requirements.txt              # Python dependencies
├── frontend/                         # React frontend application
│   ├── public/                       # Static files
│   │   ├── mediapipe/               # Local MediaPipe scripts
│   │   └── index.html               # Main HTML file
│   ├── src/                         # Source code
│   │   ├── components/              # React components
│   │   └── App.js                   # Main application
│   └── package.json                 # Node.js dependencies
├── shared/                          # Shared resources
├── tests/                          # Test suite
└── docs/                           # Documentation
```

## Technical Architecture

### Machine Learning Pipeline

1. **Data Collection & Preprocessing**

   - Hand landmark detection using MediaPipe
   - Data normalization and augmentation
   - Sequence validation and correction

2. **Model Architecture**

   - LSTM with Attention (40% weight)
   - CNN with Transfer Learning (40% weight)
   - Random Forest Classifier (20% weight)

3. **Prediction Pipeline**
   - Real-time hand tracking
   - Feature extraction
   - Ensemble prediction
   - Confidence scoring
   - Sequence validation

### Frontend Architecture

1. **Webcam Integration**

   - Local MediaPipe scripts for reliability
   - Custom frame processing loop
   - Efficient hand tracking
   - Real-time visualization

2. **User Interface**
   - React-based components
   - Responsive design
   - Real-time feedback
   - Error handling and recovery

## API Documentation

### Endpoints

1. **Prediction**

   - `POST /predict`
   - Input: Hand landmarks
   - Output: Predicted sign with confidence

2. **Training**

   - `POST /train`
   - Input: Sign sequences
   - Purpose: Model updates

3. **Upload**
   - `POST /upload`
   - Purpose: Training data management

## Testing

Run the test suite:

```bash
# Backend tests
cd backend
python -m pytest tests/

# Frontend tests
cd frontend
npm test
```

## Performance Metrics

- Model Accuracy: >95%
- Real-time Processing: <100ms latency
- Sequence Validation Accuracy: >98%

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- MediaPipe for hand tracking
- TensorFlow for deep learning
- React for frontend development
- Flask for backend services

## Support

For support, email support@signlanguagetranslator.com or join our Slack channel.

## Troubleshooting

### Common Issues

1. **Camera Access Issues**

   - Error: Webcam not detected or access denied
   - Solution:
     - Grant camera permissions in browser settings
     - Ensure no other application is using the webcam
     - Try a different browser
     - Check browser console for specific error messages
     - Clear browser cache and refresh

2. **MediaPipe Script Loading Issues**

   - Error: Failed to load MediaPipe scripts
   - Solution:
     - Ensure local MediaPipe scripts are present in public/mediapipe/
     - Check browser console for specific error messages
     - Try clearing browser cache
     - Verify internet connection

3. **Memory Allocation Error**

   - Error: `Unable to allocate X MiB for an array`
   - Solution:
     - Close other memory-intensive applications
     - Increase system swap space
     - Use a machine with more RAM
     - Consider using a smaller model or reducing batch size

4. **Model Loading Failures**

   - Error: Model file not found or corrupted
   - Solution:
     - Verify model files are in correct location
     - Check file permissions
     - Re-download model files if necessary

5. **Performance Issues**
   - Slow translation or high latency
   - Solution:
     - Use a more powerful machine
     - Enable GPU acceleration if available
     - Reduce webcam resolution
     - Close background applications

### Getting Help

If you encounter issues not covered above:

1. Check the [GitHub Issues](https://github.com/your-repo/issues) page
2. Join our [Discord Community](https://discord.gg/your-invite)
3. Email support@signlanguagetranslator.com

## Recent Updates

- **May 2025**: Added backspace functionality for easier text editing
- **May 2025**: Improved word suggestions to prioritize common words
- **April 2025**: Enhanced UI with clearer labeling and feedback
- **March 2025**: Optimized hand tracking performance
