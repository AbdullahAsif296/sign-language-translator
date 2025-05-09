import React from "react";
import WebcamFeed from "./components/WebcamFeed";
import "./App.css";
import { FaHandPaper } from "react-icons/fa";

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>
          <FaHandPaper style={{ marginRight: "12px" }} /> Sign Language
          Translator
        </h1>
        <p className="app-subtitle">
          Translate sign language gestures to text in real-time
        </p>
      </header>
      <main>
        <WebcamFeed />
      </main>
      <footer className="app-footer">
        <p>
          © {new Date().getFullYear()} Sign Language Translator - Making
          communication accessible
        </p>
      </footer>
    </div>
  );
}

export default App;
