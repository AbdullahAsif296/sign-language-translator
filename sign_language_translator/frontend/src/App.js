import React from "react";
import WebcamFeed from "./components/WebcamFeed";
import "./App.css";

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Sign Language Translator</h1>
      </header>
      <main>
        <WebcamFeed />
      </main>
    </div>
  );
}

export default App;
