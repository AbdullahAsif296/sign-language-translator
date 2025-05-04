// This script should be run to set up MediaPipe files
const MEDIAPIPE_VERSION = "0.4.1646424915";
const MEDIAPIPE_FILES = [
  "hands.js",
  "hands_solution_packed_assets_loader.js",
  "hands_solution_packed_assets.data",
  "hands_solution_simd_wasm_bin.js",
  "hands_solution_simd_wasm_bin.wasm",
  "hands_solution_wasm_bin.js",
  "hands_solution_wasm_bin.wasm",
];

const BASE_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/hands@${MEDIAPIPE_VERSION}/solution`;

async function downloadFile(url, filename) {
  try {
    console.log(`Downloading ${filename} from ${url}...`);
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to download ${filename}: ${response.statusText}`);
    }
    const blob = await response.blob();
    const objectUrl = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = objectUrl;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(objectUrl);
    console.log(`Successfully downloaded ${filename}`);
  } catch (error) {
    console.error(`Error downloading ${filename}:`, error);
    throw error;
  }
}

async function setupMediaPipe() {
  console.log("Setting up MediaPipe files...");

  try {
    for (const file of MEDIAPIPE_FILES) {
      const url = `${BASE_URL}/${file}`;
      await downloadFile(url, file);
    }

    console.log("MediaPipe setup complete!");
  } catch (error) {
    console.error("MediaPipe setup failed:", error);
    throw error;
  }
}

// Run setup
setupMediaPipe().catch((error) => {
  console.error("Failed to set up MediaPipe:", error);
});
