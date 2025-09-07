/*
 * ===============================================
 * Arduino Script for Thermal Sensor-Based People Counting
 * ===============================================
 *
 * Adjustable Variables:
 * 1. Wi-Fi Credentials:
 *    - ssid, password: Set these to connect to your local Wi-Fi network.
 * 2. Server Details:
 *    - serverIP, serverPort: Specify the IP and port of your server.
 * 3. Sensor Update & Thresholds:
 *    - DHT_INTERVAL: Time (ms) between DHT sensor readings.
 *    - MATCH_THRESHOLD: Manhattan distance threshold (in pixels) to match a detected blob to an existing tracked object.
 *    - BOUNDARY: Row index in the 71x71 grid that acts as a crossing boundary.
 * 4. Persistent Object Detection:
 *    - OBJECT_STATIC_FRAME_THRESHOLD: Number of frames an object must remain static to be considered persistent.
 *      (Lowered from 100 to 20 for quicker detection on Arduino.)
 *    - MAX_MOVEMENT_THRESHOLD: Maximum allowed movement (in pixels) for an object to be considered static.
 *    - MIN_MOVEMENT_FOR_RESET: Movement threshold beyond which persistence is reset.
 *    - DILATION_PIXELS: Extra pixels added when computing the locked mask.
 *
 * This script performs computations analogous to the Python version:
 * - Interpolates an 8×8 thermal grid to 71×71.
 * - Extracts blob features via flood-fill, computes a Manhattan distance transform,
 *   and detects a central point.
 * - Tracks objects using centroid matching with a threshold of 25.
 * - Normalizes the image with a baseline (DHT temperature – 8).
 * - Detects persistent (static/hot) objects using similar thresholds as in the Python script. 
 * Ensure the required libraries (Wire, WiFi, Adafruit_AMG88xx, DHT, ArduinoJson) are installed.
 */

// ===================================================
// 1. Global Declarations and Sensor Configuration
// ===================================================

#include <Wire.h>
#include <WiFi.h>               // For ESP32 Wi-Fi
#include <Adafruit_AMG88xx.h>   // AMG8833 Grid-EYE library
#include <DHT.h>
#include <ArduinoJson.h>
#include <math.h>

// Wi-Fi Credentials
const char* ssid = "ManojithM31";
const char* password = "manojithtemp";

// Server Details
const char* serverIP = "192.168.41.224";  // Server IP
const int serverPort = 5000;              // Server Port

// AMG8833 (Grid-EYE) Sensor
Adafruit_AMG88xx amg;

// DHT Sensor
#define DHTPIN 5       // Adjust as needed
#define DHTTYPE DHT11  
DHT dht(DHTPIN, DHTTYPE);

// JSON and Wi-Fi Client
WiFiClient client;

// Global sensor values and timing
float globalDHTTemperature = 0;
float globalDHTHumidity = 0;
unsigned long lastDHTUpdateTime = 0;
const unsigned long DHT_INTERVAL = 10000;  // 10 seconds

// ===================================================
// 2. Type Definitions for Feature Extraction and Tracking
// ===================================================

// (2.1) Structure for blob features (from flood-fill)
struct BlobFeatures {
  int size;           // Number of pixels in the blob
  float centroidRow;  // Average row (sum/size)
  float centroidCol;  // Average column
  int linearIndex;    // Computed as round(centroidRow)*imageWidth + round(centroidCol)
  int minRow, maxRow; // Bounding box rows
  int minCol, maxCol; // Bounding box columns
};

// ===================================================
// 3. Multi-Object Tracking - HumanObject Class and Globals
// ===================================================

class HumanObject {
  public:
    float pos[2];         // Current centroid: [row, col]
    float first_pos[2];   // Initial position (for crossing)
    int size;             // Blob size (pixel count)
    int virtual_age;      // Frames without update
    bool updated;         // Flag set if updated in current frame
    static const int maxPath = 20;
    float path[maxPath][2];  // Path history
    int pathCount;
    // Persistent object tracking fields
    bool persistent;      // True if marked as persistent (hot object)
    float locked_centroid[2];  // Locked position when persistent
    uint8_t* locked_mask; // Computed binary mask for persistent object (71x71)
    
    // Constructor
    HumanObject(float row, float col, int size) {
      pos[0] = row; pos[1] = col;
      first_pos[0] = row; first_pos[1] = col;
      this->size = size;
      virtual_age = 0;
      updated = true;
      pathCount = 0;
      path[pathCount][0] = row;
      path[pathCount][1] = col;
      pathCount++;
      persistent = false;
      locked_mask = NULL;
    }
    
    // Update position and size.
    void update(float row, float col, int size) {
      pos[0] = row; pos[1] = col;
      this->size = size;
      if (pathCount < maxPath) {
        path[pathCount][0] = row;
        path[pathCount][1] = col;
        pathCount++;
      }
      virtual_age = 0;
      updated = true;
    }
    
    // Mark not updated.
    void virtualPropagate() {
      virtual_age++;
      updated = false;
    }
    
    // Check crossing the horizontal boundary.
    bool crossing(int boundary) {
      if ((first_pos[0] < boundary && pos[0] >= boundary) ||
          (first_pos[0] >= boundary && pos[0] < boundary))
        return true;
      return false;
    }
};

// Global tracking array and parameters
#define MAX_TRACKED 10
HumanObject* trackedObjects[MAX_TRACKED];
int trackedCount = 0;
const int MATCH_THRESHOLD = 25; // Matching threshold (Manhattan distance)
const int BOUNDARY = 35;        // Horizontal crossing boundary

// ===================================================
// 4. Persistent Object Mask Management Constants
// ===================================================
// Adjusted OBJECT_STATIC_FRAME_THRESHOLD lowered from 100 to 20 for quicker persistence.
const int OBJECT_STATIC_FRAME_THRESHOLD = 20;  // Number of frames (approx. 1 second at 50ms delay)
const float MAX_MOVEMENT_THRESHOLD = 0.8;        // Maximum allowed movement (in pixels)
const float MIN_MOVEMENT_FOR_RESET = 2.0;        // Movement threshold to reset persistence
const int DILATION_PIXELS = 2;                   // Dilation for locked mask computation

// ===================================================
// 5. Interpolation and Preprocessing Functions
// ===================================================

// (5.1) Upscale 8x8 to 71x71.
float* interpolate8to71(const float* input) {
  const int inSize = 8;
  const int outSize = 71;  // (8-1)*10 + 1
  float* output = new float[outSize * outSize];
  for (int i = 0; i < outSize * outSize; i++) output[i] = 0.0;
  
  // Horizontal interpolation
  for (int r = 0; r < inSize; r++) {
    for (int c = 0; c < inSize; c++) {
      output[r * 10 * outSize + c * 10] = input[r * inSize + c];
    }
    for (int c = 0; c < inSize - 1; c++) {
      float left = input[r * inSize + c];
      float right = input[r * inSize + c + 1];
      float diff = right - left;
      for (int newcol = 1; newcol < 10; newcol++) {
        output[r * 10 * outSize + (c * 10 + newcol)] = left + (diff * newcol / 10.0);
      }
    }
  }
  // Vertical interpolation
  for (int r = 0; r < inSize - 1; r++) {
    for (int c = 0; c < outSize; c++) {
      float up = output[r * 10 * outSize + c];
      float down = output[(r + 1) * 10 * outSize + c];
      float diff = down - up;
      for (int newrow = 1; newrow < 10; newrow++) {
        output[(r * 10 + newrow) * outSize + c] = up + (diff * newrow / 10.0);
      }
    }
  }
  return output;
}

// (5.2) Create binary mask using the threshold (current DHT temperature).
uint8_t* thresholdImage(const float* interpArray, int size, float threshold) {
  uint8_t* binaryMask = new uint8_t[size * size];
  for (int i = 0; i < size * size; i++) {
    binaryMask[i] = (interpArray[i] >= threshold) ? 1 : 0;
  }
  return binaryMask;
}

// ===================================================
// 6. Feature Extraction Functions
// ===================================================

// (6.1) Check if (r, c) is valid in an array.
bool isValid(int r, int c, int size) {
  return (r >= 0 && r < size && c >= 0 && c < size);
}

// (6.2) Flood-fill to compute blob features.
void floodFill(int startR, int startC, uint8_t* binaryMask, int size, bool* visited, BlobFeatures &blob) {
  const int maxStackSize = size * size;
  int* stackR = new int[maxStackSize];
  int* stackC = new int[maxStackSize];
  int stackIndex = 0;
  
  blob.size = 0;
  blob.centroidRow = 0;
  blob.centroidCol = 0;
  blob.minRow = startR;
  blob.maxRow = startR;
  blob.minCol = startC;
  blob.maxCol = startC;
  
  stackR[stackIndex] = startR;
  stackC[stackIndex] = startC;
  stackIndex++;
  visited[startR * size + startC] = true;
  
  while (stackIndex > 0) {
    stackIndex--;
    int r = stackR[stackIndex];
    int c = stackC[stackIndex];
    blob.size++;
    blob.centroidRow += r;
    blob.centroidCol += c;
    if (r < blob.minRow) blob.minRow = r;
    if (r > blob.maxRow) blob.maxRow = r;
    if (c < blob.minCol) blob.minCol = c;
    if (c > blob.maxCol) blob.maxCol = c;
    
    int dr[4] = {-1, 1, 0, 0};
    int dc[4] = {0, 0, -1, 1};
    for (int i = 0; i < 4; i++) {
      int nr = r + dr[i];
      int nc = c + dc[i];
      if (isValid(nr, nc, size)) {
        int index = nr * size + nc;
        if (binaryMask[index] == 1 && !visited[index]) {
          visited[index] = true;
          stackR[stackIndex] = nr;
          stackC[stackIndex] = nc;
          stackIndex++;
        }
      }
    }
  }
  delete[] stackR;
  delete[] stackC;
  
  blob.centroidRow /= blob.size;
  blob.centroidCol /= blob.size;
  blob.linearIndex = (int)(round(blob.centroidRow)) * size + (int)(round(blob.centroidCol));
}

// (6.3) Compute a naïve Manhattan distance transform for a blob's bounding box.
void compute_distance_transform(uint8_t* binaryMask, int imageSize, BlobFeatures blob, float* &dt, int &dtRows, int &dtCols) {
  dtRows = blob.maxRow - blob.minRow + 1;
  dtCols = blob.maxCol - blob.minCol + 1;
  int total = dtRows * dtCols;
  dt = new float[total];
  const int INF = 1000;
  
  for (int r = 0; r < dtRows; r++) {
    for (int c = 0; c < dtCols; c++) {
      int globalR = blob.minRow + r;
      int globalC = blob.minCol + c;
      int index = globalR * imageSize + globalC;
      if (binaryMask[index] == 1)
        dt[r * dtCols + c] = INF;
      else
        dt[r * dtCols + c] = 0;
    }
  }
  
  for (int r = 0; r < dtRows; r++) {
    for (int c = 0; c < dtCols; c++) {
      if (dt[r * dtCols + c] == INF) {
        int minDist = INF;
        for (int i = 0; i < dtRows; i++) {
          for (int j = 0; j < dtCols; j++) {
            if (dt[i * dtCols + j] == 0) {
              int dist = abs(r - i) + abs(c - j);
              if (dist < minDist) minDist = dist;
            }
          }
        }
        dt[r * dtCols + c] = minDist;
      }
    }
  }
}

// (6.4) Select the pixel with the maximum distance value (above threshold) as the central point.
void detect_central_points(float* dt, int dtRows, int dtCols, float valueThreshold, int countThreshold, int &centralRow, int &centralCol, float &centralValue) {
  centralValue = -1;
  centralRow = -1;
  centralCol = -1;
  for (int r = 0; r < dtRows; r++) {
    for (int c = 0; c < dtCols; c++) {
      float val = dt[r * dtCols + c];
      if (val > valueThreshold && val > centralValue) {
        centralValue = val;
        centralRow = r;
        centralCol = c;
      }
    }
  }
}

// (6.5) Process the binary mask to extract blob features for tracking.
#define MAX_BLOBS 20
BlobFeatures detectedBlobs[MAX_BLOBS];
int detectedBlobCount = 0;

void processBlobsForTracking(uint8_t* binaryMask, int size) {
  bool* visited = new bool[size * size];
  for (int i = 0; i < size * size; i++) visited[i] = false;
  detectedBlobCount = 0;
  for (int r = 0; r < size; r++) {
    for (int c = 0; c < size; c++) {
      int index = r * size + c;
      if (binaryMask[index] == 1 && !visited[index]) {
        BlobFeatures blob;
        floodFill(r, c, binaryMask, size, visited, blob);
        if (detectedBlobCount < MAX_BLOBS) {
          detectedBlobs[detectedBlobCount] = blob;
          detectedBlobCount++;
        }
      }
    }
  }
  delete[] visited;
}

// ===================================================
// 7. Normalization Function
// ===================================================

// Normalize a 71x71 image using baseline (globalDHTTemperature - 8).
uint8_t* normalize_with_baseline(const float* img, int size) {
  uint8_t* normImg = new uint8_t[size * size];
  float baseline = globalDHTTemperature - 8;
  float max_val = baseline;
  for (int i = 0; i < size * size; i++) {
    if (img[i] > max_val) max_val = img[i];
  }
  for (int i = 0; i < size * size; i++) {
    if (max_val == baseline) {
      normImg[i] = 0;
    } else {
      float val = ((img[i] - baseline) / (max_val - baseline)) * 255.0;
      if (val < 0) val = 0;
      if (val > 255) val = 255;
      normImg[i] = (uint8_t)val;
    }
  }
  return normImg;
}

// ===================================================
// 8. Persistent Object Mask Management Functions
// ===================================================

// (8.1) Compute a circular binary mask (71x71) centered at the given centroid.
uint8_t* compute_locked_mask(float centroidRow, float centroidCol, int dilation_pixels, int blobSize) {
  const int outSize = 71;
  uint8_t* mask = new uint8_t[outSize * outSize];
  for (int i = 0; i < outSize * outSize; i++) mask[i] = 0;
  float radius = sqrt((float)blobSize / 3.14159) + dilation_pixels;
  for (int r = 0; r < outSize; r++) {
    for (int c = 0; c < outSize; c++) {
      float d = sqrt((r - centroidRow) * (r - centroidRow) + (c - centroidCol) * (c - centroidCol));
      if (d <= radius) mask[r * outSize + c] = 1;
    }
  }
  return mask;
}

// (8.2) Update each tracked object to mark it as persistent if static, or reset if movement is detected.
void update_persistent_objects() {
  for (int i = 0; i < trackedCount; i++) {
    HumanObject* obj = trackedObjects[i];
    if (!obj->persistent) {
      if (obj->pathCount >= OBJECT_STATIC_FRAME_THRESHOLD) {
        float dx = obj->pos[0] - obj->first_pos[0];
        float dy = obj->pos[1] - obj->first_pos[1];
        float dist = sqrt(dx * dx + dy * dy);
        if (dist < MAX_MOVEMENT_THRESHOLD) {
          obj->persistent = true;
          obj->locked_centroid[0] = obj->pos[0];
          obj->locked_centroid[1] = obj->pos[1];
          obj->locked_mask = compute_locked_mask(obj->pos[0], obj->pos[1], DILATION_PIXELS, obj->size);
        }
      }
    } else {
      float dx = obj->pos[0] - obj->locked_centroid[0];
      float dy = obj->pos[1] - obj->locked_centroid[1];
      float dist = sqrt(dx * dx + dy * dy);
      if (dist > MIN_MOVEMENT_FOR_RESET) {
        obj->persistent = false;
        if (obj->locked_mask != NULL) {
          delete[] obj->locked_mask;
          obj->locked_mask = NULL;
        }
      }
    }
  }
}

// (8.3) Combine locked masks from all persistent objects into one 71x71 binary mask.
uint8_t* generate_combined_persistent_mask() {
  const int outSize = 71;
  uint8_t* combined = new uint8_t[outSize * outSize];
  for (int i = 0; i < outSize * outSize; i++) combined[i] = 0;
  for (int i = 0; i < trackedCount; i++) {
    HumanObject* obj = trackedObjects[i];
    if (obj->persistent && obj->locked_mask != NULL) {
      for (int j = 0; j < outSize * outSize; j++) {
        combined[j] = combined[j] | obj->locked_mask[j];
      }
    }
  }
  return combined;
}

// ===================================================
// 9. Wi-Fi and Data Transmission Functions
// ===================================================

// (9.1) Connect to Wi-Fi.
void connectToWiFi() {
  Serial.print("Connecting to WiFi: ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWi-Fi Connected!");
}

// (9.2) Send sensor data to the server via HTTP POST.
void sendDataToServer(float* gridData, float dhtTemp, float dhtHum) {
  if (!client.connected()) {
    Serial.println("Connecting to server...");
    if (!client.connect(serverIP, serverPort)) {
      Serial.println("Server Connection Failed!");
      return;
    }
  }
  StaticJsonDocument<1024> doc;
  JsonArray gridArray = doc.createNestedArray("gridEye");
  for (int i = 0; i < 64; i++) {
    gridArray.add(gridData[i]);
  }
  JsonObject dhtObj = doc.createNestedObject("dht");
  dhtObj["temperature"] = dhtTemp;
  dhtObj["humidity"] = dhtHum;
  String jsonOutput;
  serializeJson(doc, jsonOutput);
  String httpRequest = "POST /data HTTP/1.1\r\n";
  httpRequest += "Host: " + String(serverIP) + "\r\n";
  httpRequest += "Content-Type: application/json\r\n";
  httpRequest += "Content-Length: " + String(jsonOutput.length()) + "\r\n";
  httpRequest += "Connection: keep-alive\r\n\r\n";
  httpRequest += jsonOutput + "\r\n";
  client.print(httpRequest);
  Serial.println("Data Sent to Server!");
}

// ===================================================
// 10. setup() and loop()
// ===================================================

void setup() {
  Serial.begin(115200);
  Wire.begin();
  if (!amg.begin()) {
    Serial.println("AMG8833 Sensor Initialization Failed!");
    while (1);
  }
  delay(100);
  dht.begin();
  connectToWiFi();
  pinMode(2, OUTPUT);

}

void loop() {
  // 10.1 Update DHT sensor every DHT_INTERVAL.
  if (millis() - lastDHTUpdateTime >= DHT_INTERVAL) {
    float temp = dht.readTemperature();
    float hum = dht.readHumidity();
    if (!isnan(temp) && !isnan(hum)) {
      // Add a 0.1 offset to the DHT temperature.
      globalDHTTemperature = temp + 0.1;
      globalDHTHumidity = hum;
      Serial.println("DHT sensor updated.");
    } else {
      Serial.println("DHT11 Read Failed!");
    }
    lastDHTUpdateTime = millis();
  }
  
  // 10.2 Read AMG8833 sensor data (8x8 grid).
  float gridData[64];
  amg.readPixels(gridData);
  
  // 10.3 Transmit sensor data to server.
  sendDataToServer(gridData, globalDHTTemperature, globalDHTHumidity);
  
  // 10.4 Preprocessing: Interpolate and threshold the 8x8 grid.
  float* interpArray = interpolate8to71(gridData);
  uint8_t* binaryMask = thresholdImage(interpArray, 71, globalDHTTemperature);
  
  // 10.5 Feature Extraction: Process binary mask for tracking.
  processBlobsForTracking(binaryMask, 71);
  
  // 10.6 Multi-Object Tracking:
  for (int i = 0; i < trackedCount; i++) {
    trackedObjects[i]->updated = false;
  }
  
  for (int i = 0; i < detectedBlobCount; i++) {
    float blobRow = detectedBlobs[i].centroidRow;
    float blobCol = detectedBlobs[i].centroidCol;
    int blobSize = detectedBlobs[i].size;
    bool matched = false;
    for (int j = 0; j < trackedCount; j++) {
      float d = fabs(blobRow - trackedObjects[j]->pos[0]) + fabs(blobCol - trackedObjects[j]->pos[1]);
      if (d < MATCH_THRESHOLD) {
        trackedObjects[j]->update(blobRow, blobCol, blobSize);
        matched = true;
        if (trackedObjects[j]->crossing(BOUNDARY)) {
          trackedObjects[j]->first_pos[0] = trackedObjects[j]->pos[0];
          trackedObjects[j]->first_pos[1] = trackedObjects[j]->pos[1];
        }
        break;
      }
    }
    if (!matched && trackedCount < MAX_TRACKED) {
      trackedObjects[trackedCount] = new HumanObject(blobRow, blobCol, blobSize);
      trackedCount++;
    }
  }
  
  // Remove objects not updated recently.
  for (int i = 0; i < trackedCount; ) {
    if (!trackedObjects[i]->updated) {
      trackedObjects[i]->virtualPropagate();
      if (trackedObjects[i]->virtual_age >= 3) {
        delete trackedObjects[i];
        for (int j = i; j < trackedCount - 1; j++) {
          trackedObjects[j] = trackedObjects[j + 1];
        }
        trackedCount--;
        continue;
      }
    }
    i++;
  }
  
  // 10.7 Persistent Object Mask Management.
  update_persistent_objects();
  uint8_t* persistentMask = generate_combined_persistent_mask();
  delete[] persistentMask;
  
  // 10.8 Normalization (if needed for further processing)
  uint8_t* normImg = normalize_with_baseline(interpArray, 71);
  delete[] normImg;
  
  // 10.9 Final Output: Print only people count and persistent object count.
  int persistentCount = 0;
  for (int i = 0; i < trackedCount; i++) {
    if (trackedObjects[i]->persistent) persistentCount++;
  }
  Serial.print("Current people in room count: ");
  Serial.println(trackedCount);
  Serial.print("Current number of objects count: ");
  Serial.println(persistentCount);
  
  // Free memory.
  delete[] interpArray;
  delete[] binaryMask;

  if (trackedCount > 0) {
      digitalWrite(2, HIGH);  // Turn LED ON if at least one person detected
  } else {
      digitalWrite(2, LOW);   // Turn LED OFF if no person detected
  }


  
  delay(50);
}
