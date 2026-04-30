**Project: Depth Estimation and Object Distance Measurement**

You will work in teams of two to implement and evaluate both classical and learning-based approaches for depth estimation, and combine them with object detection to estimate real-world distances.

---

### **Objective**

Estimate the distance of objects from a camera by integrating:

* Depth estimation (classical + ML-based)
* Object detection
* Distance inference techniques

Analyze and compare the results across methods.

---

### **Subtask 1: Depth Estimation**

Implement and compare:

* **One classical method** (e.g., stereo matching or structure-from-motion)
* **One ML-based model** (e.g., MiDaS, DPT)

**Output:**

* A depth map (heatmap visualization)

  * Warm colors → closer objects
  * Cool colors → farther objects

---

### **Subtask 2: Object Distance Estimation**

1. Apply **object detection** using a pre-trained model (e.g., YOLO, Faster R-CNN)
2. Estimate the **distance (in meters)** to each detected object using:

   * Depth map values, and/or
   * Heuristic or learned approaches (e.g., known object sizes, perspective geometry)

**Output:**

* Image with bounding boxes labeled as:
  `"object_name: distance (m)"`
  *(e.g., "person: 5.2 m")*

---

### **Analysis Requirements**

* Compare classical vs ML-based depth estimation
* Discuss accuracy, limitations, and failure cases
* Evaluate how depth quality affects distance estimation

---

### **Deliverables**

* Code implementation
* Visual results (depth maps + annotated detections)
* Short report with analysis
