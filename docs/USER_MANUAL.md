# User Manual — Medical Image Classification System

**For non-technical users. No coding required.**

---

## What Does This System Do?

This system is an AI-powered medical screening tool that can:

1. **Detect Pneumonia** — Upload a chest X-ray image and the system will tell you if pneumonia is likely present.
2. **Classify Brain Tumors** — Upload a brain MRI scan and the system will identify whether a tumor is present, and if so, what type.

> ⚠️ **Important**: This tool is for research and educational purposes only. It is **not** a medical diagnosis. Always consult a qualified physician.

---

## Getting Started

### Step 1 — Open the Web Interface

Open your browser and go to:
```
http://localhost
```
or
```
http://localhost:3000
```

You will see the Medical AI Dashboard with four tabs at the top.

---

## The Four Tabs

### Tab 1 — 🔮 Predict (Main Feature)

This is where you upload images and get predictions.

**How to get a prediction:**

1. Click the **"🔮 Predict"** tab (it opens by default)
2. Choose the type of image you are uploading:
   - **Pneumonia Model** → for chest X-ray images
   - **Brain Tumor Model** → for brain MRI images
3. Click **"Choose File"** and select your image from your computer
4. Click the **"Analyze Image"** button
5. Wait 1–3 seconds for the result to appear

**Understanding the result:**

| Result | What it means |
|---|---|
| `PNEUMONIA` with high confidence | The AI detected signs of pneumonia in the X-ray |
| `NORMAL` with high confidence | No signs of pneumonia detected |
| `glioma / meningioma / pituitary` | A specific type of brain tumor was identified |
| `notumor` | No tumor detected in the MRI |
| "low confidence" message | The AI is not sure — the image may be unclear or the wrong type |
| "input is not a chest X-ray" | You may have uploaded a brain MRI to the Pneumonia model by mistake |

**Tips for best results:**
- Use clear, properly oriented medical images
- Chest X-rays should be anterior-posterior (front-facing)
- MRI images should be clear, single-slice images
- Accepted file formats: JPEG, PNG, BMP, TIFF
- If you get a "low confidence" result, try a clearer image

---

### Tab 2 — 📊 Pipeline

Shows the status of all 13 stages of the AI training pipeline.

- **Green stages** = completed successfully
- Each stage builds on the previous one, from raw data collection through to model training and monitoring
- The metrics shown (accuracy, F1-score) reflect the current trained models

You do not need to interact with this tab during normal use.

---

### Tab 3 — 📈 Monitoring

Shows real-time system health information:

| Section | What it shows |
|---|---|
| **API Status** | Whether the prediction service is running |
| **Models Loaded** | Whether both AI models are ready |
| **Inference Latency** | How fast predictions are returned |
| **Memory / CPU** | System resource usage |
| **Data Drift** | Whether the AI is still performing reliably |

A green **HEALTHY** status means everything is working normally.

Click **"Refresh Status"** to get the latest information from the server.

---

### Tab 4 — 📚 User Manual

The tab you are reading now. Contains instructions for using the system.

---

## Frequently Asked Questions

**Q: How long does a prediction take?**
A: Usually 1–3 seconds. The first prediction after startup may take slightly longer.

**Q: What image file formats are accepted?**
A: JPEG (.jpg, .jpeg), PNG (.png), BMP (.bmp), and TIFF (.tif, .tiff).

**Q: I uploaded a chest X-ray to the Brain Tumor model by mistake. What happens?**
A: The system will detect the mismatch and return a message saying "input is not a brain MRI" with 0% confidence. Simply switch the model type and try again.

**Q: What does "low confidence" mean?**
A: The AI is less than 70% certain about its prediction. This can happen with unclear images, unusual image angles, or images that are significantly different from what the AI was trained on.

**Q: Is my image stored anywhere?**
A: No. Uploaded images are deleted immediately after the prediction is made.

**Q: The page shows an error or the prediction button does nothing.**
A: The API server may not be running. Check with your system administrator that the service is started. If you are running it yourself, see the RUNBOOK for startup instructions.

**Q: Can I use this for actual medical decisions?**
A: No. This system has not been clinically validated and must not be used as a substitute for professional medical diagnosis.

---

## Quick Reference — Model Outputs

### Pneumonia Model
- **NORMAL** — No pneumonia detected
- **PNEUMONIA** — Pneumonia detected

### Brain Tumor Model
- **glioma** — Glioma tumor detected
- **meningioma** — Meningioma tumor detected
- **notumor** — No tumor detected
- **pituitary** — Pituitary tumor detected

---

## Technical Support

If the system is not working:
1. Check that both the API server and frontend are running
2. Refer to the [RUNBOOK](RUNBOOK.md) for startup instructions
3. Check system health at `http://localhost:8001/health`
4. Contact your system administrator
