# Anaemia-Detection-Application

A Flask web application that provides non-invasive anaemia screening by analysing palm and nail images alongside basic demographic inputs.

🚀 **Live Demo:** [https://web-production-e16a5.up.railway.app/](https://web-production-e16a5.up.railway.app/)

> ⚠️  This is a research prototype trained on a limited dataset of 527 subjects aged 18–25 from Peru. It is intended as a preliminary screening tool only and should not replace clinical diagnosis.

## What does it do?

Upload a palm image and a nail image, enter your age and gender, and the application will:

1. **Classify** whether the individual is anaemic or non-anaemic
2. **Estimate** the haemoglobin (Hb) level in g/dL
3. **Provide** a WHO severity classification (Mild / Moderate / Severe / Non-Anaemic)
4. **Display** a confidence score for the classification
5. **Visualise** spatial attention maps showing which regions of the palm and nail the model focused on when making the prediction

## Project Structure
```
Anaemia-Detection-Application/
│
├── app.py                    # Flask application, prediction logic and attention map generation
├── requirements.txt          # Python dependencies
├── Procfile                  # Railway deployment configuration
│
├── model/
│   └── CNN_Model_4.keras     # Trained CNN4 joint model weights
│
├── templates/
│   └── index.html            # Frontend HTML
│
└── static/
    ├── style.css             # Styling
    └── script.js             # Frontend prediction and display logic
```



## Related Repository

The research and training pipeline behind this application is maintained separately.

Research: [Non-Invasive-Iron-Deficiency-Anaemia-Prediction-using-Multimodal-Fusion](https://github.com/MadhunishaBala/Non-Invasive-Iron-Deficiency-Anaemia-Prediction-using-Multimodal-Fusion)
