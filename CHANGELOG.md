# 8.02.2026

### Done: 
    Download 3 dataset
    1. Bening-Monday-no-metadata.parquet
    2. Botnet-Friday-no-metadata.parquet
    3. Infiltration-Thursday-no-metadata.parquet
### TODO: 
    1. Read what is parquet files
    learn how use this files in python
        - how read this files
        - how see this files

# 10.03.2026

### Done:
    1. Learn how use, read, write, take info about parquet files
    2. Concat 3 files in one, and normalize colums in result file
        - 16 of 78 lines remaining
            -  Flow IAT Mean             
            -  Flow IAT Std              
            -  Flow IAT Max              
            -  Flow IAT Min              
            -  Total Fwd Packets         
            -  Total Backward Packets    
            -  Fwd Packets Length Total  
            -  Bwd Packets Length Total  
            -  Packet Length Min         
            -  Packet Length Max         
            -  Packet Length Mean        
            -  Packet Length Std         
            -  Packet Length Variance    
            -  Flow Bytes/s              
            -  Flow Packets/s            
            -  Label  
            -  y

### TODO: 
    1. Clean result dataset(delete rows with emty columns)

# 11.03.2026
### Done:
    1. Clean result dataset(delete rows with emty columns)
    2. Fix cancatination of datasets
    3. Prepare datasets to train dataset(80%) and test(datasets)

### TODO:
    1. Normalize data
    2. Solution problem of disbalance( SMOTE )
    3. Learn ML
        Algoritms:
        - RandomForest
        - LightGBM

# 12.03.2026
### Done:
    1. Added normalization scallers function: min_max_scaler(), standard_scaler()
    2. Add new function for print graphics:
        - Take the logarithm of values that are included in the graph for clarity
        - Use matplotlib library for the image
    3. Small fixes for datasets(they where having negative values in rows):
        - deleted

### TODO:
    1. Use SMOTE algorithm with a ready-made dataset
    2. ML training

# 13.03.2026

### Done: 
    1. Use SMOTE algorithm
    2. Make script for full automatic preparing datasets
    3. Code refactoring: 
        - all code for preparing in a functions

### TODO: 
    1. ML training

# 14.03.2026

### Done:
    1. Trained and evaluated ML models: Random Forest and LightGBM on Z-score normalized data
    2. Implemented Threshold Tuning (set threshold to 0.90) to solve the False Positives problem and increase Precision
    3. Created a universal function train_and_evaluate_model() for training, predicting, and metric calculation
    4. Added automatic generation and saving of Confusion Matrix graphics (heatmaps) using seaborn and matplotlib
    5. Refactored ML training scripts (added English docstrings, clean prints, and modular structure)
    6. Created and executed a parallel training pipeline for Min-Max normalized data (`train_min_max.py`)

### TODO: 
    1. Analyze and compare the performance (Accuracy, Precision, Recall, Time) between Z-score and Min-Max results
    2. Start writing Chapter 5 (Vyhodnotenie / Evaluation) in the thesis using the collected metrics and saved confusion matrices
    3. Formulate the final conclusion about which model and preprocessing method is best for Real-time IDS

# 15.03.2026

### Done: 
    1. Saved trained ML models (Random Forest and LightGBM) to disk using `joblib` for future deployment.
    2. Designed and implemented the Server-side architecture (IDS Engine) using `FastAPI`.
    3. Created a Pydantic data model (`NetworkPacket`) to handle incoming network traffic data (15 features).
    4. Implemented a `POST /analyze` API endpoint that processes data through both models and returns raw Trojan probabilities.
    5. Tested the server successfully using FastAPI's built-in Swagger UI.
    6. Applied the "Separation of Concerns" principle: Server acts as a "dumb oracle" returning probabilities, moving the threshold/blocking logic to the client side.
### TODO:
    1. Finish writing the practical part (Chapter 5 / Vyhodnotenie) of the thesis.
    2. Implement the Client (Sensor) script to simulate real-time network traffic using test `.parquet` datasets.
    3. Add the threshold-based blocking logic to the Client script and test the full Server-Client interaction.

