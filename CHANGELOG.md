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