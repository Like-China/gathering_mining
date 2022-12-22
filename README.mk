This code is the implementation of ¡°Relaxed Group Pattern Detection over Massive-Scale Trajectories¡±
# datasets
Beijing https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/
Porto https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data
# indexing
The indexing includes the R-tree for pre-checjing.
# baselines
The baseline includes the ES-ECMC for single threading, and ES-ECMC-multi for multiple-threading setting.
In addition, our method LCS_SCAN is included.
# utils
Trjaectory is the base util for data store and processing.
# loader
Data_loader is used to load trajectory data from files.

For usage, open the settings.py for parameter settings and run mainpy for implementation.