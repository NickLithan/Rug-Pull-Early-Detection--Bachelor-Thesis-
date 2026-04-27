# Can Market Microstructure Metrics Enhance Rug Pull Detection?

### *Bachelor Thesis*

****

## Abstract

While rug pulls are among the biggest threats in cryptocurrency markets, only a few of studies have attempted to build early detection models for malicious tokens. I conducted a comprehensive rug pull prediction study on Solana’s Raydium DEX, testing whether market microstructure measures can enhance identification of malicious coins.

With 32,426 token launches analyzed, I conducted an A/B test on model performance before and after adding 8 features, including Kyle’s lambda, Amihud's illiquidity, VPIN and others. The analysis suggests that these features can lead to an incremental, yet statistically significant improvement in performance, partly consistent with microstructure theory. The robustness of these results was validated across multiple model architectures, hyperparameter configurations and labeling strategies.

![image](visuals/shap_target1/shap_target1_scatter_cb.png)

## Repository Structure

    BT
    ├── data                    # data collected from outside sources + the final dataset
    ├── data_collection         # SQL and Python scripts for data collection
    ├── data_engineering        # raw data processing and feature/target calculations
    ├── model_storage           # saves of trained models, split by configuration type
    ├── utils                   # utility funcions, used in the main pipeline
    ├── visuals                 # saved chart images
    ├── LICENSE
    ├── README.md
    └── main.ipynb              # Jupyter notebook with the core study pipeline

