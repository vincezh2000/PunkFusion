# PunkFusion 

### Project Introduction

PunkFusion is a groundbreaking demo that combines the distinctive aesthetics of CryptoPunks with the power of Ethereum and the latest in generative AI technology. Drawing inspiration from the original collection of 10,000 algorithmically generated 24x24 pixel art characters, PunkFusion explores new frontiers at the intersection of blockchain technology and AI-driven creativity.

Each CryptoPunk is not only a unique digital artwork but also a symbol of the innovation brought about by Ethereum’s smart contracts, which enable trustless ownership and exchange of these assets. By integrating the decentralized nature of Ethereum with state-of-the-art generative AI, PunkFusion showcases how the future of digital art can evolve, creating new and unique assets that resonate with the NFT and blockchain communities.

Our goal is to push the boundaries of NFT creation and generative modeling, demonstrating how these two technologies can be fused to create unique and innovative digital assets.

### Project Overview

PunkFusion is based on a custom-built **Diffusion model**, specifically designed to generate CryptoPunk-style assets. Our primary contributions are twofold:

1. **Data Collection**: We developed a comprehensive method for collecting and curating high-quality data related to the CryptoPunks collection. This includes automated crawlers and data processing scripts, which we have made publicly available. Additionally, we offer a ready-to-use dataset to facilitate further research and experimentation.
  
2. **Model Development**: From scratch, we built a Diffusion model tailored to CryptoPunk-style image generation. This model has been trained using our curated dataset and is designed to reproduce the distinctive characteristics of CryptoPunks while allowing for creative variations. We have also released the model’s weights as open-source, enabling others to build on our work and further explore generative NFT art.

Below are some new creations from our model:

<img width="521" alt="image" src="https://github.com/user-attachments/assets/9e280f49-4fb8-4a7f-83d5-dcc19ccbb900">


### Installation

1. Create and activate the Conda environment:

   ```bash
   conda create -n punk python=3.11
   conda activate punk

2. Install PyTorch and the required dependencies. The experiment was tested with CUDA 12.1 and PyTorch 2.4.0:

   ```bash
   
   pip3 install torch torchvision torchaudio # if you have CUDA 12.1 and a linux machine
   pip install -r requirements.txt

3. For data collection, run the crawler script. Please note that it may take around 15 hours to complete due to the relay setup, which ensures our crawler avoids detection:

   ```bash
   python crawler.py

4. Alternatively, you can download the dataset from Hugging Face or ModelScope (recommended if you are in mainland China).

   
   Here is the download link: [Insert Link Here]

6. Once you have the dataset, start training the model:

   ```bash
   python train_punk.py

Based on our experiments, with the default settings, you can achieve stable results after approximately 200 epochs.

### To-Do List:
- [x] Add Exponential Moving Average (EMA)
- [ ] Integrate Multi-Modality
- [ ] Implement Classifier Guidance and CFG (Class-Factorized Guidance)
- [ ] Incorporate Huber Loss
- [ ] Add new architechture-DiT (Diffusion Transformer)
- [ ] Integrate RoPE (Rotary Position Embedding)
- [ ] Implement Flow Matching


### Disclaimer

This project is for educational purposes only. All NFT assets referenced or utilized in this project are sourced from [CryptoPunks](https://cryptopunks.app/) and are the property of their respective owners.
