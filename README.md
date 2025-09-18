Summer Internship 2025: Machine Vision and Intelligent Systems Lab,Nust Repository
=============================================================================

Overview
--------

Welcome to the repository for my **Summer Internship 2025** at the **Machine Vision and Intelligent Systems Lab**. This collection captures my hands-on journey through foundational and advanced deep learning concepts, culminating in a practical project on **Image Caption Generation**.

During the 12-week internship, I explored neural networks, data pipelines, convolutional architectures, recurrent models, and transformer-based systems. The work spans from basic gradient descent implementations to state-of-the-art multimodal models using EfficientNet and beam search for captioning. All code is implemented in **PyTorch** (with some TensorFlow/Keras for data loading) and Jupyter Notebooks for reproducibility.

This repository serves as a portfolio of my learning outcomes, including:

*   **9 Core Notebook Implementations**: Progressive exercises building from linear models to complex vision-language tasks.
    

The internship emphasized practical ML engineering: data preprocessing, model training, evaluation, and deployment insights. Key themes include ethical AI, computational efficiency, and bridging vision with language.

**Repository Structure**:

text

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   internship-2025/  ├── notebooks/  │   ├── 1.Batch_gradient_descent.ipynb          # Basic optimization  │   ├── 2.internship_neural_network_Practice_of_pipeline.ipynb  # NN pipeline practice  │   ├── 3.Internship_Convolution_Neural_Network_exercise1.ipynb # CNN on MNIST  │   ├── 4.Internship_Convolution_Neural_Network_modified_exercise2.ipynb # CNN on CIFAR-10  │   ├── 5.VGG16_Internship_Convolution_Neural_Network_modified.ipynb # VGG16 on CIFAR-10  │   ├── 6.Sentiment_Analysis_by_using_lstm.ipynb # LSTM for text sentiment  │   ├── 7.Imdb_review_predictor.ipynb            # RNN predictor on IMDB  │   ├── 8.Transformer_from_the_scratch.ipynb     # Custom Transformer implementation  │   └── Project_image-to-caption-efficientnet-b2-version-beam-1.ipynb # Capstone: Image Captioning  ├── reports/  │   └── Machine_Vision_report.pdf                #                                       # Sample datasets (e.g., MNIST, CIFAR-10 subsets)  ├── requirements.txt                             # Dependencies  └── README.md                                    # This file   `

Quick Start
-----------

### Prerequisites

*   Python 3.8+
    
*   PyTorch 2.0+ (with CUDA for GPU acceleration)
    
*   Jupyter Lab/Notebook
    
*   Additional libraries: NumPy, Pandas, Matplotlib, Scikit-learn, TensorFlow (for data loading), Seaborn, NLTK (for text processing)
    

### Running the Notebooks

1.  Clone the repo: git clone
    
2.  Navigate to notebooks/
    
3.  Launch Jupyter: jupyter lab
    
4.  Open any .ipynb file and run cells sequentially. Most notebooks include:
    
    *   Data loading/preprocessing
        
    *   Model definition and training
        
    *   Evaluation (e.g., accuracy, loss plots, confusion matrices)
        
    *   Visualizations (e.g., sample predictions)
        

**Hardware Recommendation**: Use a GPU (e.g., Google Colab T4 or local NVIDIA) for CNN/Transformer training to avoid long runtimes.

Notebook Summaries
------------------

Each notebook builds progressively, aligning with the internship's curriculum from foundational ML to advanced vision-language tasks.

### 1\. Batch Gradient Descent (1.Batch\_gradient\_descent.ipynb)

*   **Focus**: Implement batch gradient descent from scratch for linear regression.
    
*   **Key Concepts**: Cost function minimization, parameter updates (weights w and bias b), epochs, learning rate tuning.
    
*   **Dataset**: Synthetic 1D data (50 samples).
    
*   **Outcomes**: Achieves low MSE loss; visualizes convergence. Serves as a baseline for understanding optimization in neural networks.
    
*   **Runtime**: <1 minute on CPU.
    

### 2\. Neural Network Pipeline Practice (2.internship\_neural\_network\_Practice\_of\_pipeline.ipynb)

*   **Focus**: End-to-end binary classification pipeline using a simple feedforward NN.
    
*   **Key Concepts**: Data loading (CSV), normalization, train-test split, PyTorch DataLoader, binary cross-entropy loss, Adam optimizer.
    
*   **Dataset**: Anthropometric measurements (gender prediction; 10k+ samples).
    
*   **Outcomes**: Train accuracy ~50.9%, test accuracy ~53.2% after 50 epochs. Plots loss/accuracy curves; highlights class imbalance issues.
    
*   **Runtime**: ~2 minutes on CPU.
    

### 3\. CNN Exercise 1 on MNIST (3.Internship\_Convolution\_Neural\_Network\_exercise1.ipynb)

*   **Focus**: Basic CNN for digit classification.
    
*   **Key Concepts**: Conv2D layers, pooling, flattening, softmax output; class-balanced sampling.
    
*   **Dataset**: MNIST (6k train, 1k test images).
    
*   **Outcomes**: High accuracy (~95%+); confusion matrix visualization. Explores filter effects on edge detection.
    
*   **Runtime**: ~5 minutes on GPU.
    

### 4\. Modified CNN Exercise 2 on CIFAR-10 (4.Internship\_Convolution\_Neural\_Network\_modified\_exercise2.ipynb)

*   **Focus**: Enhanced CNN with multiple conv blocks for multi-class image classification.
    
*   **Key Concepts**: Data augmentation, dropout, batch normalization; evaluation with confusion matrices.
    
*   **Dataset**: CIFAR-10 subset (3k train, 9k test; 10 classes like airplane, cat).
    
*   **Outcomes**: ~70% test accuracy; seaborn heatmaps for misclassifications (e.g., dog vs. cat confusion).
    
*   **Runtime**: ~10 minutes on GPU.
    

### 5\. VGG16-Modified on CIFAR-10 (5.VGG16\_Internship\_Convolution\_Neural\_Network\_modified.ipynb)

*   **Focus**: Transfer learning with a modified VGG16 architecture.
    
*   **Key Concepts**: Pre-trained weights adaptation, deeper conv stacks, global average pooling.
    
*   **Dataset**: CIFAR-10 (3k train, 900 test).
    
*   **Outcomes**: Improved accuracy (~75%+); loss/accuracy plots show faster convergence vs. scratch training.
    
*   **Runtime**: ~15 minutes on GPU.
    

### 6\. LSTM for Sentiment Analysis (6.Sentiment\_Analysis\_by\_using\_lstm.ipynb)

*   **Focus**: Sequence modeling for binary sentiment classification.
    
*   **Key Concepts**: Embedding layers, bidirectional LSTM, dropout for regularization; padding/truncation.
    
*   **Dataset**: IMDB reviews (25k train/test; positive/negative labels).
    
*   **Outcomes**: ~85% accuracy after 50 epochs; confusion matrix and ROC curves. Demonstrates handling variable-length text.
    
*   **Runtime**: ~20 minutes on GPU.
    

### 7\. IMDB Review Predictor (7.Imdb\_review\_predictor.ipynb)

*   **Focus**: RNN-based predictor using Keras IMDB dataset.
    
*   **Key Concepts**: Tokenization, padding to fixed length (200), embedding matrix; integrates TensorFlow with PyTorch eval.
    
*   **Dataset**: IMDB (5k train, 1k test; top 10k words).
    
*   **Outcomes**: ~88% accuracy; visualizes review length distribution and prediction samples.
    
*   **Runtime**: ~8 minutes on CPU/GPU.
    

### 8\. Transformer from Scratch (8.Transformer\_from\_the\_scratch.ipynb)

*   **Focus**: Full implementation of the Transformer architecture (encoder-decoder).
    
*   **Key Concepts**: Multi-head attention, positional encoding, layer normalization, feed-forward blocks, masking.
    
*   **Dataset**: Synthetic sequences for seq2seq tasks.
    
*   **Outcomes**: Modular build function; initializes Xavier uniforms. Prepares for translation/NMT applications.
    
*   **Runtime**: ~5 minutes for forward pass tests.
    

### 9\. Capstone: Image-to-Caption with EfficientNet-B2 & Beam Search (Project\_image-to-caption-efficientnet-b2-version-beam-1.ipynb)

*   **Focus**: Multimodal project generating descriptive captions for images.
    
*   **Key Concepts**: CNN-RNN fusion (EfficientNet-B2 encoder + LSTM decoder), beam search decoding, BLEU score evaluation.
    
*   **Dataset**: MS COCO (subsets for train/val/test); handles image preprocessing (resize, normalize).
    
*   **Outcomes**: Generates coherent captions (e.g., "A dog chasing a frisbee in the grass"); sample grid visualizations with predicted vs. ground-truth captions. Achieves BLEU-4 ~0.25.
    
*   **Runtime**: ~30 minutes per epoch on GPU (Kaggle setup with PyDrive for data).
    

Reflections & Future Work
-------------------------

This internship transformed my theoretical knowledge into deployable skills. Challenges like dataset imbalances and computational bottlenecks taught resilience. Moving forward, I'd integrate attention in Transformers for better captioning and explore diffusion models for generative tasks.

**Contributions Welcome**: Fork, star, or open issues for discussions!

License & Contact
-----------------

*   **License**: MIT (feel free to use/adapt).
    
*   **Author**: \[Mursal bajwa\] – \[mursal.bajwa786@gmail.com \] | LinkedIn: \[[https://www.linkedin.com/in/mursal-bajwa-a0ab5030b](https://www.linkedin.com/in/mursal-bajwa-a0ab5030b)\] | GitHub: \[Mursal Bajwa\]
    

_Last Updated: September 18, 2025Built with ❤️ during Summer Internship 2025_
