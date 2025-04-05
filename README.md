<div align="center">

  <h1>Visual Search Using Vision-Language Models</h1>

</div>

![image](https://github.com/aiishwarrya/VisualLanguageModel/blob/main/ss/image.png)

<div align="justify">

---

## Introduction  

This project is under the **Intel® Unnati Industrial Training 2025** initiative and is developed by **Team Raven**, consisting of **Aishwarya Joshi** and **Mahathi R**.  

In an era where information retrieval is dominated by text-based search engines, visual search is emerging as a transformative technology. Traditional search methods rely heavily on keyword matching, often failing to capture the **semantic intent** behind a query. This project explores **Vision-Language Models (VLMs)** to bridge this gap, enabling a more intuitive and **context-aware** search experience.  

We are building a **visual search engine** powered by a **custom Vision-Language Model** that prioritizes **SigLip over CLIP** for improved contrastive learning. Our approach focuses on **contrastive learning, vision encoding, and text generation** to create a **highly efficient image-text retrieval system**. SigLip enhances **training stability, retrieval accuracy, and computational efficiency**, making searches more robust and reliable.  

### Key Features  

- **Text-to-Image Search**: Retrieve images based on natural language descriptions.  
- **Image-to-Image Search**: Find visually similar images from a dataset.  
- **Hybrid Querying**: Combine text and images for enhanced search precision.  
- **Contextual Understanding**: Moves beyond keyword dependency to capture **semantic meaning**.  
- **Efficient Indexing**: Utilizes optimized retrieval techniques for large-scale datasets.  

## Objectives  

This project aims to develop a **Visual Search Engine** using a **custom-built VLM**, leveraging **contrastive learning, multi-head attention, and transformer-based architectures** to align images and text in a **shared semantic space**. The core objectives include:  

### 1️⃣ Implementing Contrastive Learning for Vision-Language Alignment  
- Prioritizing **SigLip over CLIP** due to its improved numerical stability and better softmax behavior, preventing gradient explosion during contrastive training.  
- Training the model on paired **image-text datasets**, ensuring robust **cross-modal embeddings**.  

### 2️⃣ Developing a High-Performance Vision Encoder  
- Use a **Vision Transformer (ViT)** backbone for **image feature extraction**, incorporating **patch embeddings, positional encodings, and self-attention mechanisms**.  
- Optimize training stability with **Batch Normalization, Layer Normalization, and RMS Normalization**, ensuring efficient gradient flow and weight updates.  

### 3️⃣ Implementing an Efficient Text Encoder  
- Fine-tune a **decoder-only language model** to generate high-dimensional embeddings for textual queries.  
- Employ **Rotary Positional Embeddings (RoPE)** for improved contextual understanding in transformer-based sequence processing.  

### 4️⃣ Optimizing Inference with KV-Cache & Attention Mechanisms  
- Implement **Key-Value (KV) Caching** to store previously computed attention scores, reducing redundant computations and accelerating inference.  
- Utilize **Grouped Query Attention (GQA)** to improve memory efficiency and speed during large-scale query processing.  

### 5️⃣ Enhancing Image-Text Retrieval Efficiency  
- Implement **Image Features Projection** to align vision embeddings with textual embeddings in a **latent space**.  
- Use **Top-P (Nucleus) Sampling and Temperature Scaling** during inference to control the diversity and randomness of text-based retrieval.  

### 6️⃣ Deploying and Benchmarking Performance  
- Evaluate retrieval accuracy using standard **vision-language benchmarks** and compare results against state-of-the-art **VLM models**.  
- Fine-tune indexing and retrieval techniques for **scalability** in large datasets.

---

## ✅ Steps to Run PaLIGemma Locally

### **Step 1: Clone the Repo Locally**
```bash
git clone https://github.com/google-research/paligemma
cd paligemma
```

### **Step 2: Download the Weights**
Go to [https://huggingface.co/google/paligemma-3b-pt-224/tree/main](https://huggingface.co/google/paligemma-3b-pt-224/tree/main) and download the model files (e.g., `pytorch_model.bin`, `config.json`, etc.).

Save them to a local path, for example:
```
C:\Users\rmaha\OneDrive\Desktop\paligemma-3b-pt-224
```

### **Step 3: Upload Any Image You Want to Use**
Place your image at a known location. For example:
```
C:\Users\rmaha\OneDrive\Desktop\visual-search-machine\pic1.jpeg
```

### **Step 4: Set the Model Path**
Update your script or environment with the path to the model directory:

```bash
MODEL_PATH="C:\Users\rmaha\OneDrive\Desktop\paligemma-3b-pt-224"
```

### **Step 5: Set the Image Path**
Update the path to point to the image you want to use:

```bash
IMAGE_FILE_PATH="C:\Users\rmaha\OneDrive\Desktop\visual-search-machine\pic1.jpeg"
```

### **Step 6: Set the Prompt**
Define the text prompt that the model will complete based on the image:

```bash
PROMPT="this building is "
```

### ✅ Run Inferency.py
```

---

## Approach taken and Why?

To build a **Visual-Language Model (VLM)** that understands both text and images, we use **contrastive learning**—a method that trains the model to pull matching image-text pairs closer while pushing mismatched ones apart. This allows the model to build a **shared semantic space** where similar concepts are aligned, even if they come from different modalities.

### Why Contrastive Learning?  
Traditional supervised learning struggles to capture the **relationships** between text and images since they exist in different representational spaces. Contrastive learning solves this by ensuring:  

- **Semantic Alignment**: Text and images with similar meanings have closer embeddings.  
- **Zero-Shot Learning Capability**: Once trained on diverse data, the model generalizes to unseen text-image pairs without retraining.  
- **Better Representation Learning**: Unlike simple classification, contrastive learning teaches the model to understand nuanced relationships.  

### Implementation in Our Model  
Instead of using **CLIP**, which has known issues with numerical stability in the **Softmax function**, we implement **SigLip**, an improved contrastive learning technique.  

- **SigLip optimizes Softmax behavior**, preventing gradient explosion and ensuring stable training.  
- **We train on large-scale image-text pairs**, embedding them in a unified space for robust retrieval.  
- **The model is optimized using normalized temperature scaling**, improving convergence speed and accuracy.

With a well-trained contrastive learning backbone, our VLM can match queries with the most relevant images, making visual search highly effective.  

---

## **Vision Transformer (ViT): How Our Model Sees Images**  

To make sense of images, our Vision-Language Model (VLM) needs a **Vision Transformer (ViT)**—a deep learning architecture that encodes images into numerical representations. Instead of analyzing images pixel by pixel, **ViT divides them into smaller patches and processes them like a sequence**, similar to how transformers handle text.  

### **Why Vision Transformers?**  
Unlike CNNs (Convolutional Neural Networks), which focus on local patterns, ViTs learn **global relationships** within an image using **self-attention mechanisms**. This allows our model to:  
✅ Capture **long-range dependencies** between different parts of an image.  
✅ Learn **contextual relationships** instead of just edges, textures, or colors.  
✅ Align image representations with text embeddings for **better contrastive learning**.  

### **How It Works in Our VLM**  
1️. **Image Patching** → The input image is split into fixed-size patches (e.g., 16×16 pixels).  
2️. **Linear Embedding** → Each patch is converted into a numerical vector using a learnable linear projection.  
3️. **Position Encoding** → Since transformers don’t inherently understand spatial relationships, we add **Rotary Positional Encoding (RoPE)** to retain the structure of the image.  
4️. **Self-Attention** → The model applies **multi-head self-attention**, allowing it to compare patches and **understand their relationships** within the image.  
5️. **Final Image Representation** → The output is a **compressed numerical representation of the image**, which can now be aligned with text using **contrastive learning**.  

## **Contrastive Learning: Why We Use SigLip Instead of CLIP**  
Our VLM trains its vision encoder using **contrastive learning**, where paired **image-text** data is used to bring matching pairs closer in embedding space while pushing non-matching pairs apart.  
- CLIP originally introduced contrastive learning for vision-language tasks.  
- **We use SigLip instead of CLIP** because it improves the **numerical stability** of the Softmax function, preventing gradient explosion and leading to more **robust training**.  
- SigLip enhances the alignment of **image and text embeddings**, making visual search more **accurate and efficient**.  

### **Implementation Breakdown**  
- **Coding SigLip’s Vision Encoder** → We implement the **ViT-based encoder** optimized for contrastive learning.  
- **Applying Normalization** → We integrate **BatchNorm, LayerNorm, and RMSNorm** to ensure stable training.  
- **Using RoPE** → We apply **Rotary Positional Encoding** to help the model understand spatial relationships within the image.

---

## **Language Model: How Our VLM Understands Text and Generates Meaning**

While the Vision Transformer (ViT) helps our model understand images, we also need a **language model** to interpret text and generate meaningful responses. In our implementation, we use a **decoder-based transformer**, which enables the VLM to produce context-aware outputs from image embeddings, including captions, search queries, or related textual results.

### **Why Do We Need a Language Model?**

- It **interprets the image embeddings** produced by the vision encoder and grounds them in natural language.  
- It enables the model to **generate coherent text**, allowing image captioning, Q&A, or visual search.  
- It helps **align multimodal embeddings** by unifying vision and language features into a shared space.

## **Our Choice: A Lightweight Decoder-Only Transformer**

We implement a **lightweight, decoder-only transformer** inspired by open models like **Gemma**, adapted for multimodal learning:

- It is efficient and scalable, suitable for both research and real-world use.  
- It uses **causal self-attention**, ensuring the output depends only on previously seen tokens.  
- It supports **Key-Value Caching (KV-Cache)** to reduce redundant computations during inference.

### **How It Works in Our VLM**

1️. **Text Tokenization** → The input text (or image-derived caption) is converted into token IDs and embedded into vectors.  
2️. **Causal Self-Attention** → Multi-head attention is used to learn dependencies between tokens, respecting their sequence order.  
3️. **Feedforward Network (FFN)** → Each transformer block uses FFNs to refine the representations from the attention layer.  
4️. **Cross-Modal Integration** → Vision encoder outputs are injected into the language decoder to merge visual and textual context.  
5️. **Text Output Generation** → After several decoding layers, the final embeddings are projected back to vocabulary space to generate text.

## **KV-Cache: Making Inference Faster**

Transformers typically recompute attention for every previous token during inference, which becomes costly in long sequences.  
**Key-Value Caching (KV-Cache)** resolves this by storing the key and value matrices from earlier steps:

- Reduces redundant calculations during decoding  
- Speeds up inference significantly, especially in large-scale retrieval tasks  
- Allows real-time response in visual search applications

### **Implementation Breakdown**

- **Decoder Transformer Stack** → A causal transformer with multi-head attention, FFNs, and token embeddings  
- **KV-Cache Integration** → Saves and reuses past key-value states for efficient inference  
- **RMSNorm** → We use **RMS normalization** across the model, following best practices from Gemma for stability and efficiency

---

## **Normalization Techniques: Making Training Stable and Efficient**

This section will explain how we stabilize training and improve generalization using normalization layers. Since we're dealing with deep transformer stacks (both in the encoder and decoder), normalization plays a key role in keeping gradients stable, preventing exploding/vanishing activations, and accelerating convergence.
In large models like ours, especially those with deep transformer layers, **normalization is essential**. It ensures that the scale of intermediate activations stays manageable and helps the model converge faster and more reliably during training.
We experiment with several types of normalization and select the best one based on stability, speed, and compatibility with our lightweight architecture.

### **Why Normalization Matters**
- Helps stabilize gradients during backpropagation  
- Prevents vanishing or exploding activations in deep models  
- Enables higher learning rates, leading to faster convergence  
- Improves generalization on unseen image-text pairs

### **Normalization Methods We Explored**

1. **Batch Normalization (BatchNorm)**  
   - Normalizes across the batch dimension  
   - Works well in convolutional architectures  
   - Not ideal for variable-length sequences or transformer-based models  
2. **Layer Normalization (LayerNorm)**  
   - Normalizes across the features of a single token  
   - Common in standard transformer architectures  
   - Sensitive to initialization and may require careful tuning  
3. **RMS Normalization (RMSNorm)**  
   - A simpler variant of LayerNorm that only scales inputs based on their root-mean-square  
   - No bias term, fewer parameters  
   - Works well with lightweight transformers (e.g., Gemma)  
   - Chosen for our VLM due to better empirical stability and speed
   
### **Why We Chose RMSNorm**

- Empirically more stable in our decoder-based language model  
- Reduces computational complexity by removing the bias and mean subtraction steps  
- Keeps memory and parameter usage low — a good fit for our lightweight VLM  
- Compatible with other techniques like KV-Cache and Rotary Positional Encoding

### **Implementation Highlights**

- RMSNorm is applied **after attention and feedforward layers** in both vision and language stacks  
- We keep normalization placement consistent to ensure predictable training behavior  
- Our implementation is modular, allowing easy switching between normalization types for experimentation

---

## **Rotary Positional Encoding (RoPE): How Our Model Understands Order**

Transformers, by design, are unaware of the order of their inputs. Whether it's a sequence of words or a set of image patches, they treat each element independently unless explicitly told otherwise. But in both language and vision tasks, **order matters** — the structure of a sentence, the layout of objects in an image, or even the timing of events in a video all rely on sequence.

To bridge this gap, models use **positional encoding** — a method for injecting information about the relative or absolute position of tokens into the model. In our VLM, we use a technique called **Rotary Positional Encoding (RoPE)**, which introduces this ordering in a more elegant and computationally stable way than traditional methods.

- ### **Why We Chose RoPE Over Other Techniques**

Traditional positional encodings, like sinusoidal or learnable embeddings, represent position in an absolute sense — token 3 is "here", token 4 is "there", and so on. This works, but it doesn't help the model understand how far apart two tokens are or how their relative positions impact meaning. RoPE takes a different approach by applying **rotations in complex space** to the attention mechanism itself, allowing the model to encode relative positions directly into the **query and key vectors** of self-attention.

This method is not only **parameter-free** (it doesn't require extra learned weights), but it’s also **compatible with inference optimizations** like **KV-Cache**, which is crucial for speeding up generation. That makes RoPE a natural fit for our lightweight, fast, and scalable architecture.

- ### **How Rotary Encoding Works in Practice**

Instead of using a separate embedding layer to encode position, RoPE modifies the attention mechanism directly. It applies a mathematically defined rotation to each vector in a way that subtly alters how the model perceives spatial or sequential relationships. These rotations help the model capture **how far apart** two tokens or image patches are, and how their relative position affects their meaning or interaction.

In our architecture, we integrate RoPE in both the **Vision Transformer (ViT)** and the **decoder-only language model**, ensuring that both image and text modalities carry a strong sense of order and structure throughout the pipeline.

### **Summary**

By using RoPE, our VLM gains the ability to understand not just what elements are in an image or sentence, but also how those elements are arranged. This helps improve alignment between visual and textual representations, which is essential for tasks like **captioning**, **semantic search**, or **image-text retrieval**. It’s a small change in architecture — but one that makes a big impact on performance and coherence.

---

## **Top-p Sampling: How Our VLM Generates Meaningful Output**

<p align="center">
  <img src="https://github.com/aiishwarrya/VisualLanguageModel/blob/main/ss/top%20-p.png?raw=true" width="400" height="600">
</p>

Once our model has aligned visual and textual features, processed the inputs, and built rich embeddings, it’s time to generate actual text — whether it’s a caption, a search query, or a response. But choosing the next word isn’t just about picking the one with the highest probability. That approach often leads to **repetitive or bland outputs**.

To make the output **more diverse yet coherent**, we use a technique called **Top-p Sampling**, also known as **nucleus sampling**.

- ### **Why Top-p Sampling?**

Language models usually output a probability distribution over the vocabulary at each step. Instead of always picking the most likely word (as in greedy decoding), Top-p Sampling introduces **controlled randomness**:

- It selects from the **smallest possible set of tokens** whose **cumulative probability exceeds a threshold _p_** (e.g., 0.9).
- This dynamic cutoff ensures that **rare but contextually appropriate words still have a chance to be selected**, improving fluency and diversity.
- It avoids issues like repetition or overly safe outputs, which are common in greedy or top-k sampling methods.
  
### **How It Works in Our VLM**

Here’s a simplified breakdown of what happens during generation:

1. The model processes the input (image and/or text).
2. It outputs a probability distribution over possible next tokens.
3. The tokens are sorted by probability.
4. The model keeps the top tokens whose combined probability is ≥ _p_ (say, 90%).
5. One token is then randomly sampled from this **nucleus**.

This method ensures **contextual diversity without losing coherence** — ideal for a visual search engine that needs to generate accurate yet natural-sounding responses.

- ### **Why Not Top-k Sampling?**

Top-k sampling keeps the top _k_ tokens regardless of their cumulative probability. That can lead to:
- Fixed-size candidate pools that sometimes ignore context.
- Missing out on high-probability rare words that lie just outside the top _k_.
Top-p is more adaptive — the size of the candidate pool changes based on how confident the model is, which helps in **multimodal settings** where context richness varies from image to image.

### **Implementation Details**

We implement Top-p Sampling as the final step in our decoding loop:

- After computing token probabilities using the decoder’s output logits, we:
  - Sort tokens by probability.
  - Compute cumulative probabilities.
  - Select the smallest set of tokens exceeding our threshold (_p_).
  - Sample the next token from this nucleus.

This process repeats for each generated token until an end token is reached or a maximum length is met.

---

## **Final Inference Pipeline: How Our VLM Processes Image & Text Queries**

Now that all the components of our Vision-Language Model (VLM) are in place — from the Vision Transformer (ViT) and contrastive learning with SigLip, to the transformer-based language decoder and optimized inference — it's time to bring everything together.
This section explains how the full inference pipeline works when a user submits either an image or a textual query. This is the **engine room** of our visual search system.

### **Step-by-Step: From Input to Output**

When the model receives an input (image or text), here’s what happens behind the scenes:

#### 1. **Input Processing**  
- **For image queries**, the Vision Transformer (ViT) converts the image into a dense embedding using patching and self-attention.  
- **For text queries**, the language model tokenizes and embeds the text.

#### 2. **Embedding Alignment**  
Both types of inputs (image and text) are mapped into the same **shared multimodal embedding space**. This ensures that semantically similar images and texts are close together — a critical feature for **visual search**.

#### 3. **Similarity Computation**  
We compute the **cosine similarity** between the query embedding and all embeddings in the database (which could be images or text). This determines how closely the query matches existing items.

#### 4. **Retrieval or Generation**  
- If it’s a **search task**, the top-k most similar items are retrieved and returned.  
- If it’s a **captioning or generation task**, the language model uses **causal decoding** with **KV-cache** and **top-p sampling** to generate descriptive text based on the input.

---

## **Putting It All Together**

At this point, our VLM acts like a complete **multimodal engine**. Here's a high-level overview of how it functions:

| Component         | Role in Inference Pipeline                            |
|------------------|--------------------------------------------------------|
| Vision Transformer (ViT) | Converts images into dense embeddings              |
| SigLip (Contrastive Learning) | Ensures image-text alignment in shared space    |
| Transformer Decoder | Generates or interprets text using embeddings         |
| KV-Cache + RoPE  | Speeds up inference and preserves token order          |
| Cosine Similarity | Matches queries to database items                      |


### **Example Use Cases**

- **Text-to-Image Search**: "Find me an image of a red sports car"  
→ Model encodes the text and retrieves visually similar images.
- **Image-to-Text Search**: Upload an image of a city skyline  
→ Model encodes the image and retrieves related text captions or similar images.
- **Image Captioning**: Upload a photo  
→ Model generates a caption using the language decoder.

---

## **Conclusion: Wrapping Up Our Vision-Language Model**

Our Vision-Language Model (VLM) is designed to perform **image-to-text tasks** — specifically, **recognizing visual inputs and generating natural language descriptions**. From processing image patches with a Vision Transformer (ViT), aligning visual and textual embeddings through **SigLip-style contrastive learning**, and decoding responses using a **lightweight language model**, each part of our architecture is built for **efficiency, alignment, and clarity**.
Whether it's for visual search, caption generation, or semantic image understanding, our custom VLM ties together recent advances in transformer-based modeling while retaining simplicity and flexibility.

## **📊 End-to-End VLM Architecture**

Here’s a **flow diagram** that visualizes the full architecture of our system:

<p align="center">
  <img src="https://github.com/aiishwarrya/VisualLanguageModel/blob/main/ss/flowdiagram.png">
</p>

### 🔹 **1. Image Input**
The process starts with a raw image. This image is divided into **fixed-size patches** (e.g., 16×16 pixels), which are treated as tokens — similar to words in a sentence.

### 🔵 **2. Vision Encoder (ViT)**  
The **Vision Transformer (ViT)** takes these image patches and converts them into high-dimensional embeddings.  
- We use **Rotary Positional Encoding** so that the model retains spatial information.
- **Multi-head Self-Attention** helps the encoder understand relationships between different image regions.
- The final output is a **compact embedding** representing the full image's content.

### 🟠 **3. Contrastive Learning with SigLip**  
During training, we use **SigLip** (a variant of CLIP) to align image embeddings and text embeddings.  
- Positive image-text pairs are pulled closer in the shared embedding space.  
- Negative (mismatched) pairs are pushed apart.  
- SigLip improves training stability through a more numerically stable Softmax.

### 🟣 **4. Text Decoder (Language Model)**  
The visual embedding from ViT is passed to a **decoder-only transformer** (inspired by Gemma) which generates the output text.
- Uses **Causal Self-Attention** to generate one word at a time, based on prior context.
- **Key-Value Caching (KV-Cache)** is applied to speed up inference by reusing previous attention computations.
- Outputs a human-readable **caption**, **description**, or **semantic interpretation** of the input image.
  
### 🟡 **5. Output**  
Finally, the model returns a natural language output — this could be:
- An image caption
- A textual description
- A keyword summary
- Or a phrase relevant to search or retrieval
---

</div>






