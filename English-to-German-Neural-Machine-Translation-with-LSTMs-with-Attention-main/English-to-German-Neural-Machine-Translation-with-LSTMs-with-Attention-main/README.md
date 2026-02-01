# English-to-German Neural Machine Translation (NMT) with Attention

This project implements a sequence-to-sequence (Seq2Seq) Deep Learning model capable of translating English sentences into German. Unlike standard RNN approaches, this implementation utilizes an Attention Mechanism to solve the vanishing gradient problem, allowing the model to handle longer sequences with higher accuracy.

## Project Overview

Machine translation is a complex NLP task that requires understanding context, grammar, and vocabulary. This project builds an NMT system from scratch using the Trax deep learning framework.

The architecture consists of:
1.  **Encoder:** An LSTM-based network that processes the input English sentence.
2.  **Attention Layer:** A Scaled Dot-Product Attention mechanism that allows the decoder to "focus" on relevant parts of the input sentence at each generation step.
3.  **Decoder:** An LSTM-based network that generates the German translation.

## Tech Stack & Concepts

* **Language:** Python 3
* **Frameworks:** Google Trax, NumPy
* **Model Architecture:** Encoder-Decoder with Attention
* **Decoding Strategies:**
    * **Greedy Decoding:** Selecting the highest probability token at each step.
    * **Minimum Bayes-Risk (MBR) Decoding:** Generating multiple samples and selecting the one with the highest similarity score (ROUGE-1/Jaccard) to the others.

##  Project Structure

* `data/`: Contains the pre-processed vocabulary and dataset (Opus/Medical dataset).
* `nmt_attention.ipynb`: The main Jupyter Notebook containing the model architecture, training loop, and inference logic.
* `model.pkl.gz`: (Optional) Saved model weights.

##  Model Architecture Details

### The Attention Mechanism
In a standard Seq2Seq model, the encoder compresses the entire input sentence into a single fixed-length vector. This causes information loss for long sentences.

This project implements Scaled Dot Product Attention:
$$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$

This allows the decoder to access all encoder hidden states (Keys/Values) using its current state (Query), effectively "looking back" at the source sentence dynamically.

### Decoding Strategy
The project implements Minimum Bayes-Risk (MBR) decoding to improve translation quality. Instead of just picking one output, the model:
1.  Generates multiple candidate translations using temperature sampling.
2.  Compares candidates against each other using Jaccard Similarity.
3.  Selects the candidate that is most similar to the average of all samples.

## Results

The model was trained on the `opus/medical` dataset.

**Example Translation:**
> **Input:** "I love languages."
> **Output:** "Ich liebe Sprachen."

**MBR Decoding Example:**
> **Input:** "She speaks English and German."
> **Output:** "Sie spricht Deutsch und Englisch."

##  How to Run

1.  **Install Dependencies:**
    ```bash
    pip install trax numpy termcolor
    ```

2.  **Run the Notebook:**
    Launch Jupyter Notebook and open `English to German Neural Machine Translation with LSTM's + Attention.ipynb`.

3.  **Training:**
    Run the training cells to train the model from scratch (Note: This may take time depending on your hardware).

4.  **Inference:**
    Use the `mbr_decode()` function at the end of the notebook to translate custom sentences.