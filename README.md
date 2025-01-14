# Nutri-Gen
# Nutrigen Personalized Meal Plan Generator

This repository contains the implementation of a fine-tuned language model designed to generate personalized meal plans based on user health data. The model was developed as part of a Final Year Project (FYP) and leverages state-of-the-art techniques in natural language processing.

## Features

- **Data Cleaning and Preparation**: The dataset is preprocessed to ensure proper formatting, with missing or null values handled appropriately.
- **Custom Fine-Tuning**: The model is fine-tuned on a custom dataset using the [unsloth](https://github.com/unslothai/unsloth) framework and PEFT (Parameter-Efficient Fine-Tuning) with LoRA (Low-Rank Adaptation).
- **Alpaca-Style Prompting**: Input-output pairs are formatted in a structured template for training and inference.
- **Efficient Training**: Optimized for low memory usage with techniques like mixed precision (FP16/BFloat16), gradient accumulation, and 4-bit quantization.
- **Evaluation**: The model is evaluated using BLEU scores to measure the quality of generated outputs against ground truth data.

## Tools and Libraries Used

### Python Libraries
- `pandas`: For data manipulation and cleaning.
- `datasets`: For handling and formatting datasets in Hugging Face format.
- `transformers`: For working with pretrained models and tokenizers.
- `trl`: For training language models with reinforcement learning.
- `torch`: For GPU-accelerated model training and inference.
- `evaluate`: For computing BLEU scores and other metrics.
- `unsloth`: For loading and fine-tuning the base model (Qwen2.5-1.5B-Instruct).

### Tools
- **Google Colab**: For prototyping and model training.
- **GitHub**: For version control and repository hosting.
- **Hugging Face Datasets**: For dataset handling and preprocessing.

## Workflow

1. **Dataset Preparation**:
    - Load a cleaned dataset (`cleaned_dataset.csv`) containing `Input` and `Output` columns.
    - Format the dataset into Alpaca-style prompts using a custom mapping function.
    - Split the dataset into training (80%) and validation (20%) sets.

2. **Model Configuration**:
    - Load the `Qwen2.5-1.5B-Instruct` model using the `FastLanguageModel` class.
    - Configure PEFT with LoRA to enable memory-efficient fine-tuning.

3. **Fine-Tuning**:
    - Define training arguments (batch size, learning rate, epochs, etc.).
    - Fine-tune the model using `SFTTrainer` on the formatted dataset.
    - Save the fine-tuned model and tokenizer.

4. **Inference**:
    - Generate personalized meal plans using Alpaca-style prompts.
    - Adjust sampling parameters (temperature, top-k, top-p) for diverse outputs.

5. **Evaluation**:
    - Compute BLEU scores on the validation dataset to evaluate model performance.

## Installation

To replicate this project:

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/nutrigen-meal-plan.git
   cd nutrigen-meal-plan
   ```

2. Install required dependencies:
   ```bash
   pip install pandas datasets transformers accelerate trl unsloth evaluate sacrebleu
   ```

3. Mount Google Drive if using Colab:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

## Example Usage

To generate a meal plan, format the input and call the model:

```python
alpaca_prompt = """Input: {}
Output: {}"""

input_text = "I am 20 years old and I do active cardio. Please suggest a complete day meal plan."
formatted_input = alpaca_prompt.format(input_text, "")

inputs = tokenizer(
    [formatted_input],
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=256
).to("cuda" if torch.cuda.is_available() else "cpu")

outputs = model.generate(
    input_ids=inputs["input_ids"],
    max_new_tokens=150,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    eos_token_id=tokenizer.eos_token_id
)

response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(response[0])
```

## Dataset

- **Name**: `cleaned_dataset.csv`
- **Columns**: `Input`, `Output`
- **Description**: Contains user inputs (e.g., health details) and corresponding output meal plans.

## Model Details

- **Base Model**: `Qwen2.5-1.5B-Instruct`
- **Fine-Tuning Framework**: PEFT with LoRA
- **Evaluation Metric**: BLEU Score

## Results

- **Validation BLEU Score**: Calculated using `sacrebleu`.

## Acknowledgments

Special thanks to:
- [Unsloth GitHub Repository](https://github.com/unslothai/unsloth)
- Hugging Face for their datasets and transformers library.

---

Feel free to contribute or raise issues!
