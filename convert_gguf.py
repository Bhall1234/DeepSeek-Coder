from transformers import AutoModelForCausalLM, AutoTokenizer

# Path to your model directory
model_path = "./OUTPUT"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load the model with safetensors
model = AutoModelForCausalLM.from_pretrained(model_path, use_safetensors=True)

# Example usage
input_text = "Hello, world!"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model(**inputs)

# Print the output logits
print(outputs.logits)
