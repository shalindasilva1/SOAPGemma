import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import time
import threading
import sys
import argparse
import os
# Assuming inference.py is in the same directory or src is in PYTHONPATH
# If inference.py is in src/, and model.py is in the root, use:
# from src import inference
import inference


# --- Global variables for spinner ---
BASE_MODEL_ID = "google/txgemma-2b-predict"
DEFAULT_ADAPTER_PATH = "models/SOAPgemma_v1"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

bnb_config = None
if DEVICE != "cpu":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

spinner_stop_event = threading.Event()
spinner_thread_instance = None # To keep track of the spinner thread

# --- Spinner Animation Logic (remains in model.py) ---
def spinner_animation():
    """Displays a spinning cursor animation."""
    spinner_chars = ['|', '/', '-', '\\']
    idx = 0
    while not spinner_stop_event.is_set():
        # Message for the spinner
        sys.stdout.write(f"\rGenerating SOAP note... {spinner_chars[idx % len(spinner_chars)]}")
        sys.stdout.flush()
        idx += 1
        time.sleep(0.1)
    # Clear the spinner line when done
    sys.stdout.write('\r' + ' ' * (len("Generating SOAP note... ") + 5) + '\r')
    sys.stdout.flush()

# --- Spinner Control Functions (to be passed as callbacks) ---
def start_spinner_thread():
    """Starts the spinner animation in a separate thread."""
    global spinner_stop_event, spinner_thread_instance
    spinner_stop_event.clear() # Reset event for multiple calls
    spinner_thread_instance = threading.Thread(target=spinner_animation)
    spinner_thread_instance.daemon = True # Allow main program to exit even if thread is running
    spinner_thread_instance.start()

def stop_spinner_thread():
    """Stops the spinner animation thread."""
    global spinner_stop_event, spinner_thread_instance
    spinner_stop_event.set()
    if spinner_thread_instance and spinner_thread_instance.is_alive():
        spinner_thread_instance.join()
    # The "SOAP note generation complete." message is now handled by inference.py

# --- Model Loading and Dialogue Input (as before) ---
def load_model_and_tokenizer(base_model_id, adapter_path, quantization_cfg, device_to_map):
    # (Your existing load_model_and_tokenizer function - no changes needed here)
    print(f"Loading base model: {base_model_id} onto device: {device_to_map}")
    if "cuda" in device_to_map:
        current_device_idx = int(device_to_map.split(':')[1]) if ":" in device_to_map else 0
        current_device_map = {"": current_device_idx}
    else:
        current_device_map = device_to_map
    print(f"Using device_map: {current_device_map}")
    if quantization_cfg:
        print(f"Using quantization: {quantization_cfg}")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=quantization_cfg,
        device_map=current_device_map,
        torch_dtype="auto",
        attn_implementation="eager"
    )
    print(f"Loading tokenizer for: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    print(f"Loading PEFT adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    print("Model and tokenizer loaded successfully.")
    return model, tokenizer


def get_dialogue_file_path_interactively():
    # (Your existing get_dialogue_file_path_interactively function - no changes needed here)
    while True:
        try:
            file_path = input("Please enter the path to the dialogue text file: ")
            if not os.path.exists(file_path):
                print(f"Error: File not found at '{file_path}'. Please try again.")
                continue
            if os.path.isdir(file_path):
                print(f"Error: Expected a file path, but got a directory path '{file_path}'. Please try again.")
                continue
            with open(file_path, 'r', encoding='utf-8') as f:
                content_check = f.read()
                if not content_check.strip():
                    print(f"Warning: The dialogue file '{file_path}' is empty or contains only whitespace.")
            return file_path
        except Exception as e:
            print(f"An error occurred while trying to access the file: {e}. Please try again.")


def main():
    global spinner_thread_instance # Ensure main can access this for error handling
    parser = argparse.ArgumentParser(description="Generate a medical SOAP note from a dialogue file.")
    parser.add_argument("--dialogue-file", dest="dialogue_file_path", type=str, default=None,
                        help="Path to the text file containing the patient-doctor dialogue (optional).")
    parser.add_argument("--adapter_path", type=str, default=DEFAULT_ADAPTER_PATH,
                        help="Path to the PEFT adapter model.")
    parser.add_argument("--base_model_id", type=str, default=BASE_MODEL_ID,
                        help="Hugging Face model ID for the base model.")
    parser.add_argument("--device", type=str, default=DEVICE,
                        help="Device to run the model on (e.g., 'cuda:0', 'cpu').")

    args = parser.parse_args()
    dialogue_file_to_use = args.dialogue_file_path

    if dialogue_file_to_use is None:
        print("No dialogue file provided via command-line argument.")
        try:
            current_dir = os.getcwd()
            print(f"Current working directory: {current_dir}")
        except Exception as e:
            print(f"Could not determine current working directory: {e}")
        dialogue_file_to_use = get_dialogue_file_path_interactively()

    try:
        with open(dialogue_file_to_use, 'r', encoding='utf-8') as f:
            dialogue_content = f.read()
        if args.dialogue_file_path and not dialogue_content.strip():
             print(f"Error: The dialogue file '{dialogue_file_to_use}' provided via argument is empty or contains only whitespace.")
             return
    except FileNotFoundError:
        print(f"Error: Dialogue file not found at '{dialogue_file_to_use}'")
        return
    except Exception as e:
        print(f"Error reading dialogue file '{dialogue_file_to_use}': {e}")
        return

    current_bnb_config = None
    if "cuda" in args.device:
        current_bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    try:
        print("Initializing model and tokenizer...")
        model, tokenizer = load_model_and_tokenizer(
            args.base_model_id,
            args.adapter_path,
            current_bnb_config,
            args.device
        )
        print("-" * 30)
        print(f"\nInput Dialogue (from file: {dialogue_file_to_use}):\n{dialogue_content}\n")
        print("-" * 30)

        # Pass the spinner control functions as callbacks
        generated_note = inference.generate_soap_note(
            dialogue_content,
            model,
            tokenizer,
            device=args.device,
            on_start_generation=start_spinner_thread, # Callback to start spinner
            on_end_generation=stop_spinner_thread     # Callback to stop spinner
        )
        # The "SOAP note generation complete." message is now printed by inference.py
        print(f"\nGenerated SOAP Note:\n{generated_note}")

    except Exception as e:
        # Ensure spinner is stopped if an error occurs
        if spinner_thread_instance and spinner_thread_instance.is_alive():
            print("\nStopping spinner due to error...")
            stop_spinner_thread()
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()