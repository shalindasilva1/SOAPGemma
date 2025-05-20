# Ensure necessary imports are here, e.g.:
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM # etc.

def generate_soap_note(dialogue, model, tokenizer, device="cuda:0",
                       on_start_generation=None, on_end_generation=None):
    """
    Generates a SOAP note from a given dialogue.
    Optionally calls callbacks at the start and end of the generation process.
    """

    input_text = f"dialogue: {dialogue}<soap_start> soap_note:"

    # Tokenize the input
    inputs = tokenizer.encode_plus(
        input_text,
        return_tensors="pt",
        add_special_tokens=True
    ).to(device)

    if on_start_generation and callable(on_start_generation):
        on_start_generation() # Call the start callback

    soap_note = ""
    try:
        # Generate the SOAP note
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=512,
            num_beams=4,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        soap_note = tokenizer.decode(generated_ids, skip_special_tokens=True)
    finally:
        if on_end_generation and callable(on_end_generation):
            on_end_generation() # Call the end callback
        # This message will now appear after the spinner has stopped and cleared its line.
        print("SOAP note generation complete.")


    return soap_note