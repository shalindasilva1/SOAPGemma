def generate_soap_note(dialogue, model, tokenizer, device="cuda:0"):
    """Generates a SOAP note from a given dialogue."""

    input_text = f"dialogue: {dialogue}<soap_start> soap_note:"

    # Tokenize the input
    inputs = tokenizer.encode_plus(
        input_text,
        return_tensors="pt",
        # Ensure tokenizer doesn't add EOS token here if the model adds it during generation
        add_special_tokens=True
    ).to(device)

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

    return soap_note