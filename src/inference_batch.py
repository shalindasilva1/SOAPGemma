import torch


def generate_batch(dialogues, model, tokenizer, device="cuda:0"):
    inputs = [f"dialogue: {d}<soap_start> soap_note:" for d in dialogues]
    encodings = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)

    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
            num_beams=4,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    return [tokenizer.decode(output[input_ids.shape[1]:], skip_special_tokens=True) for output in outputs]
