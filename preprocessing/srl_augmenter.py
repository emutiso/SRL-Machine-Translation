from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import numpy as np

# Configuration (use GPU)
MODEL_NAME = "mbruton/gal_mBERT"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Map the labels to actual semantic roles
SRL_LABEL_MAP = {
    "arg0": "Agent",        # Generally consistent
    "arg1": "Theme",        # Generally consistent (AKA 'Patient')
    "arg2": "Goal",         # Can be 'benefactor/recipient', 'instrument', 'attribute', or 'goal' depending on context, we will leave as "Goal" "for now
    "root": "Verb"          # Verb is essentially the root (may not need to label this, but we leave it for now)
}


def batch_augment(texts):
    """Batch process texts with SRL tagging"""
    try:
        # Batch tokenization
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True
        )

        # Extract and remove offset mapping before model
        offset_mappings = inputs.pop("offset_mapping")
        inputs = inputs.to(device)

        # Model inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Process all texts in batch
        augmented_texts = []
        for i in range(len(texts)):
            text = texts[i]
            offsets = offset_mappings[i].cpu().numpy()
            input_ids = inputs["input_ids"][i]
            logits = outputs.logits[i].cpu().numpy()

            predictions = np.argmax(logits, axis=1)
            tokens = tokenizer.convert_ids_to_tokens(input_ids)

            modified = text
            offset_adjustment = 0
            predicate_counter = {}
            previous_end = 0  # Track previous token's end position

            # Sort tokens by their start offset
            sorted_indices = np.argsort([offset[0] for offset in offsets])

            for j in sorted_indices:
                token = tokens[j]
                offset = offsets[j]
                start_offset = int(offset[0])
                end_offset = int(offset[1])

                # Skip special/padding tokens and subwords
                if token in ["[CLS]", "[SEP]", "[PAD]"] or start_offset == end_offset:
                    continue
                if token.startswith("##"):
                    continue

                label = model.config.id2label[predictions[j]]
                if ":" in label:
                    root_num, role = label.split(":")
                    root_num = root_num[1:]  # Remove 'r'

                    if root_num not in predicate_counter:
                        predicate_counter[root_num] = len(predicate_counter) + 1
                    pred_num = predicate_counter[root_num]

                    semantic_role = SRL_LABEL_MAP.get(role, role)
                    tag = f"[{semantic_role}{pred_num}] "

                    # Calculate adjusted start position
                    adjusted_start = start_offset + offset_adjustment

                    # Ensure we're not inserting in the middle of a previous tag
                    if start_offset >= previous_end:
                        modified = modified[:adjusted_start] + tag + modified[adjusted_start:]
                        offset_adjustment += len(tag)
                        previous_end = end_offset

            augmented_texts.append(modified)

        return augmented_texts
    except Exception as e:
        print(f"Batch error: {str(e)}") # Incase something happens
        return texts
