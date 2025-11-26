import numpy as np
import torch

from srl_augmenter import tokenizer, device, model, SRL_LABEL_MAP, batch_augment


def augment_sentence_test(text):
    """Add SRL tags to text"""
    try:
        inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
        offset_mapping = inputs.pop("offset_mapping").squeeze(0)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        predictions = np.argmax(outputs.logits.cpu().numpy(), axis=2)[0]
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        offsets = offset_mapping.cpu().numpy()

        modified = text
        offset_adjustment = 0
        predicate_counter = {}  # Track predicate numbers

        for i, (token, offset) in enumerate(zip(tokens, offsets)):
            if token in ["[CLS]", "[SEP]"]:
                continue

            label = model.config.id2label[predictions[i]]
            if ":" in label:
                root_num, role = label.split(":")
                root_num = root_num[1:]  # Remove 'r' from 'r0'

                # Get predicate number (unique per root)
                if root_num not in predicate_counter:
                    predicate_counter[root_num] = len(predicate_counter) + 1
                pred_num = predicate_counter[root_num]

                # Map to semantic role
                semantic_role = SRL_LABEL_MAP.get(role, role)
                tag = f"[{semantic_role}{pred_num}] "

                start = offset[0] + offset_adjustment
                modified = modified[:start] + tag + modified[start:]
                offset_adjustment += len(tag)

        return modified
    except Exception as e:
        print(f"Error processing text: {text}")
        print(f"Error: {str(e)}")
        return text  # Return original text on failure

test_sentence = "\"We now have 4-month-old mice that are non-diabetic that used to be diabetic,\" he added."
augmented = augment_sentence_test(test_sentence)
augmented_batch = batch_augment([test_sentence])[0]

print(f"Original: {test_sentence}")
print(f"Augmented: {augmented}")
print(f"    Batch: {augmented_batch}")

print("Model labels:", model.config.id2label)