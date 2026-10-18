
from model import classify_text, CATEGORIES
from sklearn.metrics import accuracy_score, classification_report


# Synthetic examples with manually assigned labels

examples = [
    # Symptoms
    ("The patient feels weak and has lost appetite.", "symptoms"),
    ("She complains of discomfort when walking.", "symptoms"),
    ("He has been feeling unusually tired.", "symptoms"),

    # Diagnosis
    ("Blood tests confirmed anemia.", "diagnosis"),
    ("Imaging revealed a fracture in the left arm.", "diagnosis"),
    ("The examination established the presence of arthritis.", "diagnosis"),

    # Treatment
    ("The patient started a rehabilitation program.", "treatment"),
    ("A surgical procedure was recommended.", "treatment"),
    ("The physician suggested regular physiotherapy.", "treatment"),

    # Medication
    ("The dosage was increased to 20 mg daily.", "medication"),
    ("The physician adjusted the patient's prescription.", "medication"),
    ("The patient discontinued the tablets.", "medication"),

    # More challenging examples
    ("The patient has diabetes and takes insulin.", "medication"),
    ("The patient reports pain following surgery.", "symptoms"),
    ("The physician prescribed medication for hypertension.", "medication"),
    ("The patient was admitted for treatment of pneumonia.", "treatment")
]



true_labels = []
predicted_labels = []

for text, true_label in examples:
    result = classify_text(text)
    prediction = result["label"]

    true_labels.append(true_label)
    predicted_labels.append(prediction)

    print(f"Text: {text}")
    print(f"Expected: {true_label}")
    print(f"Predicted: {prediction}")
    print("-" * 50)


accuracy = accuracy_score(true_labels, predicted_labels)

print(f"\nAccuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(
    classification_report(
        true_labels,
        predicted_labels,
        labels=CATEGORIES,
        zero_division=0
    )
)
