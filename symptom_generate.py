import json
import os

symptoms = [
    # General
    "fever", "chills", "fatigue", "weakness", "weight loss", "weight gain",
    "night sweats", "loss of appetite", "swelling", "dehydration", "malaise",

    # Cardiovascular
    "chest pain", "palpitations", "shortness of breath on exertion",
    "shortness of breath at rest", "leg swelling", "cyanosis", "fainting", "dizziness",

    # Respiratory
    "cough", "dry cough", "productive cough", "bloody sputum", "wheezing",
    "nasal congestion", "snoring", "hoarseness", "sore throat", "difficulty breathing",

    # Gastrointestinal
    "nausea", "vomiting", "diarrhea", "constipation", "abdominal pain", "heartburn",
    "bloating", "gas", "blood in stool", "mucus in stool", "loss of bowel control",

    # Neurological
    "headache", "migraine", "seizures", "memory loss", "confusion", "tremors",
    "numbness", "tingling", "difficulty speaking", "blurred vision", "vision loss",
    "double vision", "hearing loss", "ringing in ears", "balance problems",

    # Musculoskeletal
    "joint pain", "muscle pain", "back pain", "neck pain", "joint swelling", "joint stiffness",
    "muscle weakness", "limb deformity", "bone pain", "cramps",

    # Genitourinary
    "painful urination", "frequent urination", "blood in urine", "incontinence", "nocturia",
    "pelvic pain", "vaginal discharge", "penile discharge", "erectile dysfunction",
    "testicular pain", "missed periods", "heavy periods", "spotting",

    # Dermatological
    "skin rash", "itching", "dry skin", "oily skin", "acne", "eczema", "hives",
    "bruising", "hair loss", "skin discoloration", "nail changes", "ulcers",

    # Endocrine / Metabolic
    "increased thirst", "increased hunger", "frequent urination", "heat intolerance",
    "cold intolerance", "sweating", "goiter", "menstrual irregularity", "fatigue despite sleep",

    # Psychiatric
    "depression", "anxiety", "panic attacks", "hallucinations", "delusions", "insomnia",
    "excessive sleep", "irritability", "mood swings", "apathy",

    # Pediatric
    "delayed speech", "failure to thrive", "bedwetting", "behavioral issues",

    # Female-specific
    "breast pain", "breast lump", "vaginal bleeding", "pain during intercourse",

    # Male-specific
    "prostate pain", "reduced libido", "scrotal swelling"
]

# Save to file
os.makedirs('static/data', exist_ok=True)
with open('static/data/symptoms.json', 'w') as f:
    json.dump(symptoms, f, indent=2)

print(f"{len(symptoms)} symptoms saved to static/data/symptoms.json")
