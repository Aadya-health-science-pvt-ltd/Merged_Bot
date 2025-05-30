# config/
# Contains configuration management, global constants, and environment variables.
# This would typically be a config.py file or a package with __init__.py and other config-related modules.

# config/constants.py
BATCH_SIZE = 100
EMBEDDING_MODEL = "text-embedding-3-small"
DOCTOR_WEBSITE_URL = "https://www.linqmd.com/doctor/p-v-n-sravanthi"
CLINIC_INFO = {
    "name": "Metro Allergy Clinic",
    "doctor": "Dr. Emily Johnson",
    "services": "Allergy Testing, Immunotherapy, Pediatric Allergies",
    "welcome": "Hello! I am the Metro Allergy Clinic Assistant. How can I help you today?"
}
CLINIC_CONFIG = {
    "clinic_name": "Allergy Central",
    "procedure": "Allergy",
    "doctor_name": "Dr Balachandra BV"
}
SAMPLE_PRESCRIPTION = """Prescribed Medications:
- Loratadine 10mg: 1 tablet daily AM
- Epinephrine auto-injector: 0.3mg IM PRN
- Follow-up in 2 weeks"""


