"""Links from schema fields to standard terminologies, used to build the JSON-LD vocabulary.

Every LOINC code below was checked against the NLM Clinical Tables LOINC API
(clinicaltables.nlm.nih.gov) on 2026-10-08. UCUM unit codes follow the UCUM
specification. SNOMED CT bindings are intentionally absent until the UMLS
licence is in place; FIELDS_AWAITING_SNOMED lists where they will go.
"""

NAMESPACE = "https://w3id.org/radio-to-resus/vocab#"   # placeholder: register on w3id.org before publishing

PREFIXES = {
    "r2r": NAMESPACE,
    "loinc": "https://loinc.org/",
    "snomed": "http://snomed.info/id/",
    "ucum": "http://unitsofmeasure.org/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
}

# Model.field -> LOINC code (exact match), verified 2026-10-08
LOINC = {
    "Identification.age_years": ("30525-0", "Age"),
    "Identification.sex": ("46098-0", "Sex"),
    "Vitals.systolic_bp_mmhg": ("8480-6", "Systolic blood pressure"),
    "Vitals.diastolic_bp_mmhg": ("8462-4", "Diastolic blood pressure"),
    "Vitals.heart_rate_bpm": ("8867-4", "Heart rate"),
    "Vitals.resp_rate_per_min": ("9279-1", "Respiratory rate"),
    "Vitals.spo2_percent": ("59408-5", "Oxygen saturation in Arterial blood by Pulse oximetry"),
    "GCS.total": ("9269-2", "Glasgow coma score total"),
    "GCS.eye": ("9267-6", "Glasgow coma score eye opening"),
    "GCS.verbal": ("9270-0", "Glasgow coma score verbal"),
    "GCS.motor": ("9268-4", "Glasgow coma score motor"),
}

# Model.field -> UCUM unit code
UCUM = {
    "Identification.age_years": "a",
    "Identification.gestation_weeks": "wk",
    "Identification.eta_minutes": "min",
    "FallDetails.height_ft": "[ft_i]",
    "VehicleCrashDetails.impact_speed_mph": "[mi_i]/h",
    "VehicleCrashDetails.extrication_minutes": "min",
    "VehicleCrashDetails.intrusion_occupant_site_in": "[in_i]",
    "VehicleCrashDetails.intrusion_any_site_in": "[in_i]",
    "LossOfConsciousness.duration_minutes": "min",
    "Burns.tbsa_percent": "%",
    "Vitals.systolic_bp_mmhg": "mm[Hg]",
    "Vitals.diastolic_bp_mmhg": "mm[Hg]",
    "Vitals.heart_rate_bpm": "/min",
    "Vitals.resp_rate_per_min": "/min",
    "Vitals.spo2_percent": "%",
}

# Enumerations whose values should be bound to SNOMED CT concepts once licensed.
FIELDS_AWAITING_SNOMED = [
    "MechanismType", "InjuryType", "BodyRegion", "Bone", "Intervention",
    "ShockSign", "AirwayStatus", "Hemorrhage",
]

# Optional list fields whose emptiness is meaningful ("no known allergies").
# RDF has no way to say "an empty list", so the JSON-LD view adds an explicit
# boolean `<field>Stated` beside each one.
STATED_LIST_FIELDS = ["injuries", "allergies", "medications", "background"]
