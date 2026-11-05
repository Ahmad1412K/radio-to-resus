"""Controlled vocabularies for the handover schema.

Every enum here exists because at least one criterion in
"Adult Trauma Activation Criteria - 2024" needs to tell its values apart.
Values are lower_snake_case strings so the JSON the model emits stays readable.
"""
from enum import Enum


class Sex(str, Enum):
    MALE = "male"
    FEMALE = "female"


class ArrivalMode(str, Enum):
    EMS_SCENE = "ems_scene"                    # ambulance from the scene
    EMS_INTERFACILITY = "ems_interfacility"    # transfer from another hospital
    WALK_IN = "walk_in"                        # low-risk crash criterion allows walk-ins


class MechanismType(str, Enum):
    FALL = "fall"
    MOTOR_VEHICLE_CRASH = "motor_vehicle_crash"      # patient was a vehicle occupant
    PEDESTRIAN_STRUCK = "pedestrian_struck"
    BICYCLIST_STRUCK = "bicyclist_struck"
    MOTORCYCLE_CRASH = "motorcycle_crash"
    ATV_CRASH = "atv_crash"
    ASSAULT_BLUNT = "assault_blunt"
    OTHER_BLUNT = "other_blunt"
    STAB = "stab"
    GUNSHOT = "gunshot"
    OTHER_PENETRATING = "other_penetrating"
    BURN = "burn"
    ELECTRICAL = "electrical"
    CRUSH = "crush"
    NEAR_DROWNING = "near_drowning"
    NEAR_HANGING = "near_hanging"


VEHICLE_MECHANISMS = {
    MechanismType.MOTOR_VEHICLE_CRASH,
    MechanismType.PEDESTRIAN_STRUCK,
    MechanismType.BICYCLIST_STRUCK,
    MechanismType.MOTORCYCLE_CRASH,
    MechanismType.ATV_CRASH,
}


class FallType(str, Enum):
    SAME_LEVEL = "same_level"
    FROM_HEIGHT = "from_height"


class Ejection(str, Enum):
    NONE = "none"
    PARTIAL = "partial"
    COMPLETE = "complete"


class BodyRegion(str, Enum):
    HEAD = "head"
    FACE = "face"
    NECK = "neck"
    SHOULDER_AXILLA = "shoulder_axilla"
    CHEST = "chest"
    ABDOMEN = "abdomen"
    FLANK = "flank"
    BACK = "back"
    PELVIS = "pelvis"
    GROIN_PERINEUM = "groin_perineum"
    BUTTOCK = "buttock"
    UPPER_ARM = "upper_arm"
    ELBOW = "elbow"
    FOREARM = "forearm"
    WRIST = "wrist"
    HAND = "hand"
    FINGER = "finger"
    THIGH = "thigh"
    KNEE = "knee"
    LOWER_LEG = "lower_leg"
    ANKLE = "ankle"
    FOOT = "foot"
    TOE = "toe"


class AnatomicZone(str, Enum):
    """The landmarks the criteria use for penetrating injury, amputation and open fractures."""
    HEAD = "head"
    NECK = "neck"
    TSHIRT_BOXER = "tshirt_boxer"      # torso plus the area boxer shorts cover
    PROXIMAL_LIMB = "proximal_limb"    # beyond the t-shirt/boxer area but proximal to wrist/ankle
    DISTAL_LIMB = "distal_limb"        # wrist/ankle and beyond


# ASSUMPTION (see report): wrist and ankle themselves count as distal, and the whole
# upper arm and thigh count as proximal limb rather than t-shirt/boxer area.
ZONE_OF_REGION = {
    BodyRegion.HEAD: AnatomicZone.HEAD,
    BodyRegion.FACE: AnatomicZone.HEAD,
    BodyRegion.NECK: AnatomicZone.NECK,
    BodyRegion.SHOULDER_AXILLA: AnatomicZone.TSHIRT_BOXER,
    BodyRegion.CHEST: AnatomicZone.TSHIRT_BOXER,
    BodyRegion.ABDOMEN: AnatomicZone.TSHIRT_BOXER,
    BodyRegion.FLANK: AnatomicZone.TSHIRT_BOXER,
    BodyRegion.BACK: AnatomicZone.TSHIRT_BOXER,
    BodyRegion.PELVIS: AnatomicZone.TSHIRT_BOXER,
    BodyRegion.GROIN_PERINEUM: AnatomicZone.TSHIRT_BOXER,
    BodyRegion.BUTTOCK: AnatomicZone.TSHIRT_BOXER,
    BodyRegion.UPPER_ARM: AnatomicZone.PROXIMAL_LIMB,
    BodyRegion.ELBOW: AnatomicZone.PROXIMAL_LIMB,
    BodyRegion.FOREARM: AnatomicZone.PROXIMAL_LIMB,
    BodyRegion.THIGH: AnatomicZone.PROXIMAL_LIMB,
    BodyRegion.KNEE: AnatomicZone.PROXIMAL_LIMB,
    BodyRegion.LOWER_LEG: AnatomicZone.PROXIMAL_LIMB,
    BodyRegion.WRIST: AnatomicZone.DISTAL_LIMB,
    BodyRegion.HAND: AnatomicZone.DISTAL_LIMB,
    BodyRegion.FINGER: AnatomicZone.DISTAL_LIMB,
    BodyRegion.ANKLE: AnatomicZone.DISTAL_LIMB,
    BodyRegion.FOOT: AnatomicZone.DISTAL_LIMB,
    BodyRegion.TOE: AnatomicZone.DISTAL_LIMB,
}

LIMB_REGIONS = {r for r, z in ZONE_OF_REGION.items()
                if z in (AnatomicZone.PROXIMAL_LIMB, AnatomicZone.DISTAL_LIMB)}


class Side(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    BILATERAL = "bilateral"
    MIDLINE = "midline"


class InjuryType(str, Enum):
    FRACTURE = "fracture"
    PENETRATING_WOUND = "penetrating_wound"
    AMPUTATION = "amputation"
    DEGLOVING = "degloving"
    CRUSH = "crush"
    MANGLED_EXTREMITY = "mangled_extremity"
    PULSELESS_EXTREMITY = "pulseless_extremity"
    SEATBELT_SIGN = "seatbelt_sign"
    LACERATION = "laceration"
    CONTUSION = "contusion"
    ABRASION = "abrasion"
    OTHER = "other"


class Bone(str, Enum):
    SKULL = "skull"
    VERTEBRA = "vertebra"
    CLAVICLE = "clavicle"
    RIB = "rib"
    STERNUM = "sternum"
    HUMERUS = "humerus"
    RADIUS = "radius"
    ULNA = "ulna"
    HAND_BONES = "hand_bones"
    PELVIS = "pelvis"
    FEMUR = "femur"
    PATELLA = "patella"
    TIBIA = "tibia"
    FIBULA = "fibula"
    FOOT_BONES = "foot_bones"
    OTHER = "other"


# The criteria define the long bones they count as "Humerus or Femur".
CRITERIA_LONG_BONES = {Bone.HUMERUS, Bone.FEMUR}

# Bones whose body region is unambiguous; used to reject contradictory records.
REGION_OF_BONE = {
    Bone.SKULL: {BodyRegion.HEAD, BodyRegion.FACE},
    Bone.HUMERUS: {BodyRegion.UPPER_ARM, BodyRegion.SHOULDER_AXILLA, BodyRegion.ELBOW},
    Bone.FEMUR: {BodyRegion.THIGH, BodyRegion.PELVIS, BodyRegion.KNEE},
    Bone.PELVIS: {BodyRegion.PELVIS},
}


class AirwayStatus(str, Enum):
    PATENT = "patent"
    COMPROMISED = "compromised"
    ADVANCED_AIRWAY = "advanced_airway"   # ETT, supraglottic or surgical airway in place


class Hemorrhage(str, Enum):
    NONE = "none"
    CONTROLLED = "controlled"
    ACTIVE_UNCONTROLLED = "active_uncontrolled"


class ShockSign(str, Enum):
    PALE = "pale"
    COOL_CLAMMY = "cool_clammy"
    DIAPHORETIC = "diaphoretic"
    DELAYED_CAP_REFILL = "delayed_cap_refill"
    WEAK_THREADY_PULSE = "weak_thready_pulse"
    ALTERED_MENTATION = "altered_mentation"


class Intervention(str, Enum):
    ENDOTRACHEAL_INTUBATION = "endotracheal_intubation"
    SUPRAGLOTTIC_AIRWAY = "supraglottic_airway"
    SURGICAL_AIRWAY = "surgical_airway"
    BAG_VALVE_MASK = "bag_valve_mask"
    SUPPLEMENTAL_OXYGEN = "supplemental_oxygen"
    NEEDLE_DECOMPRESSION = "needle_decompression"
    TOURNIQUET = "tourniquet"
    WOUND_PACKING = "wound_packing"
    PELVIC_BINDER = "pelvic_binder"
    BLOOD_TRANSFUSION = "blood_transfusion"
    IV_FLUIDS = "iv_fluids"
    SPINAL_MOTION_RESTRICTION = "spinal_motion_restriction"
    SPLINT = "splint"
    ANALGESIA = "analgesia"
    OTHER = "other"


ADVANCED_AIRWAY_INTERVENTIONS = {
    Intervention.ENDOTRACHEAL_INTUBATION,
    Intervention.SUPRAGLOTTIC_AIRWAY,
    Intervention.SURGICAL_AIRWAY,
}


class Trend(str, Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    DETERIORATING = "deteriorating"
