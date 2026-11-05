"""IMIST-AMBO handover record, shaped by "Adult Trauma Activation Criteria - 2024".

Conventions used throughout
---------------------------
* ``None`` means "not mentioned in the handover". It is never the same as a
  negative finding: ``loss_of_consciousness.occurred = False`` means the crew
  said there was no LOC, ``None`` means nobody said anything about it.
* Optional lists follow the same rule: ``allergies = None`` is "not mentioned",
  ``allergies = []`` is "no known allergies".
* Units are the ones the criteria document uses (feet, mph, inches), so a rule
  can be checked against the source text without conversion.
* The record describes what the crew *said*. The true patient state, used to
  generate the handover, is stored separately by the data generator.
"""
from __future__ import annotations

from typing import Annotated, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .vocab import (
    ADVANCED_AIRWAY_INTERVENTIONS, LIMB_REGIONS, REGION_OF_BONE, VEHICLE_MECHANISMS,
    AirwayStatus, ArrivalMode, Bone, BodyRegion, Ejection, FallType, Hemorrhage,
    InjuryType, Intervention, MechanismType, Sex, ShockSign, Side, Trend,
)

SCHEMA_VERSION = "1.0"

Pct = Annotated[float, Field(ge=0, le=100)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# --------------------------------------------------------------------- I
class Identification(_Strict):
    age_years: Optional[Annotated[int, Field(ge=0, le=120)]] = None
    sex: Optional[Sex] = None
    pregnant: Optional[bool] = None
    gestation_weeks: Optional[Annotated[int, Field(ge=1, le=45)]] = None
    arrival_mode: Optional[ArrivalMode] = None
    eta_minutes: Optional[Annotated[int, Field(ge=0, le=240)]] = None

    @model_validator(mode="after")
    def _pregnancy(self):
        if self.gestation_weeks is not None and self.pregnant is not True:
            raise ValueError("gestation_weeks given but pregnant is not true")
        if self.pregnant is True and self.sex == Sex.MALE:
            raise ValueError("pregnant is true but sex is male")
        return self


# --------------------------------------------------------------------- M
class FallDetails(_Strict):
    fall_type: Optional[FallType] = None
    height_ft: Optional[Annotated[float, Field(ge=0, le=1000)]] = None

    @model_validator(mode="after")
    def _height(self):
        if self.fall_type == FallType.SAME_LEVEL and self.height_ft not in (None, 0):
            raise ValueError("a same-level fall cannot have a fall height")
        return self


class VehicleCrashDetails(_Strict):
    impact_speed_mph: Optional[Annotated[float, Field(ge=0, le=250)]] = None
    extrication_minutes: Optional[Annotated[float, Field(ge=0, le=600)]] = None
    intrusion_occupant_site_in: Optional[Annotated[float, Field(ge=0, le=120)]] = None
    intrusion_any_site_in: Optional[Annotated[float, Field(ge=0, le=120)]] = None
    ejection: Optional[Ejection] = None
    death_same_compartment: Optional[bool] = None
    patient_thrown: Optional[bool] = None
    run_over: Optional[bool] = None
    rider_separated: Optional[bool] = None
    rollover: Optional[bool] = None


class LossOfConsciousness(_Strict):
    occurred: Optional[bool] = None
    duration_minutes: Optional[Annotated[float, Field(ge=0, le=1440)]] = None

    @model_validator(mode="after")
    def _duration(self):
        if self.duration_minutes is not None and self.occurred is not True:
            raise ValueError("LOC duration given but occurred is not true")
        return self


class Mechanism(_Strict):
    type: Optional[MechanismType] = None
    description: Optional[Annotated[str, Field(max_length=300)]] = None
    fall: Optional[FallDetails] = None
    vehicle: Optional[VehicleCrashDetails] = None
    loss_of_consciousness: Optional[LossOfConsciousness] = None

    @model_validator(mode="after")
    def _details_match_type(self):
        if self.fall is not None and self.type not in (None, MechanismType.FALL):
            raise ValueError("fall details given for a non-fall mechanism")
        if self.vehicle is not None and self.type is not None and self.type not in VEHICLE_MECHANISMS:
            raise ValueError("vehicle details given for a non-vehicle mechanism")
        return self


# --------------------------------------------------------------------- I (injuries)
class Injury(_Strict):
    type: InjuryType
    region: BodyRegion
    side: Optional[Side] = None
    suspected: Optional[bool] = None              # pre-hospital findings are usually suspected
    bone: Optional[Bone] = None                   # fractures only
    open_fracture: Optional[bool] = None          # fractures only
    depressed: Optional[bool] = None              # skull fractures only
    pelvis_unstable: Optional[bool] = None        # pelvic fractures only
    digits_involved: Optional[Annotated[int, Field(ge=1, le=20)]] = None  # amputations only

    @model_validator(mode="after")
    def _fields_fit_type(self):
        fracture = self.type == InjuryType.FRACTURE
        if not fracture and (self.bone or self.open_fracture is not None
                             or self.depressed is not None or self.pelvis_unstable is not None):
            raise ValueError("bone/open/depressed/unstable are only valid on a fracture")
        if self.depressed is not None and self.bone != Bone.SKULL:
            raise ValueError("'depressed' only applies to a skull fracture")
        if self.pelvis_unstable is not None and self.bone != Bone.PELVIS:
            raise ValueError("'pelvis_unstable' only applies to a pelvic fracture")
        if self.digits_involved is not None and self.type != InjuryType.AMPUTATION:
            raise ValueError("'digits_involved' only applies to an amputation")
        if self.bone in REGION_OF_BONE and self.region not in REGION_OF_BONE[self.bone]:
            raise ValueError(f"bone {self.bone.value} is not in region {self.region.value}")
        limb_only = {InjuryType.PULSELESS_EXTREMITY, InjuryType.MANGLED_EXTREMITY, InjuryType.DEGLOVING}
        if self.type in limb_only and self.region not in LIMB_REGIONS:
            raise ValueError(f"{self.type.value} must be on a limb region")
        return self


class Burns(_Strict):
    tbsa_percent: Optional[Pct] = None            # % total body surface area
    inhalation_injury: Optional[bool] = None
    electrical: Optional[bool] = None             # routed to the burn protocol


# --------------------------------------------------------------------- S
class GCS(_Strict):
    eye: Optional[Annotated[int, Field(ge=1, le=4)]] = None
    verbal: Optional[Annotated[int, Field(ge=1, le=5)]] = None
    motor: Optional[Annotated[int, Field(ge=1, le=6)]] = None
    total: Optional[Annotated[int, Field(ge=3, le=15)]] = None
    before_intubation: Optional[bool] = None      # Level I criterion: GCS < 9 *prior to* intubation

    @model_validator(mode="after")
    def _sum(self):
        parts = (self.eye, self.verbal, self.motor)
        if all(p is not None for p in parts) and self.total is not None and sum(parts) != self.total:
            raise ValueError(f"GCS components sum to {sum(parts)} but total is {self.total}")
        return self


class Vitals(_Strict):
    systolic_bp_mmhg: Optional[Annotated[int, Field(ge=0, le=300)]] = None
    diastolic_bp_mmhg: Optional[Annotated[int, Field(ge=0, le=200)]] = None
    heart_rate_bpm: Optional[Annotated[int, Field(ge=0, le=300)]] = None
    resp_rate_per_min: Optional[Annotated[int, Field(ge=0, le=80)]] = None
    spo2_percent: Optional[Annotated[int, Field(ge=0, le=100)]] = None
    spo2_on_room_air: Optional[bool] = None       # the SpO2 criterion applies on room air only

    @model_validator(mode="after")
    def _consistency(self):
        if self.spo2_on_room_air is not None and self.spo2_percent is None:
            raise ValueError("spo2_on_room_air given without spo2_percent")
        if (self.systolic_bp_mmhg is not None and self.diastolic_bp_mmhg is not None
                and self.diastolic_bp_mmhg > self.systolic_bp_mmhg):
            raise ValueError("diastolic pressure exceeds systolic")
        return self


class Signs(_Strict):
    vitals: Vitals = Field(default_factory=Vitals)
    gcs: Optional[GCS] = None
    airway: Optional[AirwayStatus] = None
    flail_chest: Optional[bool] = None
    pneumothorax_or_hemothorax: Optional[bool] = None
    clinical_signs_of_shock: Optional[bool] = None
    shock_signs: List[ShockSign] = Field(default_factory=list)
    hemorrhage: Optional[Hemorrhage] = None
    paralysis: Optional[bool] = None
    bilateral_sensory_loss: Optional[bool] = None
    suspected_spinal_cord_injury: Optional[bool] = None
    abdominal_pain: Optional[bool] = None
    abdomen_firm_or_distended: Optional[bool] = None
    trend: Optional[Trend] = None

    @model_validator(mode="after")
    def _shock(self):
        if self.shock_signs and self.clinical_signs_of_shock is False:
            raise ValueError("shock signs listed but clinical_signs_of_shock is false")
        return self


# --------------------------------------------------------------------- T
class Treatment(_Strict):
    interventions: List[Intervention] = Field(default_factory=list)
    blood_transfusion_required_to_maintain_vitals: Optional[bool] = None


# --------------------------------------------------------------------- C4: speaker corrections
class Retraction(_Strict):
    """A value the speaker said and then corrected ("BP ninety - sorry, one fifty")."""
    field: Annotated[str, Field(max_length=80, description="dot path, e.g. signs.vitals.systolic_bp_mmhg")]
    retracted_value: Annotated[str, Field(max_length=60)]
    corrected_value: Annotated[str, Field(max_length=60)]


# --------------------------------------------------------------------- the record
class HandoverRecord(_Strict):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    identification: Identification = Field(default_factory=Identification)
    mechanism: Mechanism = Field(default_factory=Mechanism)
    injuries: Optional[List[Injury]] = None
    burns: Optional[Burns] = None
    signs: Signs = Field(default_factory=Signs)
    treatment: Treatment = Field(default_factory=Treatment)
    allergies: Optional[List[Annotated[str, Field(max_length=60)]]] = None
    medications: Optional[List[Annotated[str, Field(max_length=60)]]] = None
    background: Optional[List[Annotated[str, Field(max_length=80)]]] = None
    other: Optional[Annotated[str, Field(max_length=300)]] = None
    retractions: List[Retraction] = Field(default_factory=list)

    @model_validator(mode="after")
    def _cross_section(self):
        given = set(self.treatment.interventions)
        if given & ADVANCED_AIRWAY_INTERVENTIONS and self.signs.airway not in (None, AirwayStatus.ADVANCED_AIRWAY):
            raise ValueError("an advanced airway was placed but airway status says otherwise")
        if Intervention.TOURNIQUET in given and self.signs.hemorrhage == Hemorrhage.NONE:
            raise ValueError("tourniquet applied but hemorrhage recorded as none")
        if self.burns is not None and self.burns.electrical and self.mechanism.type not in (
                None, MechanismType.ELECTRICAL, MechanismType.BURN):
            raise ValueError("electrical burn recorded under a non-burn mechanism")
        return self

    # Convenience for the rule engine; not part of the emitted JSON.
    def in_adult_scope(self) -> Optional[bool]:
        """The criteria cover adults aged 15-64. None when age was not stated."""
        age = self.identification.age_years
        return None if age is None else 15 <= age <= 64
