"""
TRINETRA-VIG: RTO / VAHAN Government Database Mock Service
Foundation: Maharashtra Motor Vehicle Department "Know Your RTO" (Slide 6 of PPT)
Jurisdiction: MH-14 (Pimpri-Chinchwad, Lonavala, Maval)
"""

import time
from typing import Dict, Any, Optional

# Sample RTO vehicle records for MH-14 and interstate corridors
VAHAN_DATABASE = {
    "KA03MN1188": {
        "plate": "KA03MN1188",
        "owner_masked": "R***SH K***R",
        "make_model": "Hyundai Verna (White Sedan)",
        "vehicle_class": "Motor Car (LMV)",
        "color": "White",
        "fuel_type": "PETROL / BS-VI",
        "registration_date": "2023-04-12",
        "rto_jurisdiction": "KA-03 (Bengaluru East / Karnataka)",
        "insurance_valid_upto": "2027-04-10",
        "fitness_valid_upto": "2038-04-11",
        "puc_status": "VALID",
        "blacklist_status": "CLEAR"
    },
    "DL01AB1234": {
        "plate": "DL01AB1234",
        "owner_masked": "A***T S***H",
        "make_model": "Honda City (White Sedan)",
        "vehicle_class": "Motor Car (LMV)",
        "color": "White",
        "fuel_type": "DIESEL",
        "registration_date": "2021-08-19",
        "rto_jurisdiction": "DL-01 (Delhi North)",
        "insurance_valid_upto": "2026-11-20",
        "fitness_valid_upto": "2036-08-18",
        "puc_status": "EXPIRED",
        "blacklist_status": "UNDER_INVESTIGATION (Duplicate Plate Suspect)"
    },
    "MH14DX9921": {
        "plate": "MH14DX9921",
        "owner_masked": "V***M P***L",
        "make_model": "Maruti Suzuki Swift Dzire",
        "vehicle_class": "Motor Car (LMV)",
        "color": "Silver",
        "fuel_type": "CNG / PETROL",
        "registration_date": "2022-01-15",
        "rto_jurisdiction": "MH-14 (Pimpri-Chinchwad / Lonavala)",
        "insurance_valid_upto": "2026-05-14",
        "fitness_valid_upto": "2037-01-14",
        "puc_status": "VALID",
        "blacklist_status": "WANTED (FIR #412/2026 Stolen Vehicle)"
    },
    "MH14EX8821": {
        "plate": "MH14EX8821",
        "owner_masked": "S***L J***I",
        "make_model": "Tata Nexon EV (Blue/White SUV)",
        "vehicle_class": "Electric Vehicle (LMV)",
        "color": "Blue",
        "fuel_type": "ELECTRIC",
        "registration_date": "2024-02-10",
        "rto_jurisdiction": "MH-14 (Pimpri-Chinchwad / Lonavala)",
        "insurance_valid_upto": "2027-02-09",
        "fitness_valid_upto": "2039-02-09",
        "puc_status": "EXEMPT (EV)",
        "blacklist_status": "CLEAR"
    }
}

class VahanService:
    @staticmethod
    def lookup_vehicle(plate: str) -> Dict[str, Any]:
        """Looks up official RTO / VAHAN registry record."""
        clean = plate.replace(" ", "").upper()
        if clean in VAHAN_DATABASE:
            return {
                "found": True,
                "data": VAHAN_DATABASE[clean]
            }

        # Dynamic fallback for arbitrary plates in MH-14 jurisdiction
        is_mh14 = clean.startswith("MH14")
        return {
            "found": True,
            "data": {
                "plate": clean,
                "owner_masked": "CITIZEN REGISTRY",
                "make_model": "Sedan / Passenger Vehicle",
                "vehicle_class": "Motor Car (LMV)",
                "color": "Silver / White",
                "fuel_type": "PETROL",
                "registration_date": "2023-06-20",
                "rto_jurisdiction": "MH-14 (Pimpri-Chinchwad / Lonavala)" if is_mh14 else f"{clean[:4]} Jurisdiction",
                "insurance_valid_upto": "2027-06-19",
                "fitness_valid_upto": "2038-06-19",
                "puc_status": "VALID",
                "blacklist_status": "CLEAR"
            }
        }
