"""Parse demographic estimation results."""

import re
import json
from datetime import datetime


def parse_estimation(estimation: str) -> dict:
    """Parse estimation text into structured data."""
    result = {}

    patterns = {
        "age_lower": r"<age_lower>(\d+)</age_lower>",
        "age_upper": r"<age_upper>(\d+)</age_upper>",
        "age_confidence": r"<age_confidence>([\d.]+)</age_confidence>",
        "income_lower": r"<income_lower>([\d,]+)</income_lower>",
        "income_upper": r"<income_upper>([\d,]+)</income_upper>",
        "income_confidence": (r"<income_confidence>([\d.]+)</income_confidence>"),
        "education": r"<education>([^<]+)</education>",
        "education_confidence": (
            r"<education_confidence>([\d.]+)</education_confidence>"
        ),
        "sex": r"<sex>([^<]+)</sex>",
        "sex_confidence": r"<sex_confidence>([\d.]+)</sex_confidence>",
        "visa_status": r"<visa_status>([^<]+)</visa_status>",
        "visa_confidence": (r"<visa_confidence>([\d.]+)</visa_confidence>"),
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, estimation)
        if match:
            value = match.group(1).strip()
            if "confidence" in key or key in [
                "age_lower",
                "age_upper",
            ]:
                result[key] = float(value.replace(",", ""))
            elif key in ["income_lower", "income_upper"]:
                result[key] = int(value.replace(",", ""))
            else:
                result[key] = value

    return result


def save_results(results: list[dict], output_path: str) -> None:
    """Save results to JSON file with timestamp."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "total_personas": len(results),
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
