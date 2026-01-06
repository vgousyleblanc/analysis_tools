"""
Module to load detector geometry from a YAML file and compute distances between detectors.
Written by Bruno, to be included by Alie in the general beam analysis code  .
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Union
import yaml


@dataclass(frozen=True)
class DetectorDB:
    """
    Container for detector geometry information loaded from a YAML file.
    frozen=True means the content cannot be modified after creation,
    """
    data: Dict[str, Any]

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "DetectorDB":
        """
        Reads a YAML file and returns a DetectorDB instance.

        Parameters
        ----------
        path : str or Path
            Path to the YAML file containing detector definitions.

        Returns
        -------
        DetectorDB
            A DetectorDB object holding the parsed YAML data.

        Raises
        ------
        ValueError
            If the YAML file does not contain a valid 'detectors' section.
        """
        path = Path(path)

        # Open and parse the YAML file
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Basic validation to ensure expected structure exists
        if not isinstance(data, dict) or "detectors" not in data:
            raise ValueError("YAML must contain a top level key named 'detectors'.")

        if not isinstance(data["detectors"], dict) or not data["detectors"]:
            raise ValueError("'detectors' must be a non empty mapping of detector names.")

        # Create and return the DetectorDB object
        return cls(data=data)

    def _center(self, det_name: str) -> float:
        """
        Internal helper method to retrieve the center position of a detector.

        Parameters
        ----------
        det_name : str
            Name of the detector (e.g. 'T0', 'T4', 'T5').

        Returns
        -------
        float
            Center position of the detector in meters.

        Raises
        ------
        KeyError
            If the detector name is not defined in the YAML.
        ValueError
            If the detector does not define 'center_m'.
        """
        dets = self.data["detectors"]

        # Check that the requested detector exists
        if det_name not in dets:
            available = ", ".join(sorted(dets.keys()))
            raise KeyError(f"Unknown detector '{det_name}'. Available: {available}")

        # Extract the center position
        center = dets[det_name].get("center_m", None)
        if center is None:
            raise ValueError(f"Detector '{det_name}' is missing 'center_m'.")

        return float(center)

    def distance_m(self, a: str, b: str) -> float:
        """
        Compute the absolute distance between two detectors.

        The order of detectors does not matter:
        distance_m('T0', 'T5') == distance_m('T5', 'T0')

        Parameters
        ----------
        a, b : str
            Names of the two detectors.

        Returns
        -------
        float
            Absolute distance between detector centers in meters.
        """
        return abs(self._center(a) - self._center(b))


def detector_distance_m(yaml_path: Union[str, Path], det_a: str, det_b: str) -> float:
    """
    Convenience function to compute detector distance in one line.

    This is useful for scripts or notebooks where you do not want
    to explicitly manage a DetectorDB object.

    Parameters
    ----------
    yaml_path : str or Path
        Path to the detector YAML file.
    det_a, det_b : str
        Names of the two detectors.

    Returns
    -------
    float
        Absolute distance between detector centers in meters.
    """
    db = DetectorDB.from_yaml(yaml_path)
    return db.distance_m(det_a, det_b)


if __name__ == "__main__":
    # Example usage when running this file directly

    db = DetectorDB.from_yaml("../include/wcte_beam_detectors.yaml")

    print("T0 to T1 distance (m):", db.distance_m("T0", "T1"))
    print("T0 to T4 distance (m):", db.distance_m("T0", "T4"))
    print("T0 to T5 distance (m):", db.distance_m("T0", "T5"))
    print("T4 to T1 distance (m):", db.distance_m("T4", "T1"))
    print("T4 to T5 distance (m):", db.distance_m("T4", "T5"))
    print("T1 to T5 distance (m):", db.distance_m("T1", "T5"))
