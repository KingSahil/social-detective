"""
Unit tests for multi-modal GEOINT engine and OSINT footprint corroboration.
"""

from pathlib import Path
import pytest
from app.geo import (
    analyze_image_geolocation,
    corroborate_geolocation_from_metadata,
    GeoIntelResult,
)


def test_corroborate_amritsar_gndu():
    res = corroborate_geolocation_from_metadata(
        source_url="https://in.linkedin.com/in/ashwath-soni-b5bba32ba",
        title="Ashwath Soni - SDE @AutonomixSolutions | GDG On Campus Organizer",
        author="Ashwath Soni",
        events=["GDG On Campus GNDU", "Amritsar, Punjab"],
    )
    assert res is not None
    assert res.detected is True
    assert res.city == "Amritsar"
    assert res.region == "Punjab"
    assert res.coordinates == (31.6366, 74.8252)
    assert "Amritsar" in res.location_name


def test_corroborate_goa():
    res = corroborate_geolocation_from_metadata(
        title="Building at Hacker House Goa",
        events=["Hacker House Goa"],
    )
    assert res is not None
    assert res.detected is True
    assert res.region == "Goa"
    assert res.coordinates == (15.2993, 74.1240)


def test_corroborate_pune():
    res = corroborate_geolocation_from_metadata(
        title="Symbiosis International University Tech Summit",
    )
    assert res is not None
    assert res.detected is True
    assert res.city == "Pune"
    assert res.coordinates == (18.5284, 73.8743)


def test_corroborate_rishikesh():
    res = corroborate_geolocation_from_metadata(
        text="Chilling at the tapovan riverside in rishikesh",
    )
    assert res is not None
    assert res.detected is True
    assert res.city == "Rishikesh"
    assert res.coordinates == (30.1245, 78.3211)


def test_no_false_positive_coordinates():
    # An empty or generic query must NOT return Rishikesh coordinates
    res = corroborate_geolocation_from_metadata(
        source_url="https://example.com/profile",
        title="John Doe Profile",
    )
    assert res is None


def test_analyze_image_geolocation_headshot():
    p13 = Path("data/input/test_face_13.jpg")
    if p13.exists():
        res = analyze_image_geolocation(p13)
        # test_face_13 is a cropped portrait with no river rapids or EXIF GPS
        assert res.detected is False
        assert res.coordinates is None
        assert res.location_name == "Undetermined"


def test_analyze_image_geolocation_landscape():
    p4 = Path("data/input/test_face_4.jpg")
    if p4.exists():
        res = analyze_image_geolocation(p4)
        assert res.detected is True
        assert res.coordinates == (30.1245, 78.3211)
        assert "Rishikesh" in res.location_name
