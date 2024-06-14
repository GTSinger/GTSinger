"""
Module containing all non-spanner subclasses of the :class:`~pymusicxml.score_components.Direction` type.
"""

#  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  #
#  This file is part of SCAMP (Suite for Computer-Assisted Music in Python)                      #
#  Copyright © 2020 Marc Evanstein <marc@marcevanstein.com>.                                     #
#                                                                                                #
#  This program is free software: you can redistribute it and/or modify it under the terms of    #
#  the GNU General Public License as published by the Free Software Foundation, either version   #
#  3 of the License, or (at your option) any later version.                                      #
#                                                                                                #
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;     #
#  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.     #
#  See the GNU General Public License for more details.                                          #
#                                                                                                #
#  You should have received a copy of the GNU General Public License along with this program.    #
#  If not, see <http://www.gnu.org/licenses/>.                                                   #
#  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  #

from __future__ import annotations
from typing import Sequence
from xml.etree import ElementTree
from pymusicxml.enums import StaffPlacement
from pymusicxml.score_components import Duration, Direction


class MetronomeMark(Direction):

    """
    Class representing a tempo-specifying metronome mark

    :param beat_length: length, in quarters, of the note that takes the beat
    :param bpm: beats per minute
    :param voice: Which voice to attach to
    :param staff: Which staff to attach to if the part has multiple staves
    :param placement: Where to place the direction in relation to the staff ("above" or "below")
    :param other_attributes: any other attributes to assign to the metronome mark, e.g. parentheses="yes" or
        font_size="5"
    """

    def __init__(self, beat_length: float, bpm: float, placement: str | StaffPlacement = "above",
                 voice: int = 1, staff: int = None, **other_attributes):
        super().__init__(placement, voice, staff)
        try:
            self.beat_unit = Duration.from_written_length(beat_length)
        except ValueError:
            # fall back to quarter note tempo if the beat length is not expressible as a single notehead
            self.beat_unit = Duration.from_written_length(1.0)
            bpm /= beat_length
        self.bpm = bpm
        self.other_attributes = {key.replace("_", "-"): value for key, value in other_attributes.items()}

    def render_direction_type(self) -> Sequence[ElementTree.Element]:
        type_el = ElementTree.Element("direction-type")
        metronome_el = ElementTree.SubElement(type_el, "metronome", self.other_attributes)
        metronome_el.extend(self.beat_unit.render_to_beat_unit_tags())
        ElementTree.SubElement(metronome_el, "per-minute").text = str(self.bpm)
        return type_el,


class TextAnnotation(Direction):
    """
    Class representing text that is attached to the staff

    :param text: the text of the annotation
    :param font_size: the font size of the text
    :param italic: whether or not the text is italicized
    :param placement: Where to place the direction in relation to the staff ("above" or "below")
    :param voice: Which voice to attach to
    :param staff: Which staff to attach to if the part has multiple staves
    :param kwargs: any extra properties of the musicXML "words" tag aside from font-size and italics can be
        passed to kwargs
    """

    def __init__(self, text: str, font_size: float = None, italic: bool = False, bold: bool = False,
                 placement: str | StaffPlacement = "above", voice: int = 1, staff: int = None, **kwargs):
        super().__init__(placement, voice, staff)
        self.text = text
        self.text_properties = kwargs
        if font_size is not None:
            self.text_properties["font-size"] = font_size
        if italic:
            self.text_properties["font-style"] = "italic"
        if bold:
            self.text_properties["font-weight"] = "bold"

    def render_direction_type(self) -> Sequence[ElementTree.Element]:
        type_el = ElementTree.Element("direction-type")
        ElementTree.SubElement(type_el, "words", self.text_properties).text = self.text
        return type_el,


class Dynamic(Direction):
    """
    Class representing a dynamic that is attached to the staff

    :param dynamic_text: the text of the dynamic, e.g. "mf"
    :param voice: Which voice to attach to
    :param staff: Which staff to attach to if the part has multiple staves
    """

    STANDARD_TYPES = ("f", "ff", "fff", "ffff", "fffff", "ffffff", "fp", "fz", "mf", "mp", "p", "pp", "ppp", "pppp",
                      "ppppp", "pppppp", "rf", "rfz", "sf", "sffz", "sfp", "sfpp", "sfz")

    def __init__(self, dynamic_text: str, placement: str | StaffPlacement = "below",
                 voice: int = 1, staff: int = None):
        self.dynamic_text = dynamic_text
        super().__init__(placement, voice, staff)

    def render_direction_type(self) -> Sequence[ElementTree.Element]:
        type_el = ElementTree.Element("direction-type")
        dynamics_el = ElementTree.SubElement(type_el, "dynamics")
        if self.dynamic_text in Dynamic.STANDARD_TYPES:
            ElementTree.SubElement(dynamics_el, self.dynamic_text)
        else:
            ElementTree.SubElement(dynamics_el, "other-dynamics").text = self.dynamic_text
        return type_el,


class Harmony(Direction):
    """
    Class representing harmonic notation.
    """
    KINDS = ("augmented", "augmented-seventh", "diminished",
            "diminished-seventh", "dominant", "dominant-11th", "dominant-13th",
            "dominant-ninth", "French", "German", "half-diminished", "Italian",
            "major", "major-11th", "major-13th", "major-minor", "major-ninth",
            "major-seventh", "major-sixth", "minor", "minor-11th", "minor-13th",
            "minor-ninth", "minor-seventh", "minor-sixth", "Neapolitan", "none",
            "other", "pedal", "power", "suspended-fourth", "suspended-second",
            "Tristan")

    def __init__(self, root_letter: str, root_alter: int, kind: str,
                 use_symbols: bool = False, degrees: Sequence[Degree] = (),
                 placement: str | StaffPlacement = "above"):
        if kind not in self.KINDS:
            raise ValueError(f"Chord {kind} of invalid kind. Allowed values: {self.KINDS}")
        self.root_letter = root_letter
        self.root_alter = root_alter
        self.kind = kind
        self.use_symbols = use_symbols
        self.degrees = list(degrees)
        super().__init__(placement, 1, None)

    def render(self) -> Sequence[ElementTree.Element]:
        harmony_el = ElementTree.Element("harmony")
        root_el = ElementTree.SubElement(harmony_el, "root")
        ElementTree.SubElement(root_el, "root-step").text = str(self.root_letter)
        ElementTree.SubElement(root_el, "root-alter").text = str(self.root_alter)
        ElementTree.SubElement(harmony_el, "kind").text = str(self.kind)
        for d in self.degrees:
            harmony_el.extend(d.render())
        return harmony_el,

    def render_direction_type(self) -> Sequence[ElementTree.Element]:
        return self.render(),


class Degree:
    """
    The <degree> element is used to add, alter, or subtract individual notes in the chord.

    :param value: The number of the degree, a positive integer.
    :param alter: An integer meaning alteration by semitones.
    :param degree_type: Type of alteration. A positive alter + 'subtract' = semitone down.
    :param print_object: Whether to print the degree or not.
    """
    DEGREE_TYPES = ("add", "alter", "subtract")

    def __init__(self, value: int, alter: int, degree_type: str = "alter", print_object: bool = True):
        assert degree_type in self.DEGREE_TYPES
        self.value = value
        self.alter = alter
        self.degree_type = degree_type
        self.print_object = print_object

    def render(self) -> Sequence[ElementTree.Element]:
        degree_element = ElementTree.Element("degree", {"print-object": "yes" if self.print_object else "no"})
        ElementTree.SubElement(degree_element, "degree-value").text = str(self.value)
        ElementTree.SubElement(degree_element, "degree-alter").text = str(self.alter)
        ElementTree.SubElement(degree_element, "degree-type").text = str(self.degree_type)
        return degree_element,
