"""
Module containing all of the classes representing the hierarchy of a musical score.
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
from typing import MutableSequence, Sequence, Type, Iterator, Any
from pymusicxml.enums import StaffPlacement
from ._utilities import _least_common_multiple, _is_power_of_two, _escape_split, get_average_square_correlation
from xml.etree import ElementTree
from abc import ABC, abstractmethod
from fractions import Fraction
from numbers import Real
import math
import datetime
import tempfile
import subprocess
from copy import deepcopy
import logging


# TODO: Make nested printing possible

# --------------------------------------------------- Utilities ----------------------------------------------------


def pad_with_rests(components, desired_length):
    """
    Appends rests to a list of components to fill out the desired length

    :param components: a list of MusicXMLComponents
    :param desired_length: in quarters
    :return: an expanded list of components
    """
    components = [components] if not isinstance(components, (tuple, list)) else components
    assert all(hasattr(component, "true_length") for component in components)
    sum_length = sum(component.true_length for component in components)
    assert sum_length <= desired_length
    remaining_length = Fraction(desired_length - sum_length).limit_denominator()
    assert _is_power_of_two(remaining_length.denominator), "Remaining length cannot require tuplets."
    components = list(components)

    longer_rests = []
    for longer_rest_length in (4, 2, 1):
        while remaining_length >= longer_rest_length:
            longer_rests.append(Rest(longer_rest_length))
            remaining_length -= longer_rest_length
            remaining_length = Fraction(remaining_length).limit_denominator()
    longer_rests.reverse()

    while remaining_length > 0:
        odd_remainder = 1 / remaining_length.denominator
        components.append(Rest(odd_remainder))
        remaining_length -= odd_remainder
        remaining_length = Fraction(remaining_length).limit_denominator()

    return components + longer_rests


# --------------------------------------------- Abstract Parent Class -----------------------------------------------


class MusicXMLComponent(ABC):

    """
    Abstract base class of all musical objects, providing functionality for rendering and exporting to a file.
    """

    @abstractmethod
    def render(self) -> Sequence[ElementTree.Element]:
        """
        Renders this component to a tuple of ElementTree.Element. (The reason for making it a tuple is that musical
        objects like chords are represented by several notes side by side, with all but the first containing
        a </chord> tag.)
        """
        pass

    @abstractmethod
    def wrap_as_score(self) -> Score:
        """
        Wraps this component in a :class:`Score` so that it can be exported and viewed
        """
        pass

    def to_xml(self, pretty_print: bool = False) -> str:
        """
        Renders this component to MusicXML, adding a version tag, but not wrapping it up as a full score.

        :param pretty_print: If True, breaks the MusicXML onto multiple lines, with indentation
        """
        element_rendering = self.render()

        if pretty_print:
            # this is not ideal; it's ugly and requires parsing and re-rendering and then removing version tags
            # for now, though, it's good enough and gets the desired results
            from xml.dom import minidom
            header = '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD ' \
                     'MusicXML 3.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">\n'
            import re
            return header + ''.join(
                re.sub(
                    r"<\?xml version.*\?>\n",
                    "",
                    minidom.parseString(ElementTree.tostring(element, 'utf-8')).toprettyxml(indent="\t")
                )

                for element in element_rendering
            )
        else:
            header = b'<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD ' \
                     b'MusicXML 3.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">'
            return (header + b''.join(ElementTree.tostring(element, 'utf-8') for element in element_rendering)).decode()

    def export_to_file(self, file_path: str, pretty_print: bool = True) -> None:
        """
        Exports this musical object (wrapped as a score) to the given file path.

        :param file_path: The path of the file we want to write to.
        :param pretty_print: If True, breaks the MusicXML onto multiple lines, with indentation
        """
        with open(file_path, 'w') as file:
            file.write(self.wrap_as_score().to_xml(pretty_print))

    def view_in_software(self, command: str) -> None:
        """
        Uses the given terminal command to create a score out of this musical object, and open it in music
        notation software.

        :param command: The terminal command corresponding to the software with which we want to open the score.
        """
        with tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False) as file:
            # Note: For some reason the minified (non-pretty print) version causes MuseScore to spin out and fail
            # This seems to be on MuseScore's end, because there's no difference aside from minification
            file.write(self.wrap_as_score().to_xml(pretty_print=True).encode())
        subprocess.Popen(_escape_split(command, " ") + [file.name])


class MusicXMLContainer(MutableSequence):

    """
    Base class for all musical objects that contain other musical objects, such as :class:`Part`,
    :class:`Measure`, and `Tuplet`.

    :param contents: Musical objects to be contained within this object
    :param allowed_types: Allowable types for objects contained within this object
    :ivar contents: Musical objects contained within this object
    :ivar allowed_types: Allowable types for objects contained within this object
    """

    def __init__(self, contents: Sequence[MusicXMLComponent], allowed_types: Sequence[Type]):
        contents = [] if contents is None else contents
        self.allowed_types = tuple(allowed_types) if not isinstance(allowed_types, tuple) else allowed_types
        if not all(isinstance(x, self.allowed_types) for x in contents):
            raise ValueError("Contents not of correct type.")
        self.contents = list(contents)

    def insert(self, i, o) -> None:
        """
        Insert the given object before the given index.

        :param i: Index at which to insert
        :param o: Object to insert
        """
        if not isinstance(o, self.allowed_types):
            raise ValueError("Trying to insert object of incorrect type.")
        self.contents.insert(i, o)

    def __getitem__(self, i):
        self.contents.__getitem__(i)

    def __setitem__(self, i, o):
        self.contents.__setitem__(i, o)

    def __delitem__(self, i):
        self.contents.__delitem__(i)

    def __len__(self):
        return self.contents.__len__()


class DurationalObject(MusicXMLComponent, ABC):

    """
    Abstract base class for all objects that have duration within a measure.
    """

    @property
    @abstractmethod
    def true_length(self) -> float:
        """
        True length in terms of the number of quarter notes, taking into tuplet time modification. Returns 0 in the
        case of grace notes.
        """
        pass

    @property
    @abstractmethod
    def written_length(self) -> float:
        """
        Written length in terms of the number of quarter notes.
        """
        pass

    @property
    @abstractmethod
    def length_in_divisions(self) -> int:
        """
        Length in terms of subdivisions. (See description of "divisions" attribute in :class:`Duration`)
        """
        pass

    @property
    @abstractmethod
    def divisions(self) -> int:
        """
        Subdivision used when representing this duration.
        """
        pass

    @abstractmethod
    def min_denominator(self) -> int:
        """
        Minimum divisor of a quarter note that would be needed to represent the duration of this note accurately. For
        instance, a triplet quarter note would have min_denominator 3, since it is 2/3 of a quarter.
        """
        pass


# --------------------------------------------- Pitch and Duration -----------------------------------------------


class Pitch(MusicXMLComponent):

    """
    Class representing a notated musical pitch.

    :param step: letter name of the pitch ("c", "d", "e", "f", "g", "a" or "b")
    :param octave: which octave it is in (the octave starting with middle C is octave 4)
    :param alteration: number of half steps sharp or flat. For instance, 1 would be sharp, -2 would be double-flat,
        and 0.5 would be quarter-tone sharp.
    :ivar step: letter name of the pitch ("c", "d", "e", "f", "g", "a" or "b")
    :ivar octave: octave of the pitch
    :ivar alteration: number of half steps sharp or flat
    """

    def __init__(self, step: str, octave: int, alteration: float = 0):
        self.step = step
        self.octave = octave
        self.alteration = alteration

    @classmethod
    def from_string(cls, pitch_string: str):
        """
        Constructs Pitch from either a lilypond pitch string or from standard pitch/octave notation

        :param pitch_string: can take the form "C#5" (specifying octave with number, and using '#' for sharp) or "cs'"
            (specifying octave in the lilypond style and using 's' for sharp)
        :return: a Pitch
        """
        pitch_string = pitch_string.lower()
        assert pitch_string[0] in ('c', 'd', 'e', 'f', 'g', 'a', 'b'), "Pitch string not understood"
        step = pitch_string[0].upper()
        if pitch_string[1:].startswith(('b', 'f', '#', 's')):
            alteration = -1 if pitch_string[1:].startswith(('b', 'f')) else 1
            octave_string = pitch_string[2:]
        elif pitch_string[1:].startswith(('qb', 'qf', 'q#', 'qs')):
            alteration = -0.5 if pitch_string[1:].startswith(('qb', 'qf')) else 0.5
            octave_string = pitch_string[3:]
        else:
            alteration = 0
            octave_string = pitch_string[1:]

        try:
            octave = int(octave_string)
        except ValueError:
            if all(x == '\'' for x in octave_string):
                octave = 3 + len(octave_string)
            elif all(x == ',' for x in octave_string):
                octave = 3 - len(octave_string)
            else:
                raise ValueError("Pitch string not understood")
        return cls(step, octave, alteration)

    def render(self) -> Sequence[ElementTree.Element]:
        pitch_element = ElementTree.Element("pitch")
        step_el = ElementTree.Element("step")
        step_el.text = self.step
        alter_el = ElementTree.Element("alter")
        alter_el.text = str(self.alteration)
        octave_el = ElementTree.Element("octave")
        octave_el.text = str(self.octave)
        pitch_element.append(step_el)
        pitch_element.append(alter_el)
        pitch_element.append(octave_el)
        return pitch_element,

    def wrap_as_score(self) -> Score:
        return Note(self, 1.0).wrap_as_score()

    def __eq__(self, other):
        if not isinstance(other, Pitch):
            return False
        return self.step == other.step and self.octave == other.octave and self.alteration == other.alteration

    def __repr__(self):
        return "Pitch(\"{}\", {}{})".format(self.step, self.octave,
                                            ", {}".format(self.alteration) if self.alteration != 0 else "")


class Duration(DurationalObject):
    """
    Represents a length that can be written as a single note or rest.

    :param note_type: written musicXML duration type, e.g. "quarter"
    :param num_dots: number of duration dots
    :param tuplet_ratio: One of (a) None, indicating not part of a tuplet (b) a tuple of either
        (# actual notes, # normal notes) (c) a tuple of (# actual, # normal, note type), e.g. (4, 3, 0.5)
        for 4 in the space of 3 eighths.
    :ivar note_type: written musicXML duration type, e.g. quarter
    :ivar num_dots: number of duration dots
    :ivar tuplet_ratio: see param definition.
    """

    _length_to_note_type = {
        8.0: "breve",
        4.0: "whole",
        2.0: "half",
        1.0: "quarter",
        0.5: "eighth",
        0.25: "16th",
        1.0 / 8: "32nd",
        1.0 / 16: "64th",
        1.0 / 32: "128th",
        1.0 / 64: "256th",
        1.0 / 128: "512th",
        1.0 / 256: "1024th"
    }

    _note_type_to_length = {b: a for a, b in _length_to_note_type.items()}

    _valid_divisors = tuple(4 / x for x in _length_to_note_type)

    _note_type_to_num_beams = {
        "breve": 0,
        "whole": 0,
        "half": 0,
        "quarter": 0,
        "eighth": 1,
        "16th": 2,
        "32nd": 3,
        "64th": 4,
        "128th": 5,
        "256th": 6,
        "512th": 7,
        "1024th": 8
    }

    def __init__(self, note_type: str, num_dots: int = 0, tuplet_ratio: tuple = None):
        assert note_type in Duration._length_to_note_type.values()
        self.note_type = note_type
        self.num_dots = num_dots
        assert isinstance(tuplet_ratio, (type(None), tuple))
        self.tuplet_ratio = tuplet_ratio
        self._divisions = Fraction(self.true_length).limit_denominator().denominator

    @property
    def divisions(self) -> int:
        """
        In MusicXML, "divisions" is a measure attribute that specifies a smallest metric subdivision for
        the measure in terms of how many would fit in a quarter note. So if divisions is 8, then a dotted quarter would
        have duration 12. When we create a duration, divisions is automatically set to the the smallest number with
        which we could accurately specify this duration. So a dotted quarter note would get set to 2 (since it's three
        8th notes), and a triplet quarter would get set to 3 (since it's 2/3rds of a quarter note). Later, when we go
        to render a whole measure, the least common multiple of all of these minimum divisions for all the objects in
        the measure gets calculated, and the divisions attribute for all the objects gets reset to that.
        """
        return self._divisions

    @divisions.setter
    def divisions(self, value):
        self._divisions = value

    @staticmethod
    def _dot_multiplier(num_dots):
        return (2.0 ** (num_dots + 1) - 1) / 2.0 ** num_dots

    @property
    def written_length(self) -> float:
        return Duration._note_type_to_length[self.note_type] * Duration._dot_multiplier(self.num_dots)

    @property
    def true_length(self) -> float:
        tuplet_modification = 1 if self.tuplet_ratio is None else float(self.tuplet_ratio[1]) / self.tuplet_ratio[0]
        return self.written_length * tuplet_modification

    def min_denominator(self) -> int:
        return Fraction(self.true_length).limit_denominator().denominator

    @property
    def length_in_divisions(self) -> int:
        return int(round(self.true_length * self.divisions))

    def num_beams(self) -> int:
        """
        The number of beams needed to for a note of this duration.
        """
        return Duration._note_type_to_num_beams[self.note_type]

    @classmethod
    def from_written_length(cls, written_length: float, tuplet_ratio=None, max_dots_allowed=4):
        """
        Constructs a Duration from a written length

        :param written_length: written length in quarter notes
        :param tuplet_ratio: see description in :class:`Duration` constructor
        :param max_dots_allowed: The maximum number of dots to allow
        :raise: ValueError if the given note length is impossible with max_dots_allowed dots or fewer
        """
        try:
            note_type, num_dots = Duration.get_note_type_and_number_of_dots(written_length, max_dots_allowed)
        except ValueError as err:
            raise err
        tuplet_ratio = tuplet_ratio
        return cls(note_type, num_dots, tuplet_ratio)

    @classmethod
    def from_divisor(cls, divisor: int, num_dots=0, tuplet_ratio=None):
        """
        Constructs a Duration from a divisor

        :param divisor: the number of this duration that fit in a whole note. 4 = quarter note, 8 = 8th note, etc.
        :param num_dots: see description in :class:`Duration` constructor
        :param tuplet_ratio: see description in :class:`Duration` constructor
        """
        if not (4.0 / divisor) in Duration._length_to_note_type:
            raise ValueError("Bad divisor")
        return cls.from_written_length(4.0 / divisor * Duration._dot_multiplier(num_dots), tuplet_ratio=tuplet_ratio)

    @classmethod
    def from_string(cls, duration_string: str):
        """
        Parses various string representations into a Duration

        :param duration_string: Can take a variety of forms, e.g. "dotted eighth", "16."
        """
        if duration_string in Duration._note_type_to_length:
            return cls(duration_string)
        elif duration_string.startswith("dotted "):
            if not duration_string[7:] in Duration._note_type_to_length:
                raise ValueError("Bad duration string.")
            return cls(duration_string[7:], 1)

        num_dots = 0
        while duration_string.endswith('.'):
            duration_string = duration_string[:-1]
            num_dots += 1
        try:
            divisor = int(duration_string)
            assert divisor in Duration._valid_divisors
        except (ValueError, AssertionError):
            raise ValueError("Bad duration string.")
        return cls.from_divisor(divisor, num_dots=num_dots)

    @staticmethod
    def get_note_type_and_number_of_dots(length: float, max_dots_allowed: int = 4) -> tuple[str, int]:
        """
        Given a length in quarter notes, get the note type and number of dots.

        :param length: length in quarter notes
        :param max_dots_allowed: maximum number of dots to allow
        :raise: ValueError if the given note length is impossible with max_dots_allowed dots or fewer
        :return: tuple of (note type string, number of dots)
        """
        if length in Duration._length_to_note_type:
            return Duration._length_to_note_type[length], 0
        else:
            dots_multiplier = 1.5
            dots = 1
            while length / dots_multiplier not in Duration._length_to_note_type:
                dots += 1
                dots_multiplier = (2.0 ** (dots + 1) - 1) / 2.0 ** dots
                if dots > max_dots_allowed:
                    raise ValueError("Duration length of {} does not resolve to single note type.".format(length))
            return Duration._length_to_note_type[length / dots_multiplier], dots

    def render(self) -> Sequence[ElementTree.Element]:
        duration_elements = []
        # specify all the duration-related attributes
        duration_el = ElementTree.Element("duration")
        duration_el.text = str(self.length_in_divisions)
        duration_elements.append(duration_el)

        type_el = ElementTree.Element("type")
        type_el.text = self.note_type
        duration_elements.append(type_el)

        for _ in range(self.num_dots):
            duration_elements.append(ElementTree.Element("dot"))

        if self.tuplet_ratio is not None:
            time_modification = ElementTree.Element("time-modification")
            ElementTree.SubElement(time_modification, "actual-notes").text = str(self.tuplet_ratio[0])
            ElementTree.SubElement(time_modification, "normal-notes").text = str(self.tuplet_ratio[1])
            if len(self.tuplet_ratio) > 2:
                if self.tuplet_ratio[2] not in Duration._length_to_note_type:
                    raise ValueError("Tuplet normal note type is not a standard power of two length.")
                ElementTree.SubElement(time_modification, "normal-type").text = \
                    Duration._length_to_note_type[self.tuplet_ratio[2]]
            duration_elements.append(time_modification)
        return tuple(duration_elements)

    def render_to_beat_unit_tags(self) -> Sequence[ElementTree.Element]:
        """
        Renders the beat unit tags needed in metronome directions.
        """
        beat_unit_el = ElementTree.Element("beat-unit")
        beat_unit_el.text = self.note_type
        out = (beat_unit_el, )
        for _ in range(self.num_dots):
            out += (ElementTree.Element("beat-unit-dot"), )
        return out

    def wrap_as_score(self) -> Score:
        return Note("c4", self).wrap_as_score()

    def __repr__(self):
        return "Duration(\"{}\", {}{})".format(
            self.note_type, self.num_dots, ", {}".format(self.tuplet_ratio) if self.tuplet_ratio is not None else ""
        )


class BarRestDuration(DurationalObject):

    """
    Special duration object used for bar rests.

    :param length: Length of bar in quarter notes.
    """

    def __init__(self, length: float):
        self.length = length
        self._divisions = Fraction(length).limit_denominator().denominator

    @property
    def written_length(self) -> float:
        return self.length

    @property
    def divisions(self) -> int:
        return self._divisions

    @divisions.setter
    def divisions(self, value):
        self._divisions = value

    def min_denominator(self) -> int:
        return 1

    @property
    def length_in_divisions(self) -> int:
        return int(round(self.true_length * self.divisions))

    @property
    def true_length(self) -> float:
        return self.length

    def render(self) -> Sequence[ElementTree.Element]:
        duration_el = ElementTree.Element("duration")
        duration_el.text = str(int(round(self.length * self.divisions)))
        return duration_el,

    def wrap_as_score(self) -> Score:
        return BarRest(self).wrap_as_score()


# ---------------------------------------- Note class and all it variations -----------------------------------------


class _XMLNote(DurationalObject):
    """
    Implementation for the xml note element, which includes notes and rests

    :param pitch: a Pitch, or None to indicate a rest, or "bar rest" to indicate that it's a bar rest
    :param duration: a Duration, or just a float representing the bar length in quarter in the case of "bar rest"
    :param ties: "start", "continue", "stop", or None
    :param notations: either a single notation, or a list of notations that will populate the musicXML "notations" tag
    :param articulations: either a single articulations or a list of articulations that will populate the musicXML
        "articulations" tag (itself in the "notations" tag).
    :param notehead: str representing XML notehead type
    :param beams: a dict of { beam_level (int): beam_type } where beam type is one of ("begin", "continue", "end",
        "forward hook", "backward hook")
    :param directions: either a single direction, or a list of directions (e.g. :class:`TextAnnotation`,
        :class:`MetronomeMark`) to populate the musicXML "directions" tag.
    :param stemless: boolean for whether to render the note with no stem
    :param grace: boolean for whether to render the note a grace note with no actual duration, or the string
        "slashed" if it's a slashed grace note
    :param is_chord_member: boolean for whether this is a secondary member of a chord (in which case it contains
        the <chord /> tag
    :param voice: which voice this note belongs to within its given staff
    :param staff: which staff this note belongs to within its given part
    :param velocity: a note velocity which gets passed along and used for playback by many applications
    """

    def __init__(self, pitch, duration, ties=None, notations=(), articulations=(), notehead=None, beams=None,
                 directions=(), stemless=False, grace=False, is_chord_member=False, voice=None, staff=None,
                 velocity=None):

        assert not (grace and pitch is None)  # can't have grace rests
        self.pitch = pitch
        assert isinstance(duration, (Duration, BarRestDuration))
        self.duration = duration
        assert ties in ("start", "continue", "stop", None)
        self.ties = ties
        self.notations = list(notations) if isinstance(notations, (list, tuple)) else [notations]
        self.articulations = list(articulations) if isinstance(articulations, (list, tuple)) else [articulations]
        self.tuplet_bracket = None
        assert isinstance(notehead, (Notehead, str, type(None)))
        self.notehead = Notehead(notehead) if isinstance(notehead, str) else notehead
        self.beams = {} if beams is None else beams
        self.directions = list(directions) if isinstance(directions, (list, tuple)) else [directions]
        self.stemless = stemless
        self.is_grace = grace is not False
        self.slashed = grace == "slashed"
        self.is_chord_member = is_chord_member
        self.voice = voice
        self.staff = staff
        self.velocity = velocity

    @property
    def true_length(self) -> float:
        if self.is_grace:
            return 0
        return self.duration.true_length if isinstance(self.duration, (Duration, BarRestDuration)) else self.duration

    @property
    def written_length(self) -> float:
        return self.duration.written_length if isinstance(self.duration, Duration) else self.duration

    @property
    def length_in_divisions(self) -> int:
        if self.is_grace:
            return 0
        return self.duration.length_in_divisions

    @property
    def divisions(self) -> int:
        return self.duration.divisions

    @divisions.setter
    def divisions(self, value):
        self.duration.divisions = value

    def min_denominator(self) -> int:
        if self.is_grace:
            return 1
        return Fraction(self.duration.true_length).limit_denominator().denominator

    def num_beams(self) -> int:
        """
        Returns the number of beams needed to represent this note's duration.
        """
        return 0 if self.pitch is None else self.duration.num_beams()

    def render(self) -> Sequence[ElementTree.Element]:
        note_element = ElementTree.Element(
            "note", {} if self.velocity is None else {"dynamics": "{:.2f}".format(self.velocity / 90 * 100)}
        )

        # ------------ set pitch and duration attributes ------------

        if self.pitch == "bar rest":
            # bar rest; in this case the duration is just a float, since it looks like a whole note regardless
            note_element.append(ElementTree.Element("rest"))
            note_element.extend(self.duration.render())
        else:
            if self.is_grace:
                ElementTree.SubElement(note_element, "grace", {"slash": "yes"} if self.slashed else {})

            # a note or rest with explicit written duration
            if self.pitch is None:
                # normal rest
                note_element.append(ElementTree.Element("rest"))
            else:
                # pitched note
                assert isinstance(self.pitch, Pitch)

                if self.is_chord_member:
                    note_element.append(ElementTree.Element("chord"))
                note_element.extend(self.pitch.render())

            duration_elements = self.duration.render()

            if not self.is_grace:
                # this is the actual duration tag; gracenotes don't have them
                note_element.append(duration_elements[0])

            # for some reason, the tie element and the voice are generally sandwiched in here

            if self.ties is not None:
                if self.ties.lower() == "start" or self.ties.lower() == "continue":
                    note_element.append(ElementTree.Element("tie", {"type": "start"}))
                if self.ties.lower() == "stop" or self.ties.lower() == "continue":
                    note_element.append(ElementTree.Element("tie", {"type": "stop"}))

            if self.voice is not None:
                ElementTree.SubElement(note_element, "voice").text = str(self.voice)

            # these are the note type and any dot tags
            note_element.extend(duration_elements[1:])

        # --------------- stem / notehead -------------

        if self.stemless:
            ElementTree.SubElement(note_element, "stem").text = "none"

        if self.notehead is not None:
            note_element.extend(self.notehead.render())

        # ------------------ set staff ----------------

        if self.staff is not None:
            ElementTree.SubElement(note_element, "staff").text = str(self.staff)

        # ---------------- set attributes that apply to notes only ----------------

        if self.pitch is not None:
            if self.ties is not None:
                if self.ties.lower() == "start" or self.ties.lower() == "continue":
                    self.notations.append(ElementTree.Element("tied", {"type": "start"}))
                if self.ties.lower() == "stop" or self.ties.lower() == "continue":
                    self.notations.append(ElementTree.Element("tied", {"type": "stop"}))
            for beam_num in self.beams:
                beam_text = self.beams[beam_num]
                beam_el = ElementTree.Element("beam", {"number": str(beam_num)})
                beam_el.text = beam_text
                note_element.append(beam_el)

        # ------------------ add any notations and articulations ----------------

        if len(self.notations) + len(self.articulations) > 0 or self.tuplet_bracket is not None:
            # there is either a notation or an articulation, so we'll add a notations tag
            notations_el = ElementTree.Element("notations")
            for notation in self.notations:
                if isinstance(notation, ElementTree.Element):
                    # if it's already an element, just append it directly
                    notations_el.append(notation)
                elif isinstance(notation, Notation):
                    # otherwise, if it's a Notation object, then render and append it
                    notations_el.extend(notation.render())
                elif isinstance(notation, str):
                    # otherwise, if it's a string, just make a simple element out of it and append
                    notations_el.append(ElementTree.Element(notation))
                else:
                    logging.warning("Notation {} not understood".format(notation))

            if self.tuplet_bracket in ("start", "both"):
                notations_el.append(ElementTree.Element("tuplet", {"type": "start"}))
            if self.tuplet_bracket in ("stop", "both"):
                notations_el.append(ElementTree.Element("tuplet", {"type": "stop"}))

            if len(self.articulations) > 0:
                articulations_el = ElementTree.SubElement(notations_el, "articulations")
                for articulation in self.articulations:
                    if isinstance(articulation, ElementTree.Element):
                        articulations_el.append(articulation)
                    else:
                        articulations_el.append(ElementTree.Element(articulation))
            note_element.append(notations_el)

        # place any text annotations before the note so that they show up at the same time as the note start
        return sum((direction.render() for direction in self.directions), ()) + (note_element,)

    def wrap_as_score(self) -> Score:
        if isinstance(self, BarRest):
            duration_as_fraction = Fraction(self.true_length).limit_denominator()
            assert _is_power_of_two(duration_as_fraction.denominator)
            time_signature = (duration_as_fraction.numerator, duration_as_fraction.denominator * 4)
            return Measure([self], time_signature=time_signature).wrap_as_score()
        else:
            measure_length = 4 if self.true_length <= 4 else int(self.true_length) + 1
            return Measure(pad_with_rests(self, measure_length), (measure_length, 4)).wrap_as_score()


class Note(_XMLNote):

    """
    Class representing a single, pitched note.

    :param pitch: either a Pitch object or a string to parse as a pitch (see :func:`Pitch.from_string`)
    :param duration: either a :class:`Duration` object, a string to parse as a duration (see
        :func:`Duration.from_string`), or a number of quarter notes.
    :param ties: one of "start", "continue", "stop", None
    :param notations: Either a single notation, or a list of notations that will populate the musicXML "notations" tag.
        Each is either a :class:`Notation` object, an :class:`ElementTree.Element` object, or a string that will be
        converted into an Element object.
    :param articulations: Either a single articulations or a list of articulations that will populate the musicXML
        "articulations" tag (itself in the "notations" tag). Each is either an :class:`ElementTree.Element` object,
        or a string that will be converted into an Element object.
    :param notehead: a :class:`Notehead` or a string representing XML notehead type. Note that the default of None
        represents an ordinary notehead.
    :param directions: either a single direction, or a list of directions (e.g. :class:`TextAnnotation`,
        :class:`MetronomeMark`) to populate the musicXML "directions" tag.
    :param stemless: boolean for whether to render the note with no stem.
    :param velocity: a note velocity (0-127) which gets passed along and used for playback by many applications
    """

    def __init__(self, pitch: Pitch | str, duration: Duration | str | float, ties: str = None,
                 notations=(), articulations=(), notehead: Notehead | str = None,
                 directions: Sequence[Direction] = (), stemless: bool = False, velocity: int = None):

        if isinstance(pitch, str):
            pitch = Pitch.from_string(pitch)
        assert isinstance(pitch, Pitch)

        if isinstance(duration, str):
            duration = Duration.from_string(duration)
        elif isinstance(duration, Real):
            duration = Duration.from_written_length(duration)

        if ties not in ("start", "continue", "stop", None):
            raise ValueError('Ties argument must be one of ("start", "continue", "stop", None)')

        assert isinstance(duration, Duration)
        super().__init__(pitch, duration, ties=ties, notations=notations, articulations=articulations,
                         notehead=notehead, directions=directions, stemless=stemless, velocity=velocity)

    @property
    def starts_tie(self):
        """
        Whether or not this note starts a tie.
        """
        return self.ties in ("start", "continue")

    @starts_tie.setter
    def starts_tie(self, value):
        if value:
            # setting it to start a tie if it isn't already
            self.ties = "start" if self.ties in ("start", None) else "continue"
        else:
            # setting it to not start a tie
            self.ties = None if self.ties in ("start", None) else "stop"

    @property
    def stops_tie(self):
        """
        Whether or not this note ends a tie.
        """
        return self.ties in ("stop", "continue")

    @stops_tie.setter
    def stops_tie(self, value):
        if value:
            # setting it to stop a tie if it isn't already
            self.ties = "stop" if self.ties in ("stop", None) else "continue"
        else:
            # setting it to not stop a tie
            self.ties = None if self.ties in ("stop", None) else "start"

    def __repr__(self):
        return "Note({}, {}{}{}{}{})".format(
            self.pitch, self.duration,
            ", ties=\"{}\"".format(self.ties) if self.ties is not None else "",
            ", notations={}".format(self.notations) if len(self.notations) > 0 else "",
            ", articulations={}".format(self.articulations) if len(self.articulations) > 0 else "",
            ", notehead=\"{}\"".format(self.notehead) if self.notehead is not None else "",
            ", directions=\"{}\"".format(self.directions) if self.directions is not None else "",
            ", stemless=\"{}\"".format(self.stemless) if self.stemless is not None else ""
        )


class Notehead(MusicXMLComponent):

    """
    Class representing a notehead type.

    :param notehead_name: accepts any of the MusicXML notehead types, possibly preceded by "filled" or "open". So
        "filled triangle" will create the triangle notehead type with the filled flag set to true.
    :param filled: whether or not the notehead is filled
    """

    #: List of valid MusicXML notehead types
    valid_xml_types = ["normal", "diamond", "triangle", "slash", "cross", "x", "circle-x", "inverted triangle",
                       "square", "arrow down", "arrow up", "circled", "slashed", "back slashed", "cluster",
                       "circle dot", "left triangle", "rectangle", "do", "re", "mi", "fa", "fa up", "so",
                       "la", "ti",  "none"]

    def __init__(self, notehead_name: str, filled: bool = None):
        notehead_name = notehead_name.strip().lower()
        if "filled " in notehead_name:
            filled = "yes"
            notehead_name = notehead_name.replace("filled ", "")
        elif "open " in notehead_name:
            filled = "no"
            notehead_name = notehead_name.replace("open ", "")
        assert notehead_name in Notehead.valid_xml_types, "Notehead \"{}\" not understood".format(notehead_name)
        self.notehead_name = notehead_name
        assert filled in (None, "yes", "no", True, False)
        self.filled = "yes" if filled in ("yes", True) else "no" if filled in ("no", False) else None

    def render(self) -> Sequence[ElementTree.Element]:
        notehead_el = ElementTree.Element("notehead", {"filled": self.filled} if self.filled is not None else {})
        notehead_el.text = self.notehead_name
        return notehead_el,

    def wrap_as_score(self) -> Score:
        return Note("c5", 1, notehead=self).wrap_as_score()

    def __repr__(self):
        return "Notehead({}{})".format(self.notehead_name,
                                       ", {}".format(self.filled) if self.filled is not None else "")


class Rest(_XMLNote):

    """
    Class representing a notated rest.

    :param duration: either a :class:`Duration` object, a string to parse as a duration (see
        :func:`Duration.from_string`), or a number of quarter notes.
    :param notations: see :class:`Note`
    :param directions: see :class:`Note`
    """

    def __init__(self, duration: Duration | str | float,
                 notations: Sequence[Notation | str | ElementTree.Element] = (),
                 directions: Sequence[Direction] = ()):
        if isinstance(duration, str):
            duration = Duration.from_string(duration)
        elif isinstance(duration, Real):
            duration = Duration.from_written_length(duration)

        assert isinstance(duration, Duration)
        super().__init__(None, duration, notations=notations, directions=directions)

    def __repr__(self):
        return "Rest({}{})".format(
            self.duration,
            ", notations={}".format(self.notations) if len(self.notations) > 0 else "",
            ", directions=\"{}\"".format(self.directions) if self.directions is not None else "",
        )


class BarRest(_XMLNote):

    """
    Class representing a bar rest.

    :param bar_length: length of this bar, either as a number, :class:`Duration`, or :class:`BarRestDuration`. Bar rest
        durations are a little special, since they can be lengths (like 2.5 beats) that are otherwise note notatable
        in a single note or rest object.
    :param directions: see :class:`Note`
    """

    def __init__(self, bar_length: float | Duration | BarRestDuration, directions: Sequence[Direction] = ()):
        duration = BarRestDuration(bar_length) if isinstance(bar_length, Real) \
            else BarRestDuration(bar_length.true_length) if isinstance(bar_length, Duration) else bar_length
        super().__init__("bar rest", duration, directions=directions)

    def __repr__(self):
        return "BarRest({}{})".format(
            self.duration,
            ", directions=\"{}\"".format(self.directions) if self.directions is not None else "",
        )


# -------------------------------------- Chord (wrapper around multiple Notes) ---------------------------------------


class Chord(DurationalObject):

    """
    Class representing a chord. When rendered to MusicXML, a chord is represented by sequential <note> tags, with all
    but the first carrying a </chord> tag.

    :param pitches: either a Pitch object or a string to parse as a pitch (see :func:`Pitch.from_string`)
    :param duration: either a :class:`Duration` object, a string to parse as a duration (see
        :func:`Duration.from_string`), or a number of quarter notes.
    :param ties: either one of "start", "continue", "stop", or None (in which case each note of the chord follows
        that tie indication), or a list of such tie indications, one for each note.
    :param notations: a notation or list of notations like that passed to a :class:`Note` object. Since this is a
        chord, it is also possible to include a :class:`StartMultiGliss` or :class:`StopMultiGliss` object.
    :param articulations: an articulation or list of articulations like that passed to a :class:`Note` object.
    :param noteheads: Either a single notehead (in the form of a :class:`Notehead` or string representing XML
        notehead type) or a list of such noteheads, one for each pitch. The former results in all noteheads in the
        chord being the same; the latter results in different noteheads for each chord member. Note that the default
        of None represents ordinary noteheads.
    :param directions: a direction or list of directions like that passed to a :class:`Note` object.
    :param stemless: boolean for whether to render the chord with no stem.
    :param velocity: a note velocity (0-127) which gets passed along and used for playback by many applications
    """

    def __init__(self, pitches: Sequence[Pitch | str], duration: Duration | str | float,
                 ties: str | Sequence[str | None] = None, notations=(), articulations=(),
                 noteheads=None, directions=(), stemless: bool = False, velocity: int = None):
        assert isinstance(pitches, (list, tuple)) and len(pitches) > 1, "Chord should have multiple notes."
        pitches = [Pitch.from_string(pitch) if isinstance(pitch, str) else pitch for pitch in pitches]
        assert all(isinstance(pitch, Pitch) for pitch in pitches)

        if isinstance(duration, str):
            duration = Duration.from_string(duration)
        elif isinstance(duration, Real):
            duration = Duration.from_written_length(duration)

        assert ties in ("start", "continue", "stop", None) or \
               isinstance(ties, (list, tuple)) and len(ties) == len(pitches) and \
               all(x in ("start", "continue", "stop", None) for x in ties)

        note_notations = [[] for _ in range(len(pitches))]
        for notation in (notations if isinstance(notations, (list, tuple)) else (notations, )):
            if isinstance(notation, MultiGliss):
                for i, gliss_notation in enumerate(notation.render()):
                    if i < len(note_notations) and gliss_notation is not None:
                        note_notations[i].append(gliss_notation)
            else:
                note_notations[0].append(notation)

        self.notes = tuple(
            Note(pitch, duration,
                 ties=ties if not isinstance(ties, (list, tuple)) else ties[i],
                 notations=note_notations[i],
                 articulations=articulations if i == 0 else (),
                 notehead=noteheads if isinstance(noteheads, str) else noteheads[i] if noteheads is not None else None,
                 directions=directions if i == 0 else (),
                 stemless=stemless,
                 velocity=velocity)
            for i, pitch in enumerate(pitches)
        )

        for note in self.notes[1:]:
            note.is_chord_member = True

    @property
    def pitches(self):
        """
        Tuple of the pitches of the notes in this chord.
        """
        return tuple(note.pitch for note in self.notes)

    @pitches.setter
    def pitches(self, value):
        assert isinstance(value, tuple) and len(value) == len(self.notes) \
               and all(isinstance(x, Pitch) for x in value)
        for note, pitch in zip(self.notes, value):
            note.pitch = pitch

    def num_beams(self) -> int:
        """
        Returns the number of beams needed to represent this duration.
        """
        return self.notes[0].num_beams()

    @property
    def duration(self):
        """
        The :class:`Duration` of this chord.
        """
        return self.notes[0].duration

    @property
    def true_length(self) -> float:
        return self.notes[0].true_length

    @property
    def written_length(self) -> float:
        return self.notes[0].written_length

    @property
    def length_in_divisions(self) -> int:
        return self.notes[0].length_in_divisions

    @property
    def notations(self) -> Sequence[Notation]:
        """
        Notations attached to this chord.
        """
        return self.notes[0].notations

    @property
    def articulations(self) :
        """
        Articulations attached to this chord.
        """
        return self.notes[0].articulations

    @property
    def directions(self):
        """
        Directions attached to this chord.
        """
        return self.notes[0].directions

    @property
    def divisions(self) -> int:
        return self.notes[0].divisions

    @divisions.setter
    def divisions(self, value):
        for note in self.notes:
            note.divisions = value

    @property
    def voice(self):
        """
        Which voice this chord resides in.
        """
        return self.notes[0].voice

    @voice.setter
    def voice(self, value):
        for note in self.notes:
            note.voice = value

    @property
    def beams(self):
        """
        Dictionary describing the notation of all the beams for this Chord.
        """
        return self.notes[0].beams

    @beams.setter
    def beams(self, value):
        self.notes[0].beams = value

    @property
    def ties(self):
        """
        Either a string representing the tie state of all the notes, if all notes have the same tie state, or a tuple
        representing the tie state of each note individually.
        """
        if all(note.ties == self.notes[0].ties for note in self.notes[1:]):
            return self.notes[0].ties
        else:
            return tuple(note.ties for note in self.notes)

    @ties.setter
    def ties(self, value):
        if isinstance(value, (list, tuple)):
            assert len(value) == len(self.notes)
            for i, note in enumerate(self.notes):
                note.ties = value[i]
        else:
            assert value in ("start", "continue", "stop", None)
            for note in self.notes:
                note.ties = value

    @property
    def tuplet_bracket(self):
        """
        Whether or not this chord starts or stops a tuplet bracket.
        """
        return self.notes[0].tuplet_bracket

    @tuplet_bracket.setter
    def tuplet_bracket(self, value):
        self.notes[0].tuplet_bracket = value

    def min_denominator(self) -> int:
        return self.notes[0].min_denominator()

    def render(self) -> Sequence[ElementTree.Element]:
        return sum((note.render() for note in self.notes), ())

    def wrap_as_score(self) -> Score:
        measure_length = 4 if self.true_length <= 4 else int(self.true_length) + 1
        return Measure(pad_with_rests(self, measure_length), (measure_length, 4)).wrap_as_score()

    def __repr__(self):
        noteheads = None if all(n.notehead is None for n in self.notes) else tuple(n.notehead for n in self.notes)
        return "Chord({}, {}{}{}{}{})".format(
            tuple(note.pitch for note in self.notes), self.duration,
            ", ties=\"{}\"".format(self.ties) if self.ties is not None else "",
            ", notations={}".format(self.notations) if len(self.notations) > 0 else "",
            ", articulations={}".format(self.articulations) if len(self.articulations) > 0 else "",
            ", noteheads={}".format(noteheads) if noteheads is not None else "",
            ", directions=\"{}\"".format(self.directions) if self.directions is not None else "",
            ", stemless=True" if self.notes[0].stemless else ""
        )


# -------------------------------------------- Grace Notes and Chords ---------------------------------------------


class GraceNote(Note):
    """
    Subclass of :class:`Note` representing a durationless grace note.

    :param pitch: see :class:`Note`
    :param duration: see :class:`Note`. This will be the displayed duration of the note, even though it takes up no
        metric space in the bar.
    :param ties: see :class:`Note`
    :param notations: see :class:`Note`
    :param articulations: see :class:`Note`
    :param notations: see :class:`Note`
    :param articulations: see :class:`Note`
    :param notehead: see :class:`Note`
    :param directions: see :class:`Note`
    :param stemless: see :class:`Note`
    :param slashed: whether or not this grace note is rendered with a slash.
    :param velocity: a note velocity (0-127) which gets passed along and used for playback by many applications
    """
    def __init__(self, pitch: Pitch | str, duration: Duration | str | float, ties: str = None,
                 notations=(), articulations=(), notehead: Notehead | str = None,
                 directions: Sequence[Direction] = (), stemless: bool = False, slashed=False, velocity: int = None):
        super().__init__(pitch,  duration, ties=ties, notations=notations, articulations=articulations,
                         notehead=notehead, directions=directions, stemless=stemless, velocity=velocity)
        self.is_grace = True
        self.slashed = slashed

    def __repr__(self):
        return "Grace" + super().__repr__()


class GraceChord(Chord):
    """
    Subclass of :class:`Chord` representing a durationless grace chord.

    :param pitches: see :class:`Chord`
    :param duration: see :class:`Chord`. This will be the displayed duration of the chord, even though it takes up no
        metric space in the bar.
    :param ties: see :class:`Chord`
    :param notations: see :class:`Chord`
    :param articulations: see :class:`Chord`
    :param notations: see :class:`Chord`
    :param articulations: see :class:`Chord`
    :param noteheads: see :class:`Chord`
    :param directions: see :class:`Chord`
    :param stemless: see :class:`Chord`
    :param slashed: whether or not this grace chord is rendered with a slash.
    :param velocity: a note velocity (0-127) which gets passed along and used for playback by many applications
    """

    def __init__(self, pitches: Sequence[Pitch | str], duration: Duration | str | float,
                 ties: str | Sequence[str | None] = None, notations=(), articulations=(),
                 noteheads=None, directions=(), stemless: bool = False, slashed=False, velocity: int = None):
        super().__init__(pitches, duration, ties=ties, notations=notations, articulations=articulations,
                         noteheads=noteheads, directions=directions, stemless=stemless, velocity=velocity)
        for note in self.notes:
            note.is_grace = True
            note.slashed = slashed

    def __repr__(self):
        return "Grace" + super().__repr__()


# -------------------------------------------------- Note Groups ------------------------------------------------


class BeamedGroup(DurationalObject, MusicXMLContainer):

    """
    Represents a group of notes/chords/rests joined under a single beam.

    :param contents: a list of notes, chords and rests contained in this group.
    """

    def __init__(self, contents: Sequence[MusicXMLComponent] = None):
        super().__init__(contents=contents, allowed_types=(Note, Chord, Rest))

    def render_contents_beaming(self) -> None:
        """
        Works out all the beaming for the contents of this group. (Sets the "beams" attribute for every element.)
        """
        for beam_depth in range(1, max(leaf.num_beams() for leaf in self.contents) + 1):
            leaf_start_time = 0
            for i, leaf in enumerate(self.contents):
                last_note_active = i > 0 and self.contents[i-1].num_beams() >= beam_depth
                this_note_active = leaf.num_beams() >= beam_depth
                next_note_active = i < len(self.contents) - 1 and self.contents[i+1].num_beams() >= beam_depth

                if this_note_active:
                    if last_note_active and next_note_active:
                        leaf.beams[beam_depth] = "continue"
                    elif last_note_active:
                        leaf.beams[beam_depth] = "end"
                    elif next_note_active:
                        leaf.beams[beam_depth] = "begin"
                    else:
                        if int(round(leaf_start_time / 0.5 ** leaf.num_beams())) % 2 == 0:
                            leaf.beams[beam_depth] = "forward hook"
                        else:
                            leaf.beams[beam_depth] = "backward hook"

                leaf_start_time += leaf.written_length

        for leaf in self.contents:
            if all("hook" in beam_value for beam_value in leaf.beams.values()):
                leaf.beams = {}

    @property
    def divisions(self) -> int:
        return self.contents[0].divisions

    @divisions.setter
    def divisions(self, value):
        for leaf in self.contents:
            leaf.divisions = value

    @property
    def true_length(self) -> float:
        return sum(leaf.true_length for leaf in self.contents)

    @property
    def written_length(self) -> float:
        return sum(leaf.written_length for leaf in self.contents)

    @property
    def length_in_divisions(self) -> int:
        return sum(leaf.length_in_divisions for leaf in self.contents)

    def min_denominator(self) -> int:
        return _least_common_multiple(*[n.min_denominator() for n in self.contents])

    def render(self) -> Sequence[ElementTree.Element]:
        self.render_contents_beaming()
        return sum((leaf.render() for leaf in self.contents), ())

    def wrap_as_score(self) -> Score:
        measure_length = 4 if self.true_length <= 4 else int(math.ceil(self.true_length))
        return Measure(pad_with_rests(self, measure_length), (measure_length, 4)).wrap_as_score()


class Tuplet(BeamedGroup):
    """
    Represents a tuplet; same as a BeamedGroup, but with a particular time ratio associated with it.

    :param contents: a list of notes, chords and rests contained in this group.
    :param ratio: a tuple representing the tuplet ratio (as described in :class:`Duration`)
    """

    def __init__(self, contents: Sequence[MusicXMLComponent], ratio: tuple):
        super().__init__(contents)
        self.ratio = ratio
        self._set_tuplet_info_for_contents(ratio)

    def _set_tuplet_info_for_contents(self, ratio):
        self.contents[0].tuplet_bracket = "start"
        # it would be stupid to have a tuplet that has only one note in it, but we'll cover that case anyway
        self.contents[-1].tuplet_bracket = "stop" if len(self.contents) > 0 else "both"

        for element in self.contents:
            if isinstance(element, Chord):
                for note in element.notes:
                    note.duration.tuplet_ratio = ratio
            else:
                element.duration.tuplet_ratio = ratio

    def min_denominator(self) -> int:
        self._set_tuplet_info_for_contents(self.ratio)
        return _least_common_multiple(*[n.min_denominator() for n in self.contents])

    def render(self) -> Sequence[ElementTree.Element]:
        self._set_tuplet_info_for_contents(self.ratio)
        return super().render()


# -------------------------------------------- Clef, KeySignature and Measure -----------------------------------------


class Clef(MusicXMLComponent):

    """
    Class representing a musical clef

    :param sign: Whether it is a G, F, or C clef
    :param line: Which line the clef is centered on
    :param octaves_transposition: How many octaves up or down the clef transposes
    """

    #: Dictionary mapping standard clef names to tuple of (clef letter type, clef line)
    clef_name_to_letter_and_line = {
        "treble": ("G", 2),
        "bass": ("F", 4),
        "alto": ("C", 3),
        "tenor": ("C", 4),
        "soprano": ("C", 1),
        "mezzo-soprano": ("C", 2),
        "baritone": ("F", 3)
    }

    def __init__(self, sign: str, line: int, octaves_transposition: int = 0):
        sign = sign.upper()
        if sign not in ("C", "G", "F"):
            raise ValueError("Clef sign not understood; must be \"C\", \"G\", or \"F\".")
        self.sign = sign
        self.line = str(line)
        self.octaves_transposition = octaves_transposition

    @classmethod
    def from_string(cls, clef_string: str, octaves_transposition: int = 0):
        """
        Constructs a clef from one of the standard names, e.g. treble, bass, alto

        :param clef_string: name of the clef
        :param octaves_transposition: octaves of transposition up or down to be applied to the clef
        """
        if clef_string in Clef.clef_name_to_letter_and_line:
            return cls(*Clef.clef_name_to_letter_and_line[clef_string], octaves_transposition=octaves_transposition)
        else:
            raise ValueError("Clef name not understood.")

    def render(self) -> Sequence[ElementTree.Element]:
        clef_element = ElementTree.Element("clef")
        ElementTree.SubElement(clef_element, "sign").text = self.sign
        ElementTree.SubElement(clef_element, "line").text = self.line
        if self.octaves_transposition != 0:
            ElementTree.SubElement(clef_element, "clef-octave-change").text = str(self.octaves_transposition)
        return clef_element,

    def wrap_as_score(self) -> Score:
        return Measure([BarRest(4)], time_signature=(4, 4), clef=self).wrap_as_score()


class Transpose(MusicXMLComponent):
    """
    Class representing transposition. Used for instruments like guitar or bass
    which transpose by an octave down.

    The <transpose> element represents what must be added to a written pitch to
    get a correct sounding pitch. It is used for encoding parts for transposing
    instruments that are in written vs. concert pitch.

    :param chromatic: Chromatic transposition
    :param diatonic: Diatonic transposition (optional)
    :param octave: Octave change (optional)
    """
    def __init__(self, chromatic: int, diatonic: int = None, octave: int = None):
        self.chromatic = chromatic
        self.diatonic = diatonic
        self.octave = octave

    def render(self) -> Sequence[ElementTree.Element]:
        transpose_element = ElementTree.Element("transpose")
        if self.diatonic is not None:
            ElementTree.SubElement(transpose_element, "diatonic").text = str(self.diatonic)
        ElementTree.SubElement(transpose_element, "chromatic").text = str(self.chromatic)
        if self.octave is not None:
            ElementTree.SubElement(transpose_element, "octave-change").text = str(self.octave)
        return transpose_element,

    def wrap_as_score(self) -> Score:
        return Measure([BarRest(4)], time_signature=(4, 4), transpose=self).wrap_as_score()


class KeySignature(MusicXMLComponent):

    """
    Abstract base class for traditional and non-traditional key signatures.
    Also contains the "parse" method for parsing a string into a key signature.
    """

    _name_to_fifths = {
        "c": 0,
        "g": 1,
        "d": 2,
        "a": 3,
        "e": 4,
        "b": 5,
        "f#": 6,
        "fs": 6,
        "c#": 7,
        "cs": 7,
        "f": -1,
        "bb": -2,
        "bf": -2,
        "eb": -3,
        "ef": -3,
        "ab": -4,
        "af": -4,
        "db": -5,
        "df": -5,
        "gb": -6,
        "gf": -6,
        "cb": -7,
        "cf": -7,
    }

    _alteration_to_number = {
        "b": -1,
        "f": -1,
        "s": 1,
        "#": 1,
        "x": 2,
        "bb": -2
    }

    mode_fifth_alterations = {
        "ionian": 0,
        "major": 0,
        "dorian": -2,
        "phrygian": -4,
        "lydian": 1,
        "mixolydian": -1,
        "aeolian": -3,
        "minor": -3,
        "locrian": -4,
        None: 0
    }

    @staticmethod
    def parse(interpretable_as_key_signature: KeySignature | int | str) -> KeySignature:
        """
        Parses several kinds of input into a TraditionalKeySignature or NonTraditionalKeySignature object.

        :param interpretable_as_key_signature: either a KeySignature object, an integer representing the number of
            sharps (or flats if negative), or a string to be parsed into a key signature, such as "G major",
            "F# lydian", or "C#, Ab" (which produces a non-traditional key-signature), etc.
        """
        if isinstance(interpretable_as_key_signature, KeySignature):
            return interpretable_as_key_signature
        elif isinstance(interpretable_as_key_signature, int):
            return TraditionalKeySignature(interpretable_as_key_signature)
        elif isinstance(interpretable_as_key_signature, str):
            interpretable_as_key_signature = interpretable_as_key_signature.lower().replace("-", "").\
                replace("sharp", "s").replace("flat", "f")
            if "," in interpretable_as_key_signature:
                interpretable_as_key_signature = interpretable_as_key_signature.replace(" ", "")
                non_traditional_key_signature = NonTraditionalKeySignature()
                for alteration_str in interpretable_as_key_signature.split(","):
                    non_traditional_key_signature.add_alteration(
                        alteration_str[0].upper(), KeySignature._alteration_to_number[alteration_str[1:]]
                    )
                return non_traditional_key_signature
            else:
                key_center, *mode_info = interpretable_as_key_signature.split(" ")
                mode = None if len(mode_info) == 0 else mode_info[0]
                if mode not in KeySignature.mode_fifth_alterations:
                    raise ValueError("Invalid mode for key signature.")
                return TraditionalKeySignature(
                    fifths=KeySignature._name_to_fifths[key_center] + KeySignature.mode_fifth_alterations[mode],
                    mode=mode
                )
        else:
            raise ValueError("Key signature not understood.")

    @abstractmethod
    def render(self) -> Sequence[ElementTree.Element]:
        pass

    def wrap_as_score(self) -> Score:
        return Measure([BarRest(4)], time_signature=(4, 4), clef="treble", key=self).wrap_as_score()


class TraditionalKeySignature(KeySignature):

    """
    A traditional key signature. See https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/key/.

    :param fifths: e.g. -2 for Bb major/G minor, or 3 for A major/F# minor
    :param mode: one of the standard modes, e.g. "major", "minor", "lydian"
    :param cancel: the fifths of the previous key signature being canceled out
    """

    def __init__(self, fifths, mode=None, cancel=None):
        self.fifths = fifths
        self.mode = mode
        self.cancel = cancel

    def render(self) -> Sequence[ElementTree.Element]:
        key_el = ElementTree.Element("key")
        if self.cancel is not None:
            ElementTree.SubElement(key_el, "cancel").text = str(self.cancel)
        ElementTree.SubElement(key_el, "fifths").text = str(self.fifths)
        if self.mode is not None:
            ElementTree.SubElement(key_el, "mode").text = str(self.mode)
        return key_el,


class NonTraditionalKeySignature(KeySignature):

    def __init__(self, *step_alteration_tuples):
        """
        A non-traditional key signature. See https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/key/.

        :param step_alteration_tuples: a list of (step, alteration) or (step, alteration, accidental) tuples, each of
            which corresponds to a set of <key-step>, <key-alter>, and optionally <key-accidental> tags.
        """
        self.step_alteration_tuples = list(step_alteration_tuples)

    def add_alteration(self, step, alteration, accidental=None):
        self.step_alteration_tuples.append((step, alteration, accidental) if accidental is not None
                                           else (step, alteration))

    def render(self) -> Sequence[ElementTree.Element]:
        key_el = ElementTree.Element("key")
        for step, alteration, *accidental in self.step_alteration_tuples:
            ElementTree.SubElement(key_el, "key-step").text = str(step)
            ElementTree.SubElement(key_el, "key-alter").text = str(alteration)
            if len(accidental) > 0:
                ElementTree.SubElement(key_el, "key-accidental").text = str(accidental[0])
        return key_el,


class Measure(MusicXMLComponent, MusicXMLContainer):
    """
    Class representing a measure of music, perhaps with multiple voices.

    :param contents: Either a list of Notes / Chords / Rests / Tuplets / BeamedGroups or a list of voices, each of
        which is a list of Notes / Chords / Rests / Tuplets / BeamedGroups.
    :param time_signature: in tuple form, e.g. (3, 4) for "3/4"
    :param key: either a TraditionalKeySignature or NonTraditionalKeySignature object, an integer representing the
        number of sharps (or flats if negative), or a string to be parsed into a key signature, such as "G major",
        "F# lydian", etc.
    :param clef: either None (for no clef), a Clef object, a string (like "treble"), or a tuple like ("G", 2) to
        represent the clef letter, the line it lands, and an optional octave transposition as the third parameter
    :param barline: either None, which means there will be a regular barline, "double", "end", or any of the barline
        names used in the MusicXML standard.
    :param staves: for multi-part music, like piano music
    :param number: which number in the score. Will be set automatically by the containing Part
    :param directions_with_displacements: this is a way of placing directions at arbitrary positions in the bar.
        It takes a list of tuples of (direction, position in the bar), where position in the bar is a floating point
        number of quarter notes.
    """

    _barline_xml_names = {
        "double": "light-light",
        "end": "light-heavy",
        "regular": "regular",
        "dotted": "dotted",
        "dashed": "dashed",
        "heavy": "heavy",
        "light-light": "light-light",
        "light-heavy": "light-heavy",
        "heavy-light": "heavy-light",
        "heavy-heavy": "heavy-heavy",
        "tick": "tick",
        "short": "short",
        "none": "none"
    }

    def __init__(self, contents: Sequence[DurationalObject] | Sequence[Sequence[DurationalObject]] = None,
                 time_signature: tuple = None, key: KeySignature | str | int = None,
                 clef: Clef | str | tuple = None, barline: str = None,
                 staves: str = None, number: int = 1,
                 directions_with_displacements: Sequence[tuple[Direction, float]] = (),
                 transpose: Transpose | None = None):
        super().__init__(contents=contents, allowed_types=(Note, Rest, Chord, BarRest, BeamedGroup,
                                                           Tuplet, type(None), Sequence, DurationalObject))
        assert hasattr(self.contents, '__len__') and all(
            isinstance(x, (Note, Rest, Chord, BarRest, BeamedGroup, Tuplet, type(None), DurationalObject)) or
            hasattr(x, '__len__') and all(isinstance(y, (Note, Rest, Chord, BarRest, BeamedGroup, Tuplet, DurationalObject)) for y in x)
            for x in self.contents
        )

        self.number = number
        self.time_signature = time_signature
        self.key = key
        assert isinstance(clef, (type(None), Clef, str, tuple)), "Clef not understood."
        self.clef = clef if isinstance(clef, (type(None), Clef)) \
            else Clef.from_string(clef) if isinstance(clef, str) \
            else Clef(*clef)
        assert barline is None or isinstance(barline, str) \
               and barline.lower() in Measure._barline_xml_names, "Barline type not understood"
        self.barline = barline
        self.staves = staves
        self.transpose = transpose

        self.directions_with_displacements = directions_with_displacements

    @property
    def voices(self) -> tuple[Sequence[MusicXMLComponent]]:
        """
        Returns a tuple of the voices in this Measure
        """
        return (self.contents, ) if not isinstance(self.contents[0], (tuple, list, type(None))) else self.contents

    def iter_leaves(self, which_voices=None) -> Iterator[tuple[Note | Chord | Rest, float]]:
        """
        Iterates through the Notes/Chords/Rests in this measure, expanding out any tuplets or beam groups. The
        notes/chords/rests draw from the specified voices.

        :param which_voices: List of voices to return notes from (numbered 0, 1, 2, 3). The default value of None
            returns notes from all voices.

        :return tuples of (Note/Chord/Rest, beat within measure)
        """
        # make a copy of the voice list, but with any beam groups or tuplets unraveled
        voice_list_copy = []
        for i, v in enumerate(self.voices):
            if (which_voices is None or i in which_voices) and v is not None:
                voice_list_copy.append([])
                for note_or_group in v:
                    if isinstance(note_or_group, (BeamedGroup, Tuplet)):
                        voice_list_copy[-1].extend(note_or_group.contents)
                    else:
                        voice_list_copy[-1].append(note_or_group)
        # now iteratively yield the next earliest note (choosing the shortest in the case of a tie)
        next_note_beats = [0] * len(voice_list_copy)
        while len(voice_list_copy) > 0:
            voice_to_pop = min(range(len(next_note_beats)),
                               key=lambda k: (next_note_beats[k], voice_list_copy[k][0].true_length))
            popped_note = voice_list_copy[voice_to_pop].pop(0)
            yield popped_note, next_note_beats[voice_to_pop]
            next_note_beats[voice_to_pop] += popped_note.true_length
            if len(voice_list_copy[voice_to_pop]) == 0:
                voice_list_copy.pop(voice_to_pop)
                next_note_beats.pop(voice_to_pop)

    def leaves(self, which_voices=None) -> Sequence[Note | Chord | Rest]:
        """
        Returns a tuple of all to the Notes/Chords/Rests in this measure, expanding out any tuplets or beam groups. The
        notes/chords/rests and draw from the specified voices, and are returned in order within the measure.

        :param which_voices: List of voices to return notes from (numbered 0, 1, 2, 3). The default value of None
            returns notes from all voices.
        """
        return tuple(leaf for leaf, _ in self.iter_leaves(which_voices))

    def iter_directions(self) -> Iterator[tuple[Direction, float]]:
        """
        Iterates through the Directions in this measure, both those attached to notes and those passed to the
        directions_with_displacements constructor argument.

        :return tuples of (Direction, beat within measure)
        """
        # directions_with_displacements is a list of (Direction, beat_within_measure) tuples
        directions_with_displacements = list(self.directions_with_displacements)
        # self.iter_leaves() gives (Note/Chord/Rest, beat_within_measure) tuples, so we unpack the leaf's directions
        directions_with_displacements.extend((direction, beat_within_measure)
                                             for leaf, beat_within_measure in self.iter_leaves()
                                             for direction in leaf.directions)
        directions_with_displacements.sort(key=lambda x: (x[1], isinstance(x, Direction)))
        yield from directions_with_displacements

    def iter_notations(self) -> Iterator[tuple[Notation, float]]:
        """
        Iterates through the Notations in this measure, in order.

        :return tuples of (Notation, beat_within_measure)
        """
        for leaf, beat_within_measure in self.iter_leaves():
            for notation in leaf.notations:
                yield notation, beat_within_measure

    def directions(self) -> Sequence[Direction]:
        """
        Returns a tuple of all to the Diections in this measure, both those attached to notes and those passed to the
        directions_with_displacements constructor argument. Returned in order within the measure.
        """
        return tuple(direction for direction, _ in self.iter_directions())

    def notations(self) -> Sequence[Notation]:
        """
        Returns a tuple of all to the Notation in this measure, returned in order within the measure.
        """
        return tuple(notation for notation, _ in self.iter_notations())

    def _set_leaf_voices(self):
        for i, voice in enumerate(self.voices):
            if voice is None:  # skip empty voices
                continue
            for element in voice:
                if isinstance(element, (BeamedGroup, Tuplet)):
                    # element is a container
                    for leaf in element.contents:
                        leaf.voice = i + 1
                else:
                    # element is a leaf
                    element.voice = i + 1
                    for direction in element.directions:
                        direction.voice = i + 1

    def _get_beat_division_for_directions(self):
        # determine what beat division is ideal for the independently placed directions
        if len(self.directions_with_displacements) == 0:
            return None
        return _least_common_multiple(*[Fraction(displacement).limit_denominator(256).denominator
                                        for _, displacement in self.directions_with_displacements])

    def render(self) -> Sequence[ElementTree.Element]:
        self._set_leaf_voices()

        measure_element = ElementTree.Element("measure", {"number": str(self.number)})

        attributes_el = ElementTree.SubElement(measure_element, "attributes")

        num_beat_divisions = _least_common_multiple(*[x.min_denominator() for x in self.leaves()])

        if len(self.directions_with_displacements) > 0:
            # if we're using independently placed directions, then we try to find a denominator that accommodates
            # that as precisely as possible this means
            ideal_division = _least_common_multiple(self._get_beat_division_for_directions(), num_beat_divisions)
            if ideal_division <= 1024:
                num_beat_divisions = ideal_division
            else:
                # Just in case the ideal division is totally outrageous, we just multiply the division
                # by two repeatedly until we are about to go over 1024
                num_beat_divisions *= max(1, 2 ** int(math.log2(1024 / num_beat_divisions)))

        for note in self.leaves():
            note.divisions = num_beat_divisions

        divisions_el = ElementTree.SubElement(attributes_el, "divisions")
        divisions_el.text = str(num_beat_divisions)

        if self.key is not None:
            attributes_el.extend(KeySignature.parse(self.key).render())

        if self.time_signature is not None:
            # time_signature is expressed as a tuple
            assert isinstance(self.time_signature, tuple) and len(self.time_signature) == 2
            time_el = ElementTree.SubElement(attributes_el, "time")
            ElementTree.SubElement(time_el, "beats").text = str(self.time_signature[0])
            ElementTree.SubElement(time_el, "beat-type").text = str(self.time_signature[1])

        if self.clef is not None:
            attributes_el.extend(self.clef.render())

        if self.transpose is not None:
            attributes_el.extend(self.transpose.render())

        if self.staves is not None:
            staves_el = ElementTree.SubElement(attributes_el, "staves")
            staves_el.text = str(self.staves)

        amount_to_backup = 0
        for i, voice in enumerate(self.voices):
            if voice is None:  # skip empty voices
                continue

            if i > 0:
                # for voices after 1, we need to add a backup element to go back to the start of the measure
                backup_el = ElementTree.SubElement(measure_element, "backup")
                ElementTree.SubElement(backup_el, "duration").text = str(amount_to_backup)
            amount_to_backup = 0

            for note_or_tuplet in voice:
                amount_to_backup += note_or_tuplet.length_in_divisions
                measure_element.extend(note_or_tuplet.render())

        if len(self.directions_with_displacements) > 0:
            current_location_in_measure = amount_to_backup
            for direction, displacement in self.directions_with_displacements:
                displacement = int(round(displacement * num_beat_divisions))
                if displacement < current_location_in_measure:
                    backup_el = ElementTree.SubElement(measure_element, "backup")
                    ElementTree.SubElement(backup_el, "duration").text = str(current_location_in_measure - displacement)
                elif displacement > current_location_in_measure:
                    forward_el = ElementTree.SubElement(measure_element, "forward")
                    ElementTree.SubElement(forward_el, "duration").text = str(displacement - current_location_in_measure)
                current_location_in_measure = displacement
                measure_element.extend(direction.render())

        if self.barline is not None:
            barline_el = ElementTree.SubElement(measure_element, "barline", {"location": "right"})
            ElementTree.SubElement(barline_el, "bar-style").text = Measure._barline_xml_names[self.barline.lower()]

        return measure_element,

    def wrap_as_score(self) -> Score:
        return Part("", [self]).wrap_as_score()


# -------------------------------------------- Part, PartGroup, and Score ------------------------------------------


class Part(MusicXMLComponent, MusicXMLContainer):
    """
    Represents a musical part/staff.

    :param part_name: name of this part
    :param measures: list of measures contained in this part
    :param part_id: unique identifier for the part (set automatically by the containing Score upon rendering)
    :param instrument_name: used by notation programs to understand which sound to use; not rendered in score. Set
        automatically based on part name if left as None.
    :param midi_program_num: the general midi program number for the instrument. Again, used by notation programs
        to pick an instrument sound. Set automatically based on part name if left as None.
    """

    general_midi_preset_nums = {
        # Official names
        'Acoustic Grand Piano': 1, 'Bright Acoustic Piano': 2, 'Electric Grand Piano': 3, 'Honky-tonk Piano': 4,
        'Electric Piano 1': 5, 'Electric Piano 2': 6, 'Harpsichord': 7, 'Clavinet': 8, 'Celesta': 9,
        'Glockenspiel': 10, 'Music Box': 11, 'Vibraphone': 12, 'Marimba': 13, 'Xylophone': 14, 'Tubular Bells': 15,
        'Dulcimer': 16, 'Drawbar Organ': 17, 'Percussive Organ': 18, 'Rock Organ': 19, 'Church Organ': 20,
        'Reed Organ': 21, 'Accordion': 22, 'Harmonica': 23, 'Tango Accordion': 24, 'Acoustic Guitar (nylon)': 25,
        'Acoustic Guitar (steel)': 26, 'Electric Guitar (jazz)': 27, 'Electric Guitar (clean)': 28,
        'Electric Guitar (muted)': 29, 'Overdriven Guitar': 30, 'Distortion Guitar': 31, 'Guitar harmonics': 32,
        'Acoustic Bass': 33, 'Electric Bass (finger)': 34, 'Electric Bass (pick)': 35, 'Fretless Bass': 36,
        'Slap Bass 1': 37, 'Slap Bass 2': 38, 'Synth Bass 1': 39, 'Synth Bass 2': 40, 'Violin': 41, 'Viola': 42,
        'Cello': 43, 'Contrabass': 44, 'Tremolo Strings': 45, 'Pizzicato Strings': 46, 'Orchestral Harp': 47,
        'Timpani': 48, 'String Ensemble 1': 49, 'String Ensemble 2': 50, 'Synth Strings 1': 51, 'Synth Strings 2': 52,
        'Choir Aahs': 53, 'Voice Oohs': 54, 'Synth Voice': 55, 'Orchestra Hit': 56, 'Trumpet': 57, 'Trombone': 58,
        'Tuba': 59, 'Muted Trumpet': 60, 'French Horn': 61, 'Brass Section': 62, 'Synth Brass 1': 63,
        'Synth Brass 2': 64, 'Soprano Sax': 65, 'Alto Sax': 66, 'Tenor Sax': 67, 'Baritone Sax': 68, 'Oboe': 69,
        'English Horn': 70, 'Bassoon': 71, 'Clarinet': 72, 'Piccolo': 73, 'Flute': 74, 'Recorder': 75, 'Pan Flute': 76,
        'Blown Bottle': 77, 'Shakuhachi': 78, 'Whistle': 79, 'Ocarina': 80, 'Lead 1 (square)': 81,
        'Lead 2 (sawtooth)': 82, 'Lead 3 (calliope)': 83, 'Lead 4 (chiff)': 84, 'Lead 5 (charang)': 85,
        'Lead 6 (voice)': 86, 'Lead 7 (fifths)': 87, 'Lead 8 (bass + lead)': 88, 'Pad 1 (new age)': 89,
        'Pad 2 (warm)': 90, 'Pad 3 (polysynth)': 91, 'Pad 4 (choir)': 92, 'Pad 5 (bowed)': 93, 'Pad 6 (metallic)': 94,
        'Pad 7 (halo)': 95, 'Pad 8 (sweep)': 96, 'FX 1 (rain)': 97, 'FX 2 (soundtrack)': 98, 'FX 3 (crystal)': 99,
        'FX 4 (atmosphere)': 100, 'FX 5 (brightness)': 101, 'FX 6 (goblins)': 102, 'FX 7 (echoes)': 103,
        'FX 8 (sci-fi)': 104, 'Sitar': 105, 'Banjo': 106, 'Shamisen': 107, 'Koto': 108, 'Kalimba': 109,
        'Bag pipe': 110, 'Fiddle': 111, 'Shanai': 112, 'Tinkle Bell': 113, 'Agogo': 114, 'Steel Drums': 115,
        'Woodblock': 116, 'Taiko Drum': 117, 'Melodic Tom': 118, 'Synth Drum': 119, 'Reverse Cymbal': 120,
        'Guitar Fret Noise': 121, 'Breath Noise': 122, 'Seashore': 123, 'Bird Tweet': 124, 'Telephone Ring': 125,
        'Helicopter': 126, 'Applause': 127, 'Gunshot': 128,
        # Some additional name simplifications and defaults
        'Piano': 1, 'Bells': 15, 'Harp': 47, 'Horn': 61, 'Organ': 20, 'Sax': 66, 'Strings': 49, 'Violoncello': 43,
        'Tom': 118
    }

    def __init__(self, part_name: str, measures: Sequence[Measure] = None, part_id: int = 1,
                 instrument_name: str = None, midi_program_num: int = None):
        self.part_id = part_id
        super().__init__(contents=measures, allowed_types=(Measure,))
        self.part_name = part_name
        self.instrument_name = instrument_name
        self.midi_program = midi_program_num

    @property
    def measures(self) -> Sequence[Measure]:
        """
        List of the measures in this Part.
        """
        return self.contents

    def iter_leaves(self, which_voices=None) -> Iterator[Note | Chord | Rest]:
        """
        Iterates through the Notes/Chords/Rests in this part, expanding out any measures, tuplets and beam groups. The
        notes/chords/rests are ordered in time, and draw from the specified voices.

        :param which_voices: List of voices to return notes from (numbered 0, 1, 2, 3). The default value of None
            returns notes from all voices.
        """
        for measure in self.measures:
            for leaf, _ in measure.iter_leaves(which_voices):
                yield leaf

    def iter_directions(self, direction_type=None) -> Iterator[Direction]:
        """
        Iterates through all directions, or all directions of a certain type, in this Part.

        :param direction_type: the type of direction to filter for, if any
        """
        for measure in self.measures:
            for direction, _ in measure.iter_directions():
                if direction_type is None or isinstance(direction, direction_type):
                    yield direction

    def iter_notations(self, notation_type=None) -> Iterator[Notation]:
        """
        Iterates through all notations, or all notations of a certain type, in this Part.

        :param notation_type: the type of notation to filter for, if any
        """
        for measure in self.measures:
            for notation, _ in measure.iter_notations():
                if notation_type is None or isinstance(notation, notation_type):
                    yield notation

    def render(self) -> Sequence[ElementTree.Element]:
        part_copy = deepcopy(self)
        Part._validate_spanner_numbers(part_copy)
        part_element = ElementTree.Element("part", {"id": "P{}".format(part_copy.part_id)})
        for i, measure in enumerate(part_copy.measures):
            measure.number = i + 1
            part_element.extend(measure.render())
        return part_element,

    @staticmethod
    def _validate_spanner_numbers(part: Part):
        """
        Redoes all the spanner numbers so that they are integers from 1-6 in accordance with the MusicXML number-level
        type. We want to be able to assign any number to the spanner (since keeping track of which numbers are available
        is a pain), and then adapt the numbers on export.

        :param part: the part whose spanners numbers to validate
        """
        for START_SPANNER_TYPE in StartNumberedSpanner.__subclasses__():
            STOP_SPANNER_TYPE = START_SPANNER_TYPE.STOP_TYPE
            MID_SPANNER_TYPES = START_SPANNER_TYPE.MID_TYPES

            # keep a dict of which input labels are associated with which output numbers
            input_to_output_numbers = {}

            iterator = part.iter_notations() if issubclass(START_SPANNER_TYPE, Notation) else part.iter_directions()

            for possible_spanner in iterator:
                if isinstance(possible_spanner, START_SPANNER_TYPE):
                    available_slur_numbers = [x for x in range(1, 7)
                                              if x not in sum(input_to_output_numbers.values(), [])]
                    if possible_spanner.label in available_slur_numbers:
                        available_slur_numbers.remove(possible_spanner.label)
                        input_to_output_numbers.setdefault(possible_spanner.label, []).append(possible_spanner.label)
                    elif len(available_slur_numbers) > 0:
                        output_num = available_slur_numbers[0]
                        input_to_output_numbers.setdefault(possible_spanner.label, []).append(output_num)
                        possible_spanner.label = output_num
                    else:
                        logging.warning("Ran out of available id numbers for {}; too many simultaneous.".
                                        format(START_SPANNER_TYPE))
                elif isinstance(possible_spanner, MID_SPANNER_TYPES):
                    if possible_spanner.label in input_to_output_numbers:
                        possible_spanner.label = input_to_output_numbers[possible_spanner.label][0]
                    else:
                        logging.warning("{} has no corresponding {}.".format(type(possible_spanner).__name__,
                                                                             START_SPANNER_TYPE.__name__))
                elif isinstance(possible_spanner, STOP_SPANNER_TYPE):
                    if possible_spanner.label in input_to_output_numbers:
                        output_num = input_to_output_numbers[possible_spanner.label].pop(0)
                        if len(input_to_output_numbers[possible_spanner.label]) == 0:
                            del input_to_output_numbers[possible_spanner.label]
                        possible_spanner.label = output_num
                    else:
                        logging.warning("{} has no corresponding {}.".format(STOP_SPANNER_TYPE.__name__,
                                                                             START_SPANNER_TYPE.__name__))

    def render_part_list_entry(self) -> Sequence[ElementTree.Element]:
        """
        Renders the "score-part" tag for the top of the MusicXML score.
        """
        score_part_el = ElementTree.Element("score-part", {"id": "P{}".format(self.part_id)})
        ElementTree.SubElement(score_part_el, "part-name").text = self.part_name

        name, num = Part._get_midi_preset(self.part_name)
        name = self.instrument_name if self.instrument_name is not None else name
        num = self.midi_program if self.midi_program is not None else num
        if name is not None:
            score_instrument_el = ElementTree.SubElement(score_part_el, "score-instrument",
                                                         {"id": "P{}-I1".format(self.part_id)})
            ElementTree.SubElement(score_instrument_el, "instrument-name").text = name
        if num is not None:
            midi_instrument_el = ElementTree.SubElement(score_part_el, "midi-instrument",
                                                        {"id": "P{}-I1".format(self.part_id)})
            ElementTree.SubElement(midi_instrument_el, "midi-program").text = str(num)
        return score_part_el,

    @staticmethod
    def _get_midi_preset(instrument_name):
        best_preset_match = None
        best_preset_score = 0
        altered_name = instrument_name.lower().strip()
        for gm_name in Part.general_midi_preset_nums:
            score = get_average_square_correlation(altered_name, gm_name.lower())
            if score > best_preset_score:
                best_preset_score = score
                best_preset_match = gm_name
        if best_preset_score > 1.5:
            # threshold for a close enough name match
            return best_preset_match, Part.general_midi_preset_nums[best_preset_match]
        else:
            # no good match
            return None, None

    def wrap_as_score(self) -> Score:
        return Score([self])


class PartGroup(MusicXMLComponent, MusicXMLContainer):
    """
    Represents a part group (a group of related parts, possible connected by a bracket)

    :param parts: list of parts contained in this group
    :param has_bracket: whether or not to place a bracket around the group in the score
    :param has_group_bar_line: whether or not to have bar lines cut through the entire group
    """

    def __init__(self, parts: Sequence[Part] = None, has_bracket: bool = True, has_group_bar_line: bool = True):
        super().__init__(contents=parts, allowed_types=(Part,))
        self.has_bracket = has_bracket
        self.has_group_bar_line = has_group_bar_line

    @property
    def parts(self) -> Sequence[Part]:
        """
        List of parts in this group.
        """
        return self.contents

    def render(self) -> Sequence[ElementTree.Element]:
        return sum((part.render() for part in self.parts), ())

    def render_part_list_entry(self) -> Sequence[ElementTree.Element]:
        """
        Renders the "part-group" tag and containing "score-part" tags for the top of the MusicXML score.
        """
        out = [self._render_start_element()]
        for part in self.parts:
            out.extend(part.render_part_list_entry())
        out.append(self._render_stop_element())
        return tuple(out)

    def _render_start_element(self):
        start_element = ElementTree.Element("part-group", {"type": "start"})
        if self.has_bracket:
            ElementTree.SubElement(start_element, "group-symbol").text = "bracket"
        ElementTree.SubElement(start_element, "group-barline").text = "yes" if self.has_group_bar_line else "no"
        return start_element

    @staticmethod
    def _render_stop_element():
        return ElementTree.Element("part-group", {"type": "stop"})

    def wrap_as_score(self) -> Score:
        return Score([self])


class Score(MusicXMLComponent, MusicXMLContainer):

    """
    Class representing a full musical score

    :param contents: list of parts and part groups included in this score
    :param title: title of the score
    :param composer: name of the composer
    """

    def __init__(self, contents: Sequence[Part | PartGroup] = None, title: str = None, composer: str = None):
        super().__init__(contents=contents, allowed_types=(Part, PartGroup))
        self.title = title
        self.composer = composer

    @property
    def parts(self) -> Sequence[Part]:
        """
        Returns a tuple of the parts in this score, expanding out any part groups.
        """
        return tuple(part for part_or_group in self.contents
                     for part in (part_or_group.parts if isinstance(part_or_group, PartGroup) else (part_or_group, )))

    def _set_part_numbers(self):
        next_id = 1
        for part in self.parts:
            part.part_id = next_id
            next_id += 1

    def render(self) -> Sequence[ElementTree.Element]:
        self._set_part_numbers()
        score_element = ElementTree.Element("score-partwise")
        work_el = ElementTree.SubElement(score_element, "work")
        if self.title is not None:
            ElementTree.SubElement(work_el, "work-title").text = self.title
        id_el = ElementTree.SubElement(score_element, "identification")
        if self.composer is not None:
            ElementTree.SubElement(id_el, "creator", {"type": "composer"}).text = self.composer
        encoding_el = ElementTree.SubElement(id_el, "encoding")
        ElementTree.SubElement(encoding_el, "encoding-date").text = str(datetime.date.today())
        ElementTree.SubElement(encoding_el, "software").text = "pymusicxml"
        part_list_el = ElementTree.SubElement(score_element, "part-list")
        for part_or_part_group in self.contents:
            part_list_el.extend(part_or_part_group.render_part_list_entry())
            score_element.extend(part_or_part_group.render())
        return score_element,

    def wrap_as_score(self) -> Score:
        return self


class NumberedSpanner(ABC):
    """Abstract base class for part of a Direction or Notation that spans multiple time-points."""

    def __init__(self, label: Any = 1):
        self.label = label


class StopNumberedSpanner(NumberedSpanner):
    """Abstract base class for the end of a Direction or Notation that spans multiple time-points."""
    pass


class MidNumberedSpanner(NumberedSpanner):
    """Abstract base class for the middle of a Direction or Notation that spans multiple time-points."""
    pass


class StartNumberedSpanner(NumberedSpanner):
    """Abstract base class for the start of a Direction or Notation that spans multiple time-points."""

    #: The associated stop type; used for validating line numbers
    STOP_TYPE: StopNumberedSpanner = NotImplemented

    MID_TYPES: Sequence[MidNumberedSpanner] = ()


class Notation(MusicXMLComponent, ABC):

    """Abstract base class for MusicXML Notations (glissandi, slurs)."""

    @abstractmethod
    def render(self) -> Sequence[ElementTree.Element]:
        pass

    def wrap_as_score(self) -> Score:
        return Note("c5", 1, notations=(self, )).wrap_as_score()


class Direction(MusicXMLComponent, ABC):

    """
    Abstract base class for musical directions, such as text and metronome marks.
    """

    def __init__(self, placement: str | StaffPlacement = "above", voice: int = 1, staff: int = None):
        self.placement = StaffPlacement(placement) if isinstance(placement, str) else placement
        self.voice = voice
        self.staff = staff

    def render(self) -> Sequence[ElementTree.Element]:
        direction_element = ElementTree.Element("direction", {"placement": self.placement.value})
        direction_element.extend(self.render_direction_type())
        ElementTree.SubElement(direction_element, "voice").text = str(self.voice)
        if self.staff is not None:
            ElementTree.SubElement(direction_element, "staff").text = str(self.staff)
        return direction_element,

    @abstractmethod
    def render_direction_type(self) -> Sequence[ElementTree.Element]:
        """
        Renders the <direction-type> element that constitutes the main substance of a <direction> element.
        """
        pass

    def wrap_as_score(self) -> Score:
        return Measure([BarRest(4, directions=(self, ))], time_signature=(4, 4)).wrap_as_score()


class MultiGliss(Notation):
    """Abstract base class for start and stop multi glisses"""

    def __init__(self, numbers: Sequence[int] = (1,)):
        self.numbers = numbers
