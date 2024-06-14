"""
Module containing all spanners (i.e. notations that span a time-range in the score, such as slurs,
brackets, hairpins, etc.)
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
from numbers import Real
from typing import Any, Sequence
from xml.etree import ElementTree
from .enums import LineEnd, LineType, AccidentalType, HairpinType, StaffPlacement
from .score_components import StopNumberedSpanner, MidNumberedSpanner, StartNumberedSpanner
from .notations import Notation
from .directions import TextAnnotation
from pymusicxml import Direction


class StopBracket(Direction, StopNumberedSpanner):

    def __init__(self, label: Any = 1, line_end: str | LineEnd = None, end_length: Real = None,
                 text: str | TextAnnotation = None, placement: str | StaffPlacement = "above",
                 voice: int = 1, staff: int = None):
        """
        End of a bracket spanner.

        :param label: this should correspond to the label of the associated :class:`StartBracket`
        :param line_end: Type of hook/arrow at the end of this bracket
        :param end_length: Length of the hock at the end of this bracket
        :param text: Any text to attach to the end of this bracket
        :param placement: Where to place the direction in relation to the staff ("above" or "below")
        :param voice: Which voice to attach to
        :param staff: Which staff to attach to if the part has multiple staves
        """
        StopNumberedSpanner.__init__(self, label)
        Direction.__init__(self, placement, voice, staff)
        self.line_end = LineEnd(line_end) if isinstance(line_end, str) else line_end
        self.end_length = end_length
        self.text = TextAnnotation(text) if isinstance(text, str) else text
        if self.line_end is None:
            if self.text is None:
                # default to a downward hook if there's no text
                self.line_end = LineEnd("down")
            else:
                # and no end cap if there is
                self.line_end = LineEnd("none")

    def render_direction_type(self) -> Sequence[ElementTree.Element]:
        direction_type_el = ElementTree.Element("direction-type")
        bracket_dict = {"type": "stop", "number": str(self.label)}
        if self.line_end is not None:
            bracket_dict["line-end"] = str(self.line_end.value)
        if self.end_length is not None:
            bracket_dict["end-length"] = str(self.end_length)
        ElementTree.SubElement(direction_type_el, "bracket", bracket_dict)
        return direction_type_el,

    def render(self) -> Sequence[ElementTree.Element]:
        direction = super().render()[0]
        if self.text is not None:
            direction.insert(0, self.text.render_direction_type()[0])
        return direction,


class StartBracket(Direction, StartNumberedSpanner):
    """
    Start of a bracket spanner.

    :param label: each spanner is given an label to distinguish it from other spanners of the same type. In the MusicXML
        standard, this is a number from 1 to 6, but in pymusicxml it is allowed to be anything (including, for instance,
        a string). These labels are then converted to numbers on export.
    :param line_end: Type of hook/arrow at the start of this bracket
    :param end_length: Length of the hock at the start of this bracket
    :param text: Any text to attach to the start of this bracket
    :param placement: Where to place the direction in relation to the staff ("above" or "below")
    :param voice: Which voice to attach to
    :param staff: Which staff to attach to if the part has multiple staves
    """

    STOP_TYPE = StopBracket

    def __init__(self, label: Any = 1, line_type: str | LineType = "dashed", line_end: str | LineEnd = None,
                 end_length: Real = None, text: str | TextAnnotation = None,
                 placement: str | StaffPlacement = "above", voice: int = 1, staff: int = None):
        StartNumberedSpanner.__init__(self, label)
        Direction.__init__(self, placement, voice, staff)
        self.line_type = LineType(line_type) if isinstance(line_type, str) else line_type
        self.line_end = LineEnd(line_end) if isinstance(line_end, str) else line_end
        self.end_length = end_length
        self.text = TextAnnotation(text) if isinstance(text, str) else text
        if self.line_end is None:
            if self.text is None:
                # default to a downward hook if there's no text
                self.line_end = LineEnd("down")
            else:
                # and no end cap if there is
                self.line_end = LineEnd("none")

    def render_direction_type(self) -> Sequence[ElementTree.Element]:
        direction_type_el = ElementTree.Element("direction-type")
        bracket_dict = {"type": "start", "number": str(self.label)}
        if self.line_type is not None:
            bracket_dict["line-type"] = str(self.line_type.value)
        if self.line_end is not None:
            bracket_dict["line-end"] = str(self.line_end.value)
        if self.end_length is not None:
            bracket_dict["end-length"] = str(self.end_length)
        ElementTree.SubElement(direction_type_el, "bracket", bracket_dict)
        return direction_type_el,

    def render(self) -> Sequence[ElementTree.Element]:
        direction = super().render()[0]
        if self.text is not None:
            direction.insert(0, self.text.render_direction_type()[0])
        return direction,


class StopDashes(Direction, StopNumberedSpanner):

    """
    End of a dashed spanner (e.g. used by a dashed "cresc." or "dim." marking)

    :param label: this should correspond to the label of the associated :class:`StartDashes`
    :param text: Any text to attach to the end of this dashed spanner
    :param placement: Where to place the direction in relation to the staff ("above" or "below")
    :param voice: Which voice to attach to
    :param staff: Which staff to attach to if the part has multiple staves
    """

    def __init__(self, label: Any = 1, text: str | TextAnnotation = None,
                 placement: str | StaffPlacement = "above", voice: int = 1, staff: int = None):

        StopNumberedSpanner.__init__(self, label)
        Direction.__init__(self, placement, voice, staff)
        self.text = TextAnnotation(text) if isinstance(text, str) else text

    def render_direction_type(self) -> Sequence[ElementTree.Element]:
        direction_type_el = ElementTree.Element("direction-type")
        ElementTree.SubElement(direction_type_el, "dashes", {"type": "stop", "number": str(self.label)})
        return direction_type_el,

    def render(self) -> Sequence[ElementTree.Element]:
        direction = super().render()[0]
        if self.text is not None:
            direction.insert(1, self.text.render_direction_type()[0])
        return direction,


class StartDashes(Direction, StartNumberedSpanner):
    """
    Start of a dashed spanner (e.g. used by a dashed "cresc." or "dim." marking)

    :param label: each spanner is given an label to distinguish it from other spanners of the same type. In the MusicXML
        standard, this is a number from 1 to 6, but in pymusicxml it is allowed to be anything (including, for instance,
        a string). These labels are then converted to numbers on export.
    :param dash_length: Length of the dashes
    :param space_length: Length of the space between the dashes
    :param text: Any text to attach to the start of this dashed spanner
    :param placement: Where to place the direction in relation to the staff ("above" or "below")
    :param voice: Which voice to attach to
    :param staff: Which staff to attach to if the part has multiple staves
    """

    STOP_TYPE = StopDashes

    def __init__(self, label: Any = 1, dash_length: Real = None, space_length: Real = None,
                 text: str | TextAnnotation = None, placement: str | StaffPlacement = "above",
                 voice: int = 1, staff: int = None):
        StartNumberedSpanner.__init__(self, label)
        Direction.__init__(self, placement, voice, staff)
        self.text = TextAnnotation(text) if isinstance(text, str) else text
        self.dash_length = dash_length
        self.space_length = space_length

    def render_direction_type(self) -> Sequence[ElementTree.Element]:
        direction_type_el = ElementTree.Element("direction-type")
        dash_dict = {"type": "start", "number": str(self.label)}
        if self.dash_length is not None:
            dash_dict["dash-length"] = str(self.dash_length)
        if self.space_length is not None:
            dash_dict["space-length"] = str(self.space_length)
        ElementTree.SubElement(direction_type_el, "dashes", dash_dict)
        return direction_type_el,

    def render(self) -> Sequence[ElementTree.Element]:
        direction = super().render()[0]
        if self.text is not None:
            direction.insert(0, self.text.render_direction_type()[0])
        return direction,


class StopTrill(Notation, StopNumberedSpanner):
    """
    Stops a trill spanner with a wavy line.

    :param label: this should correspond to the label of the associated :class:`StartTrill`
    :param placement: Where to place the direction in relation to the staff ("above" or "below")
    """

    def __init__(self, label: Any = 1, placement: StaffPlacement | str = "above"):
        self.placement = StaffPlacement(placement) if isinstance(placement, str) else placement
        super().__init__(label)

    def render(self) -> Sequence[ElementTree.Element]:
        ornaments_el = ElementTree.Element("ornaments")
        ElementTree.SubElement(ornaments_el, "wavy-line",
                               {"type": "stop", "placement": self.placement.value, "number": str(self.label)})
        return ornaments_el,


class StartTrill(Notation, StartNumberedSpanner):
    """
    Starts a trill spanner with a wavy line.

    :param label: each spanner is given an label to distinguish it from other spanners of the same type. In the MusicXML
        standard, this is a number from 1 to 6, but in pymusicxml it is allowed to be anything (including, for instance,
        a string). These labels are then converted to numbers on export.
    :param placement: Where to place the direction in relation to the staff ("above" or "below")
    :param accidental: Accidental annotation to go on the trill ("flat-flat", "flat", "natural", "sharp",
        or "double-sharp")
    """

    STOP_TYPE = StopTrill

    def __init__(self, label: Any = 1, placement: StaffPlacement | str = "above",
                 accidental: AccidentalType | str = None):
        self.placement = StaffPlacement(placement) if isinstance(placement, str) else placement
        self.accidental = AccidentalType(accidental)if isinstance(accidental, str) else accidental
        super().__init__(label)

    def render(self) -> Sequence[ElementTree.Element]:
        ornaments_el = ElementTree.Element("ornaments")
        ElementTree.SubElement(ornaments_el, "trill-mark")
        if self.accidental is not None:
            accidental_mark = ElementTree.SubElement(ornaments_el, "accidental-mark")
            accidental_mark.text = self.accidental.value
        ElementTree.SubElement(ornaments_el, "wavy-line",
                               {"type": "start", "placement": self.placement.value, "number": str(self.label)})
        return ornaments_el,


class StopPedal(Direction, StopNumberedSpanner):
    """
    Stops a sustain pedal spanner.

    :param label: this should correspond to the label of the associated :class:`StartPedal`
    :param sign: whether or not to include a "*" sign
    :param line: whether or not to use a line in the pedal marking
    :param placement: Where to place the direction in relation to the staff ("above" or "below")
    :param voice: Which voice to attach to
    :param staff: Which staff to attach to if the part has multiple staves
    """

    def __init__(self, label: Any = 1, sign: bool = False, line: bool = True,
                 placement: str | StaffPlacement = "below", voice: int = 1, staff: int = None):
        StartNumberedSpanner.__init__(self, label)
        Direction.__init__(self, placement, voice, staff)
        self.sign = sign
        self.line = line

    def render_direction_type(self) -> Sequence[ElementTree.Element]:
        direction_type_el = ElementTree.Element("direction-type")
        ElementTree.SubElement(direction_type_el, "pedal",
                               {"type": "stop", "number": str(self.label), "sign": ("no", "yes")[self.sign],
                                "line": ("no", "yes")[self.line]})
        return direction_type_el,


class ChangePedal(Direction, MidNumberedSpanner):
    """
    Pedal change in the middle of a sustain pedal spanner.

    :param label: this should correspond to the label of the associated :class:`StartPedal`
    :param sign: unclear what this means in the case of a change pedal
    :param line: whether or not to use a line in the pedal marking
    :param placement: Where to place the direction in relation to the staff ("above" or "below")
    :param voice: Which voice to attach to
    :param staff: Which staff to attach to if the part has multiple staves
    """

    def __init__(self, label: Any = 1, sign: bool = True, line: bool = True,
                 placement: str | StaffPlacement = "below", voice: int = 1, staff: int = None):
        StartNumberedSpanner.__init__(self, label)
        Direction.__init__(self, placement, voice, staff)
        self.sign = sign
        self.line = line

    def render_direction_type(self) -> Sequence[ElementTree.Element]:
        direction_type_el = ElementTree.Element("direction-type")
        ElementTree.SubElement(direction_type_el, "pedal",
                               {"type": "change", "number": str(self.label), "sign": ("no", "yes")[self.sign],
                                "line": ("no", "yes")[self.line]})
        return direction_type_el,


class StartPedal(Direction, StartNumberedSpanner):
    """
    Start of a sustain pedal spanner.

    :param label: each spanner is given an label to distinguish it from other spanners of the same type. In the MusicXML
        standard, this is a number from 1 to 6, but in pymusicxml it is allowed to be anything (including, for instance,
        a string). These labels are then converted to numbers on export.
    :param sign: whether or not to include a "Ped" sign
    :param line: whether or not to use a line in the pedal marking
    :param placement: Where to place the direction in relation to the staff ("above" or "below")
    :param voice: Which voice to attach to
    :param staff: Which staff to attach to if the part has multiple staves
    """

    STOP_TYPE = StopPedal
    MID_TYPES = (ChangePedal, )

    def __init__(self, label: Any = 1, sign: bool = True, line: bool = True,
                 placement: str | StaffPlacement = "below", voice: int = 1, staff: int = None):
        StartNumberedSpanner.__init__(self, label)
        Direction.__init__(self, placement, voice, staff)
        self.sign = sign
        self.line = line

    def render_direction_type(self) -> Sequence[ElementTree.Element]:
        direction_type_el = ElementTree.Element("direction-type")
        ElementTree.SubElement(direction_type_el, "pedal",
                               {"type": "start", "number": str(self.label), "sign": ("no", "yes")[self.sign],
                                "line": ("no", "yes")[self.line]})
        return direction_type_el,


class StopHairpin(Direction, StopNumberedSpanner):
    """
    Notation to attach to a note that ends a hairpin

    :param label: this should correspond to the label of the associated :class:`StartHairpin`
    """

    def __init__(self, label: Any = 1, spread: Real = None, placement: str | StaffPlacement = "below",
                 voice: int = 1, staff: int = None):
        StopNumberedSpanner.__init__(self, label)
        Direction.__init__(self, placement, voice, staff)
        self.spread = spread

    def render_direction_type(self) -> Sequence[ElementTree.Element]:
        direction_type_el = ElementTree.Element("direction-type")
        wedge_dict = {"type": "stop", "number": str(self.label)}
        if self.spread is not None:
            wedge_dict["spread"] = str(self.spread)
        ElementTree.SubElement(direction_type_el, "wedge", wedge_dict)
        return direction_type_el,


class StartHairpin(Direction, StartNumberedSpanner):
    """
    Notation to attach to a note that starts a hairpin

    :param hairpin_type: the type of hairpin ("crescendo" or "diminuendo")
    :param label: each spanner is given an label to distinguish it from other spanners of the same type. In the MusicXML
        standard, this is a number from 1 to 6, but in pymusicxml it is allowed to be anything (including, for instance,
        a string). These labels are then converted to numbers on export.
    """

    STOP_TYPE = StopHairpin

    def __init__(self, hairpin_type: str | HairpinType, label: Any = 1, spread: Real = None,
                 placement: str | StaffPlacement = "below", niente: bool = False, voice: int = 1,
                 staff: int = None):
        StopNumberedSpanner.__init__(self, label)
        Direction.__init__(self, placement, voice, staff)
        self.hairpin_type = HairpinType(hairpin_type) if isinstance(hairpin_type, str) else hairpin_type
        self.spread = spread
        self.niente = niente

    def render_direction_type(self) -> Sequence[ElementTree.Element]:
        direction_type_el = ElementTree.Element("direction-type")
        wedge_dict = {"type": self.hairpin_type.value, "number": str(self.label)}
        if self.niente:
            wedge_dict["niente"] = "yes"
        if self.spread is not None:
            wedge_dict["spread"] = str(self.spread)
        ElementTree.SubElement(direction_type_el, "wedge", wedge_dict)
        return direction_type_el,


class StopSlur(Notation, StopNumberedSpanner):

    """
    Notation to attach to a note that ends a slur

    :param label: this should correspond to the slur label of the associated :class:`StartSlur`.
    """

    def render(self) -> Sequence[ElementTree.Element]:
        return ElementTree.Element("slur", {"type": "stop", "number": str(self.label)}),


class StartSlur(Notation, StartNumberedSpanner):

    """
    Notation to attach to a note that starts a slur

    :param label: each spanner is given an label to distinguish it from other spanners of the same type. In the MusicXML
        standard, this is a number from 1 to 6, but in pymusicxml it is allowed to be anything (including, for instance,
        a string). These labels are then converted to numbers on export.
    """

    STOP_TYPE = StopSlur

    def render(self) -> Sequence[ElementTree.Element]:
        return ElementTree.Element("slur", {"type": "start", "number": str(self.label)}),
