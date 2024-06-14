"""
Module containing all non-spanner notations, such as glisses, bowings, fermatas, etc.
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
from abc import abstractmethod, ABC
from typing import Sequence
from xml.etree import ElementTree
from pymusicxml.score_components import Notation, MultiGliss
from pymusicxml.enums import StaffPlacement, ArpeggiationDirection


class StartGliss(Notation):
    """
    Notation to attach to a note that starts a glissando

    :param number: each glissando is given an id number to distinguish it from other glissandi. This must range from
        1 to 6.
    """

    def __init__(self, number: int = 1):
        self.number = number

    def render(self) -> Sequence[ElementTree.Element]:
        return ElementTree.Element("slide", {"type": "start", "line-type": "solid", "number": str(self.number)}),


class StopGliss(Notation):
    """
    Notation to attach to a note that ends a glissando

    :param number: this should correspond to the id number of the associated :class:`StartGliss`.
    """

    def __init__(self, number: int = 1):
        self.number = number

    def render(self) -> Sequence[ElementTree.Element]:
        return ElementTree.Element("slide", {"type": "stop", "line-type": "solid", "number": str(self.number)}),


class StartMultiGliss(MultiGliss):

    """
    Multi-gliss notation used for glissing multiple members of a chord

    :param numbers: most natural is to pass a range object here, for the range of numbers to assign to the glisses
        of consecutive chord member. However, in the case of a chord where, say, you want the upper two notes to
        gliss but not the bottom, pass (None, 1, 2) to this parameter.
    """

    def render(self) -> Sequence[ElementTree.Element]:
        return tuple(StartGliss(n) if n is not None else None for n in self.numbers)


class StopMultiGliss(MultiGliss):

    """
    End of a multi-gliss notation used for glissing multiple members of a chord.

    :param numbers: These should correspond to the id numbers of the associated :class:`StartMultiGliss`.
    """

    def render(self) -> Sequence[ElementTree.Element]:
        return tuple(StopGliss(n) if n is not None else None for n in self.numbers)


class Fermata(Notation):
    """
    Fermata notation.

    :param inverted: if true, an inverted fermata
    """

    def __init__(self, inverted: bool = False):
        self.inverted = inverted

    def render(self) -> Sequence[ElementTree.Element]:
        if self.inverted:
            return ElementTree.Element("fermata", {"type": "inverted"}),
        else:
            return ElementTree.Element("fermata"),


class Arpeggiate(Notation):
    """
    Chord arpeggiation notation.

    :param direction: "up" or "down"
    """

    def __init__(self, direction: str | ArpeggiationDirection = None):
        self.direction = ArpeggiationDirection(direction) if isinstance(direction, str) else direction

    def render(self) -> Sequence[ElementTree.Element]:
        if self.direction is None:
            return ElementTree.Element("arpeggiate"),
        else:
            return ElementTree.Element("arpeggiate", {"direction": self.direction.value}),


class NonArpeggiate(Notation):
    """
    Chord non-arpeggiate notation. pymusicxml only allows full chord non-arpeggiates.
    """

    def render(self) -> Sequence[ElementTree.Element]:
        return ElementTree.Element("non-arpeggiate", {"type": "top"}),


# --------------------------------------- Technical Notations --------------------------------------------


class Technical(Notation, ABC):
    """Abstract class for all technical notations"""

    @abstractmethod
    def render_technical(self) -> Sequence[ElementTree.Element]:
        """Renders the contents of the technicalelement."""
        pass

    def render(self) -> Sequence[ElementTree.Element]:
        technical_el = ElementTree.Element("technical")
        technical_el.extend(self.render_technical())
        return technical_el,


class UpBow(Technical):
    """Up-bow notation"""

    def render_technical(self) -> Sequence[ElementTree.Element]:
        return ElementTree.Element("up-bow"),


class DownBow(Technical):
    """Down-bow notation"""

    def render_technical(self) -> Sequence[ElementTree.Element]:
        return ElementTree.Element("down-bow"),


class OpenString(Technical):
    """Open-string notation"""

    def render_technical(self) -> Sequence[ElementTree.Element]:
        return ElementTree.Element("open-string"),


class Harmonic(Technical):
    """Harmonic notation.

    Note: This is not the <harmony/> notation used for chord changes. This class
    represents playing a note as a harmonic on a single string."""

    def render_technical(self) -> Sequence[ElementTree.Element]:
        return ElementTree.Element("harmonic"),


class Stopped(Technical):
    """Stopped notation"""

    def render_technical(self) -> Sequence[ElementTree.Element]:
        return ElementTree.Element("stopped"),


class SnapPizzicato(Technical):
    """Snap-pizzicato notation"""

    def render_technical(self) -> Sequence[ElementTree.Element]:
        return ElementTree.Element("snap-pizzicato"),


# -------------------------------------------- Ornaments -------------------------------------------------


class Ornament(Notation, ABC):
    "Abstract class for all ornament notations"

    @abstractmethod
    def render_ornament(self) -> Sequence[ElementTree.Element]:
        """Renders the contents of the ornaments element."""
        pass

    def render(self) -> Sequence[ElementTree.Element]:
        ornaments_el = ElementTree.Element("ornaments")
        ornaments_el.extend(self.render_ornament())
        return ornaments_el,


class Mordent(Ornament):
    """
    Mordent ornament.

    :param inverted: if true, an inverted mordent
    :param placement: "above" or "below"
    """
    def __init__(self, inverted: bool = False, placement: str | StaffPlacement = "above"):
        self.placement = StaffPlacement(placement) if isinstance(placement, str) else placement
        self.inverted = inverted

    def render_ornament(self) -> Sequence[ElementTree.Element]:
        return ElementTree.Element("inverted-mordent" if self.inverted else "mordent",
                                   {"placement": self.placement.value}),


class Turn(Ornament):
    """
    Turn ornament.

    :param inverted: if true, an inverted turn
    :param delayed: if true, a turn which is delayed until the end of the note
    :param placement: "above" or "below"
    """
    def __init__(self, inverted: bool = False, delayed: bool = False, placement: str | StaffPlacement = "above"):
        self.placement = StaffPlacement(placement) if isinstance(placement, str) else placement
        self.inverted = inverted
        self.delayed = delayed

    def render_ornament(self) -> Sequence[ElementTree.Element]:

        return ElementTree.Element("delayed-" if self.delayed else "" + "inverted-" if self.inverted else "" + "turn",
                                   {"placement": self.placement.value}),


class TrillMark(Ornament):
    """
    Trill mark on a single note (without wavy line).

    :param placement: "above" or "below"
    """

    def __init__(self, placement: str | StaffPlacement = "above"):
        self.placement = StaffPlacement(placement) if isinstance(placement, str) else placement

    def render_ornament(self) -> Sequence[ElementTree.Element]:
        return ElementTree.Element("trill-mark", {"placement": self.placement.value}),


class Schleifer(Ornament):
    """
    Schleifer mark.

    :param placement: "above" or "below"
    """

    def __init__(self, placement: str | StaffPlacement = "above"):
        self.placement = StaffPlacement(placement) if isinstance(placement, str) else placement

    def render_ornament(self) -> Sequence[ElementTree.Element]:
        return ElementTree.Element("schleifer", {"placement": self.placement.value}),


class Tremolo(Ornament):
    """
    Tremolo lines on a note stem.

    :param num_lines: number of tremolo marks on the stem. (Defaults to 3 for traditional unmeasured tremolo).
    """

    def __init__(self, num_lines: int = 3):
        if not 0 <= num_lines <= 8:
            raise ValueError("num_lines must be between 0 and 8")
        self.num_lines = num_lines

    def render_ornament(self) -> Sequence[ElementTree.Element]:
        tremolo_el = ElementTree.Element("tremolo")
        tremolo_el.text = str(self.num_lines)
        return tremolo_el,
