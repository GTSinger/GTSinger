"""
A simple utility for exporting MusicXML files that represents musical objects more hierarchically.

While it is of course possible to create MusicXML files in Python directly via ElementTree, the format is awkwardly
non-hierarchical. For instance, there are no separate objects for chords and tuplets; rather they exist only as special
annotations to the note objects themselves. In pymusicxml, chords are represented similarly to notes but with multiple
pitches, and tuplets are treated as containers for notes, chords, and rests.

At the moment, this library is intended as a tool for composition, making it easier to construct and export MusicXML
scores in Python. In the future, the ability to parse existing MusicXML scores may be added.
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

from .score_components import Pitch, Duration, BarRestDuration, Note, Rest, BarRest, Chord, GraceNote, GraceChord, \
    BeamedGroup, Tuplet, Clef, Transpose, Measure, Part, PartGroup, Score, Notehead, Notation, Direction
from pymusicxml.notations import StartGliss, StopGliss, StartMultiGliss, StopMultiGliss, Fermata, Arpeggiate, \
    NonArpeggiate, Technical, UpBow, DownBow, OpenString, Harmonic, Stopped, SnapPizzicato, Ornament, Mordent, Turn, \
    TrillMark, Schleifer, Tremolo
from pymusicxml.directions import Harmony, Degree, MetronomeMark, TextAnnotation, Dynamic
from pymusicxml.spanners import StopBracket, StartBracket, StopDashes, StartDashes, StopTrill, StartTrill, StopPedal, \
    ChangePedal, StartPedal, StopHairpin, StartHairpin, StopSlur, StartSlur
from pymusicxml.enums import *
import importlib.metadata

__version__ = importlib.metadata.version('pymusicxml')
__author__ = importlib.metadata.metadata('pymusicxml')['Author']
