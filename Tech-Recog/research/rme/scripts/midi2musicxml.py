import math

from pymusicxml import *
from typing import Sequence, Union
from xml.etree import ElementTree
from pymusicxml.enums import StaffPlacement
from pymusicxml.score_components import Duration, Direction, MusicXMLComponent, MusicXMLContainer, DurationalObject, \
    TraditionalKeySignature, pad_with_rests
from pymusicxml._utilities import _is_power_of_two
from tqdm import tqdm
import glob
import json
import datetime
from numbers import Real
import xmltodict
from fractions import Fraction
import pretty_midi
import os
import logging

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
index_to_length = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125]


class HwScore(MusicXMLComponent, MusicXMLContainer):
    """
    Class representing a full musical score

    :param contents: list of parts and part groups included in this score
    :param title: title of the score
    :param composer: name of the composer
    """

    def __init__(self, contents: Sequence[Union[Part, PartGroup]] = None, title: str = None, composer: str = None,
                 lyricist: str = None):
        super().__init__(contents=contents, allowed_types=(Part, PartGroup))
        self.title = title
        self.composer = composer
        self.lyricist = lyricist

    @property
    def parts(self) -> Sequence[Part]:
        """
        Returns a tuple of the parts in this score, expanding out any part groups.
        """
        return tuple(part for part_or_group in self.contents
                     for part in (part_or_group.parts if isinstance(part_or_group, PartGroup) else (part_or_group,)))

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
        if self.lyricist is not None:
            ElementTree.SubElement(id_el, "creator", {"type": "lyricist"}).text = self.lyricist
        encoding_el = ElementTree.SubElement(id_el, "encoding")
        ElementTree.SubElement(encoding_el, "encoding-date").text = str(datetime.date.today())
        ElementTree.SubElement(encoding_el, "software").text = "HwMusicxml"
        part_list_el = ElementTree.SubElement(score_element, "part-list")
        for part_or_part_group in self.contents:
            part_list_el.extend(part_or_part_group.render_part_list_entry())
            score_element.extend(part_or_part_group.render())
        return score_element,

    def wrap_as_score(self) -> Score:
        return self


class HwMetronomeMark(Direction):
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

    def __init__(self, beat_length: float, bpm: float, placement: Union[str, StaffPlacement] = "above",
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
        # ElementTree.SubElement(type_el, "sound", {"tempo": str(self.bpm)})
        return type_el,


class HwTextAnnotation(Direction):
    """
    Abstract base class for musical directions, such as text and metronome marks.
    """

    def __init__(self, text: str, font_size: float = None, italic: bool = False, bold: bool = False,
                 placement: Union[str, StaffPlacement] = "above", voice: int = 1, staff: int = None, **kwargs):
        super().__init__(placement, voice, staff)
        self.text = text
        self.text_properties = kwargs
        if font_size is not None:
            self.text_properties["font-size"] = font_size
        if italic:
            self.text_properties["font-style"] = "italic"
        if bold:
            self.text_properties["font-weight"] = "bold"

    def render(self) -> Sequence[ElementTree.Element]:
        direction_element = ElementTree.Element("lyric")
        ElementTree.SubElement(direction_element, "syllabic").text = 'single'
        ElementTree.SubElement(direction_element, "text", self.text_properties).text = str(self.text)
        return direction_element,

    def render_direction_type(self) -> Sequence[ElementTree.Element]:
        pass


class _XMLHwNote(DurationalObject):
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

    def __init__(self, pitch, duration, text, ties=None, notations=(), articulations=(), notehead=None, beams=None,
                 directions=(), stemless=False, grace=False, is_chord_member=False, voice=None, staff=None,
                 velocity=None):

        assert not (grace and pitch is None)  # can't have grace rests
        self.pitch = pitch
        assert isinstance(duration, (Duration, BarRestDuration))
        self.duration = duration
        assert ties in ("start", "continue", "stop", None)
        self.text = text
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

        if self.text is not None:
            lyric_el = ElementTree.Element("lyric")
            ElementTree.SubElement(lyric_el, "syllabic").text = 'single'
            ElementTree.SubElement(lyric_el, "text").text = self.text
            note_element.append(lyric_el)

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


class HwNote(_XMLHwNote):
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

    def __init__(self, pitch: Union[Pitch, str], duration: Union[Duration, str, float], text: str = None, ties: str = None,
                 notations=(), articulations=(), notehead: Union[Notehead, str] = None,
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
        super().__init__(pitch, duration, text, ties=ties, notations=notations, articulations=articulations,
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


def extract_time(dur, tempo):
    # thirty_socond_dur = 7.5 / tempo
    # num_thirty_socond = dur / thirty_socond_dur
    # return int(num_thirty_socond)
    sixteenth_dur = 15 / tempo
    num_sixteenth = dur / sixteenth_dur
    zs = math.floor(num_sixteenth)
    xs = num_sixteenth - math.floor(num_sixteenth)
    if 0.4 < xs < 0.6:
        num_thirty_socond = 1
    elif xs <= 0.4:
        num_thirty_socond = 0
    else:
        num_thirty_socond = 0
        zs += 1
    num_thirty_socond = zs * 2 + num_thirty_socond
    return int(num_thirty_socond)


def generate_mxml(submit_path, txt_path, orig_musixml_path):
    with open(txt_path, "r") as f:
        text_list = f.readline().strip().split(' ')
    text_idx = 0
    with open(orig_musixml_path, "r") as f:
        xmldict = xmltodict.parse(f.read())
    if "score-partwise" in xmldict:
        xmldict = xmldict["score-partwise"]
    if 'work' in xmldict and 'work-title' in xmldict['work']:
        title = xmldict['work']['work-title']
    else:
        title = ''
    composer = None
    lyricist = None
    if 'creator' in xmldict['identification']:
        if type(xmldict['identification']['creator']) == list:
            for creator in xmldict['identification']['creator']:
                if creator['@type'] == 'composer':
                    composer = creator['#text']
                elif creator['@type'] == 'lyricist':
                    lyricist = creator['#text']
        elif type(xmldict['identification']['creator']) == dict:
            creator = xmldict['identification']['creator']['#text'].split('\n')
            if len(creator) == 1:
                composer = creator[0]
            else:
                composer = creator[0]
                lyricist = creator[1]

    parts = xmldict["part"]
    if type(parts) != list:
        parts = [parts]
    for p, part in enumerate(parts):  # music parts
        for m, measure in enumerate(part["measure"]):
            # partwise music meta
            if "attributes" in measure and m == 0 and p == 0:
                attributes = measure["attributes"]
                divisions = attributes['divisions']
                time = attributes['time']
                beats = time['beats']
                beat_type = time['beat-type']
                fifths = attributes['key']['fifths']
                if 'mode' in attributes['key']:
                    mode = attributes['key']['mode']
                else:
                    mode = None
                if 'sign' in attributes['clef']:
                    sign = attributes['clef']['sign']
                else:
                    sign = None
                if 'line' in attributes['clef']:
                    line = attributes['clef']['line']
                else:
                    line = None
            if "direction" in measure:
                if type(measure["direction"]) is list:
                    direction = measure["direction"][0]
                else:
                    direction = measure["direction"]
                if 'metronome' in direction['direction-type']:
                    beat_unit = direction['direction-type']['metronome']['beat-unit']
                    per_minute = direction['direction-type']['metronome']['per-minute']
                    beat_unit = _note_type_to_length[beat_unit]

    if len(glob.glob(f"{submit_path}/*_tg.json")) == 0:
        print(submit_path)
    for json_path in sorted(glob.glob(f"{submit_path}/*_tg.json")):
        # if orig_musixml_path == '/home/gwx/huawei_dataprocess/data/raw/华为女声第一周（无朗读）/力度/大鱼/力度弱/女声_力度弱.musicxml':
        #     print(json_path)
        with open(json_path, "r") as f:
            word_list = json.load(f)
        score = HwScore(title=title, composer=composer, lyricist=lyricist)
        part_d = Part("Piano")
        score.append(part_d)
        measures = []
        m = Measure(time_signature=(beats, beat_type), clef=Clef(sign, int(line)),
                    key=TraditionalKeySignature(fifths, mode),
                    directions_with_displacements=[(HwMetronomeMark(beat_unit, per_minute), 0)])
        m_rest_dur = float(beats)
        for word in word_list:
            if word["word"] == "_NONE" or word["word"] == "breathe":
                note = word["note"]
                note_duration = word["note_end"][0] - word["note_start"][0]
                num_thirty_socond = extract_time(note_duration, int(per_minute))
                if num_thirty_socond == 0:
                    num_thirty_socond = 1
                note_dur = num_thirty_socond * 0.125
                while note_dur > 0:
                    if m_rest_dur >= note_dur:
                        num_thirty_socond = note_dur / 0.125
                        bin_num_thirty_socond = bin(int(num_thirty_socond))[2:]
                        bin_num_thirty_socond = bin_num_thirty_socond.zfill(6)
                        start_bin = bin_num_thirty_socond.find('1')
                        end_bin = bin_num_thirty_socond.rfind('1')
                        for i in range(start_bin, end_bin + 1):
                            if bin_num_thirty_socond[i] == '1':
                                m.append(Rest(index_to_length[i]))
                        m_rest_dur -= note_dur
                        note_dur = 0
                        if m_rest_dur == 0:
                            measures.append(m)
                            m = Measure(None)
                            m_rest_dur = float(beats)
                    else:
                        num_thirty_socond = m_rest_dur / 0.125
                        bin_num_thirty_socond = bin(int(num_thirty_socond))[2:]
                        bin_num_thirty_socond = bin_num_thirty_socond.zfill(6)
                        start_bin = bin_num_thirty_socond.find('1')
                        end_bin = bin_num_thirty_socond.rfind('1')
                        for i in range(start_bin, end_bin + 1):
                            if bin_num_thirty_socond[i] == '1':
                                m.append(Rest(index_to_length[i]))
                        note_dur -= m_rest_dur
                        m_rest_dur = float(beats)
                        measures.append(m)
                        m = Measure(None)
            else:
                # lyric = word["word"]
                try:
                    lyric = text_list[text_idx]
                    # if txt_path=='/home/gwx/huawei_dataprocess/data/raw/华为女声第一周（无朗读）/力度/大鱼/力度弱/女声_力度弱.txt':
                    #     print(lyric, word["word"])
                except:
                    print(txt_path, len(text_list), text_idx, lyric, word["word"])
                    return
                text_idx += 1
                note = word["note"]

                assert len(note) > 0
                if len(note) == 1:
                    pitch = note[0]
                    pitch = pretty_midi.note_number_to_name(pitch)
                    note_duration = word["note_end"][0] - word["note_start"][0]
                    num_thirty_socond = extract_time(note_duration, int(per_minute))
                    if num_thirty_socond == 0:
                        num_thirty_socond = 1
                    note_dur = num_thirty_socond * 0.125
                    is_start = 0
                    while note_dur > 0:
                        if m_rest_dur >= note_dur:
                            num_thirty_socond = note_dur / 0.125
                            bin_num_thirty_socond = bin(int(num_thirty_socond))[2:]
                            bin_num_thirty_socond = bin_num_thirty_socond.zfill(6)
                            start_bin = bin_num_thirty_socond.find('1')
                            end_bin = bin_num_thirty_socond.rfind('1')
                            if start_bin == end_bin and is_start == 0:
                                m.append(HwNote(pitch, index_to_length[start_bin], lyric, None))
                                is_start = 1
                            elif start_bin == end_bin and is_start == 1:
                                m.append(HwNote(pitch, index_to_length[start_bin], None, "stop"))
                            else:
                                for i in range(start_bin, end_bin + 1):
                                    if bin_num_thirty_socond[i] == '0':
                                        continue
                                    if is_start == 0:
                                        m.append(HwNote(pitch, index_to_length[i], lyric, "start"))
                                        is_start = 1
                                    elif i == end_bin:
                                        m.append(HwNote(pitch, index_to_length[i], None, "stop"))
                                    else:
                                        m.append(HwNote(pitch, index_to_length[i], None, "continue"))
                            m_rest_dur -= note_dur
                            note_dur = 0
                            if m_rest_dur == 0:
                                measures.append(m)
                                m = Measure(None)
                                m_rest_dur = float(beats)
                        else:
                            num_thirty_socond = m_rest_dur / 0.125
                            bin_num_thirty_socond = bin(int(num_thirty_socond))[2:]
                            bin_num_thirty_socond = bin_num_thirty_socond.zfill(6)
                            start_bin = bin_num_thirty_socond.find('1')
                            end_bin = bin_num_thirty_socond.rfind('1')
                            if start_bin == end_bin and is_start == 0:
                                m.append(HwNote(pitch, index_to_length[start_bin], lyric, "start"))
                                is_start = 1
                            elif start_bin == end_bin and is_start == 1:
                                m.append(HwNote(pitch, index_to_length[start_bin], None, "continue"))
                            else:
                                for i in range(start_bin, end_bin + 1):
                                    if bin_num_thirty_socond[i] == '0':
                                        continue
                                    if is_start == 0:
                                        m.append(HwNote(pitch, index_to_length[i], lyric, "start"))
                                        is_start = 1
                                    else:
                                        m.append(HwNote(pitch, index_to_length[i], None, "continue"))

                            note_dur -= m_rest_dur
                            m_rest_dur = float(beats)
                            measures.append(m)
                            m = Measure(None)
                else:

                    for idx, pitch in enumerate(note):
                        pitch = pretty_midi.note_number_to_name(pitch)
                        note_duration = word["note_end"][idx] - word["note_start"][idx]
                        num_thirty_socond = extract_time(note_duration, int(per_minute))
                        if num_thirty_socond == 0:
                            num_thirty_socond = 1
                        note_dur = num_thirty_socond * 0.125
                        is_start = 0
                        while note_dur > 0:
                            if m_rest_dur >= note_dur:
                                num_thirty_socond = note_dur / 0.125
                                bin_num_thirty_socond = bin(int(num_thirty_socond))[2:]
                                bin_num_thirty_socond = bin_num_thirty_socond.zfill(6)
                                start_bin = bin_num_thirty_socond.find('1')
                                end_bin = bin_num_thirty_socond.rfind('1')
                                if start_bin == end_bin and is_start == 0:
                                    if idx == 0:
                                        m.append(HwNote(pitch, index_to_length[start_bin], lyric, None,
                                                        notations=[StartSlur()]))
                                    elif idx == len(note) - 1:
                                        m.append(HwNote(pitch, index_to_length[start_bin], None, None,
                                                        notations=[StopSlur()]))
                                    else:
                                        m.append(HwNote(pitch, index_to_length[start_bin], lyric, None))
                                    is_start = 1

                                elif start_bin == end_bin and is_start == 1:
                                    if idx == len(note) - 1:
                                        m.append(HwNote(pitch, index_to_length[start_bin], None, "stop",
                                                        notations=[StopSlur()]))
                                    else:
                                        m.append(HwNote(pitch, index_to_length[start_bin], None, "stop"))
                                else:
                                    for i in range(start_bin, end_bin + 1):
                                        if bin_num_thirty_socond[i] == '0':
                                            continue
                                        if is_start == 0:
                                            if idx == 0:
                                                m.append(HwNote(pitch, index_to_length[i], lyric, "start",
                                                                notations=[StartSlur()]))
                                            else:
                                                m.append(HwNote(pitch, index_to_length[i], None, "start"))
                                            is_start = 1
                                        elif i == end_bin:
                                            if idx == len(note) - 1:
                                                m.append(HwNote(pitch, index_to_length[i], None, "stop",
                                                                notations=[StopSlur()]))
                                            else:
                                                m.append(HwNote(pitch, index_to_length[i], None, "stop"))
                                        else:
                                            m.append(HwNote(pitch, index_to_length[i], None, "continue"))
                                m_rest_dur -= note_dur
                                note_dur = 0
                                if m_rest_dur == 0:
                                    measures.append(m)
                                    m = Measure(None)
                                    m_rest_dur = float(beats)
                            else:
                                num_thirty_socond = m_rest_dur / 0.125
                                bin_num_thirty_socond = bin(int(num_thirty_socond))[2:]
                                bin_num_thirty_socond = bin_num_thirty_socond.zfill(6)
                                start_bin = bin_num_thirty_socond.find('1')
                                end_bin = bin_num_thirty_socond.rfind('1')
                                if start_bin == end_bin and is_start == 0:
                                    if idx == 0:
                                        m.append(HwNote(pitch, index_to_length[start_bin], lyric, "start",
                                                        notations=[StartSlur()]))
                                    else:
                                        m.append(HwNote(pitch, index_to_length[start_bin], None, "start"))
                                    is_start = 1
                                elif start_bin == end_bin and is_start == 1:
                                    m.append(HwNote(pitch, index_to_length[start_bin], None, "continue"))
                                else:
                                    for i in range(start_bin, end_bin + 1):
                                        if bin_num_thirty_socond[i] == '0':
                                            continue
                                        if is_start == 0:
                                            if idx == 0:
                                                m.append(HwNote(pitch, index_to_length[i], lyric, "start",
                                                                notations=[StartSlur()]))
                                            else:
                                                m.append(HwNote(pitch, index_to_length[i], None, "start"))
                                            is_start = 1
                                        else:
                                            m.append(HwNote(pitch, index_to_length[i], None, "continue"))

                                note_dur -= m_rest_dur
                                m_rest_dur = float(beats)
                                measures.append(m)
                                m = Measure(None)

        if m_rest_dur != float(beats):
            measures.append(m)
        part_d.extend(measures)
        score.export_to_file(json_path.replace("_tg.json", ".musicxml"))


if __name__ == '__main__':
    orig_root = "/home/gwx/huawei_dataprocess/data/raw/华为女声第一周（无朗读）"
    submit_root = "/home/gwx/huawei_dataprocess/data/submit/华为女声第一周"

    wavs_path = sorted(glob.glob(f"{orig_root}/*/*/*/*.wav"))
    for wav_fn in tqdm(wavs_path):
        txt_fn = wav_fn.replace('.wav', '.txt')
        orig_dirname = os.path.dirname(txt_fn)
        orig_musixml_path = glob.glob(f"{orig_dirname}/*.musicxml")[0]
        dir_list = wav_fn.split('/')
        submit_path = os.path.join(submit_root, dir_list[-4], dir_list[-3], dir_list[-2], dir_list[-1].split('.')[0])
        generate_mxml(submit_path, txt_fn, orig_musixml_path)



