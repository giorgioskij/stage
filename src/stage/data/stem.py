from enum import IntEnum


class Stem(IntEnum):
    DRUMS = 0
    BASS = 1
    GUITAR = 2
    KEYBOARD = 3
    PIANO = 4
    STRINGS = 5
    OTHER = 6
    VOCALS = 7
    ANY = 8

    @staticmethod
    def fromstring(s: str):
        match s:
            case "drums":
                return Stem.DRUMS
            case "drums_mixed":
                return Stem.DRUMS
            case "bass":
                return Stem.BASS
            case "guitar":
                return Stem.GUITAR
            case "other_keys":
                return Stem.KEYBOARD
            case "keyboard":
                return Stem.KEYBOARD
            case "piano":
                return Stem.PIANO
            case "bowed_strings":
                return Stem.STRINGS
            case "strings":
                return Stem.STRINGS
            case "other":
                return Stem.OTHER
            case "vocals":
                return Stem.VOCALS
            case _:
                raise ValueError("unknown stem name")

    def getname(self):
        match self:
            case Stem.DRUMS:
                return "drums_mixed"
            case Stem.BASS:
                return "bass"
            case Stem.GUITAR:
                return "guitar"
            case Stem.KEYBOARD:
                return "other_keys"
            case Stem.PIANO:
                return "piano"
            case Stem.STRINGS:
                return "bowed_strings"
            case Stem.OTHER:
                return "other"
            case Stem.VOCALS:
                return "vocals"
            case Stem.ANY:
                raise ValueError("Stem ANY has no name")
